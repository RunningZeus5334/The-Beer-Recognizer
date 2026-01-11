import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

try:
    import serial
    from serial.tools import list_ports
except Exception:
    serial = None
    list_ports = None

# --- CONFIG ---
DEFAULT_MODEL_PATH = Path("runs/detect/train_long/run1/weights/best.pt")
RTSP_URL = 0  # 0 = webcam

CLASS_ID_FACE = 1
CLASS_ID_BEER = 0

STANDARD_CONFIDENCE_THRESHOLD = 0.5
BEER_CONFIDENCE_THRESHOLD = 0.35
OVERLAP_THRESHOLD = 0.2
TARGET_FPS = 15

BEER_CAPACITY_CL = 30
CURRENT_LEVEL_PERCENT = 100.0
DRINK_DECAY_RATE_PER_SEC = 5.0

SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 1.0

WINDOW_NAME = "Bier Monitor GPU"

pico = None           # PicoController | None
pico_connected = False

WIDGET_RECT = None    # (x1, y1, x2, y2) for Pico button
RESET_RECT = None     # (x1, y1, x2, y2) for Reset button


def auto_detect_serial_port() -> str | None:
    if list_ports is None:
        print("pyserial niet aanwezig, auto-detect serial port niet mogelijk.")
        return None

    ports = list(list_ports.comports())
    if not ports:
        print("Geen seriële poorten gevonden.")
        return None

    candidates = []
    for p in ports:
        dev = p.device
        if dev.startswith("/dev/ttyACM") or dev.startswith("/dev/ttyUSB") or dev.startswith("COM"):
            candidates.append(p)

    if not candidates:
        print("Geen typische USB-serial poort; neem eerste als fallback.")
        candidates = ports

    chosen = candidates[0]
    print(f"Auto-detect serial port: {chosen.device} ({chosen.description})")
    return chosen.device


class PicoController:
    def __init__(self, port=None, baud=SERIAL_BAUD, timeout=SERIAL_TIMEOUT):
        self.port = port
        self.baud = baud
        self.timeout = timeout
        self.ser = None

    def open(self):
        if serial is None:
            print("pyserial niet beschikbaar; Pico aansturing uitgeschakeld.")
            return False

        if self.ser is not None and self.ser.is_open:
            return True

        if self.port is None:
            self.port = auto_detect_serial_port()
            if self.port is None:
                print("Geen seriële poort gevonden; Pico aansturing uitgeschakeld.")
                return False

        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=self.timeout)
            time.sleep(1.0)
            print(f"Verbonden met Pico op {self.port} @ {self.baud}")
            return True
        except Exception as e:
            print(f"Kon serial-poort niet openen ({self.port}): {e}")
            self.ser = None
            return False

    def send_angle(self, angle: float) -> bool:
        if self.ser is None:
            return False
        try:
            cmd = f"{int(round(angle))}\n"
            self.ser.write(cmd.encode("utf-8"))
            return True
        except Exception as e:
            print("Fout bij schrijven naar serial:", e)
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            return False

    def close(self):
        if self.ser:
            try:
                self.ser.close()
            except Exception:
                pass
            self.ser = None
            print("Pico verbinding gesloten.")


def percent_to_angle(percent: float) -> float:
    """
    Map 0..100% -> 170..0 graden (inverted):
    - 100% (vol)  -> 0°
    - 0%   (leeg) -> 170°
    """
    p = max(0.0, min(100.0, float(percent)))
    angle_full = 0.0      # angle bij 100%
    angle_empty = 170.0   # angle bij 0%
    return angle_empty - (p / 100.0) * (angle_empty - angle_full)


def calculate_intersection_ratio(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)

    if x_right < x_left or y_bottom < y_top:
        return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = (x2_a - x1_a) * (y2_a - y1_a)
    if box_a_area == 0:
        return 0.0
    return intersection_area / box_a_area


def find_model_path() -> Path:
    script_path = Path(__file__).resolve()
    repo_root = script_path.parent.parent
    runs_dir = repo_root / "runs" / "detect"

    candidates = []
    if runs_dir.exists():
        for p in runs_dir.rglob("best.pt"):
            if p.parent.name == "weights":
                candidates.append(p)

    if candidates:
        candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        chosen = candidates[0]
        print(f"Auto-gevonden model: {chosen}")
        return chosen

    if DEFAULT_MODEL_PATH.is_file():
        print(f"Gebruik default modelpad: {DEFAULT_MODEL_PATH}")
        return DEFAULT_MODEL_PATH

    raise FileNotFoundError(
        f"Geen best.pt gevonden onder {runs_dir} en {DEFAULT_MODEL_PATH} bestaat niet.\n"
        f"Train eerst een model of pas DEFAULT_MODEL_PATH aan."
    )


def mouse_callback(event, x, y, flags, param):
    global pico, pico_connected, WIDGET_RECT, RESET_RECT, CURRENT_LEVEL_PERCENT

    if event != cv2.EVENT_LBUTTONDOWN:
        return

    # Pico connect/disconnect
    if WIDGET_RECT is not None:
        x1, y1, x2, y2 = WIDGET_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            if not pico_connected:
                if pico is None:
                    pico = PicoController(port=None)
                if pico.open():
                    pico_connected = True
            else:
                if pico is not None:
                    pico.close()
                pico_connected = False
            return

    # Reset beer level
    if RESET_RECT is not None:
        x1, y1, x2, y2 = RESET_RECT
        if x1 <= x <= x2 and y1 <= y <= y2:
            CURRENT_LEVEL_PERCENT = 100.0
            print("Beer level reset to 100% via GUI.")
            return


def draw_connect_widget(frame):
    """Teken 'Connect Pico' / 'Disconnect' + 'Reset Beer' blokjes rechtsboven, onder elkaar."""
    global WIDGET_RECT, RESET_RECT, pico_connected

    h, w = frame.shape[:2]
    btn_w, btn_h = 180, 40
    margin = 10
    spacing = 10  # verticale tussenruimte tussen knoppen

    # Pico button (rechtsboven)
    x2 = w - margin
    x1 = x2 - btn_w
    y1 = margin
    y2 = y1 + btn_h
    WIDGET_RECT = (x1, y1, x2, y2)

    color_bg = (0, 120, 0) if pico_connected else (0, 0, 120)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color_bg, -1)
    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 1)

    text = "Disconnect Pico" if pico_connected else "Connect Pico"
    cv2.putText(frame, text, (x1 + 8, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # Reset button, ONDER de Pico button
    rx1 = x1
    rx2 = x2
    ry1 = y2 + spacing
    ry2 = ry1 + btn_h
    RESET_RECT = (rx1, ry1, rx2, ry2)

    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (120, 0, 0), -1)
    cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 255), 1)

    rtext = "Reset Beer"
    cv2.putText(frame, rtext, (rx1 + 20, ry1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def main():
    global CURRENT_LEVEL_PERCENT, pico, pico_connected

    device = "0" if torch.cuda.is_available() else "cpu"
    print(f"Model laden op: {device.upper()}...")

    model_path = find_model_path()
    model = YOLO(str(model_path))
    print(f"Model geladen van: {model_path}")

    cap = cv2.VideoCapture(RTSP_URL)

    prev_time = time.time()
    frame_duration = 1.0 / TARGET_FPS

    pico = PicoController(port=None)
    pico_connected = False

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

    last_sent_angle = None
    last_send_time = 0.0
    SEND_INTERVAL = 1.0
    ANGLE_DELTA_THRESHOLD = 1.0

    print(f"Start detectie op max {TARGET_FPS} FPS. Druk op 'q' om te stoppen.")

    while True:
        loop_start_time = time.time()

        ret, frame = cap.read()
        if not ret:
            print("Geen beeld meer.")
            break

        current_time = time.time()
        dt = current_time - prev_time
        prev_time = current_time

        results = model(frame, stream=True, verbose=False, device=device)

        face_box = None
        beer_box = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy()

                current_threshold = STANDARD_CONFIDENCE_THRESHOLD
                if cls_id == CLASS_ID_BEER:
                    current_threshold = BEER_CONFIDENCE_THRESHOLD

                if conf < current_threshold:
                    continue

                if cls_id == CLASS_ID_FACE:
                    face_box = coords
                    color = (255, 0, 0)
                    label = f"Gezicht ({conf:.2f})"
                    cv2.rectangle(frame, (int(coords[0]), int(coords[1])),
                                  (int(coords[2]), int(coords[3])), color, 2)
                    cv2.putText(frame, label, (int(coords[0]), int(coords[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                elif cls_id == CLASS_ID_BEER:
                    beer_box = coords
                    color = (0, 255, 0)
                    label = f"Bier ({conf:.2f})"
                    cv2.rectangle(frame, (int(coords[0]), int(coords[1])),
                                  (int(coords[2]), int(coords[3])), color, 2)
                    cv2.putText(frame, label, (int(coords[0]), int(coords[1]) - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        is_drinking = False
        if face_box is not None and beer_box is not None:
            overlap = calculate_intersection_ratio(beer_box, face_box)
            cv2.putText(frame, f"Overlap: {overlap:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if overlap > OVERLAP_THRESHOLD:
                is_drinking = True

        if is_drinking:
            cv2.putText(frame, "DRINKEN...", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if CURRENT_LEVEL_PERCENT > 0:
                CURRENT_LEVEL_PERCENT -= (DRINK_DECAY_RATE_PER_SEC * dt)
                if CURRENT_LEVEL_PERCENT < 0:
                    CURRENT_LEVEL_PERCENT = 0

        # Pico aansturen alleen als verbonden
        if pico_connected and pico is not None:
            try:
                desired_angle = percent_to_angle(CURRENT_LEVEL_PERCENT)
                now_t = time.time()
                need_send = False
                if last_sent_angle is None:
                    need_send = True
                elif abs(desired_angle - last_sent_angle) >= ANGLE_DELTA_THRESHOLD:
                    need_send = True
                elif (now_t - last_send_time) >= SEND_INTERVAL:
                    need_send = True

                if need_send:
                    sent = pico.send_angle(desired_angle)
                    if sent:
                        last_sent_angle = desired_angle
                        last_send_time = now_t
            except Exception as e:
                print("Pico send error:", e)

        bar_h = 200
        fill_height = int((CURRENT_LEVEL_PERCENT / 100.0) * bar_h)
        cv2.rectangle(frame, (10, 100), (40, 300), (50, 50, 50), -1)
        cv2.rectangle(frame, (10, 100 + bar_h - fill_height),
                      (40, 300), (0, 165, 255), -1)
        cl_over = (CURRENT_LEVEL_PERCENT / 100.0) * BEER_CAPACITY_CL
        cv2.putText(frame, f"{int(CURRENT_LEVEL_PERCENT)}%", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{cl_over:.1f}cl", (10, 320),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        real_fps = 1 / dt if dt > 0 else 0
        cv2.putText(frame, f"FPS: {real_fps:.1f}",
                    (frame.shape[1] - 120, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        draw_connect_widget(frame)

        cv2.imshow(WINDOW_NAME, frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

        process_time = time.time() - loop_start_time
        wait_time = frame_duration - process_time
        if wait_time > 0:
            time.sleep(wait_time)

    cap.release()
    if pico is not None:
        pico.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()