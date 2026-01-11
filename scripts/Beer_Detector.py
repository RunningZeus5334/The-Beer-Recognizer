import cv2
import time
import torch
from ultralytics import YOLO

# --- CONFIGURATIE ---
MODEL_PATH = 'runs/detect/train_long/run1/weights/best.pt'  # Pas dit aan naar het pad van jouw getrainde model
RTSP_URL = 0  # Gebruik 0 voor webcam, of een RTSP string (bijv. "rtsp://admin:pass@192.168.1.x:554/...")

# Class ID's (Check dit in jouw model config/data.yaml!)
CLASS_ID_FACE = 1 
CLASS_ID_BEER = 0

# Instellingen voor detectie
STANDARD_CONFIDENCE_THRESHOLD = 0.5  # Standaard drempel (voor gezicht, bijv.)
BEER_CONFIDENCE_THRESHOLD = 0.35     # Lagere drempel specifiek voor bier (pas dit aan!)
OVERLAP_THRESHOLD = 0.2
TARGET_FPS = 15

# Bier instellingen
BEER_CAPACITY_CL = 30
CURRENT_LEVEL_PERCENT = 100.0
DRINK_DECAY_RATE_PER_SEC = 5.0

def calculate_intersection_ratio(box_a, box_b):
    x1_a, y1_a, x2_a, y2_a = box_a
    x1_b, y1_b, x2_b, y2_b = box_b
    x_left = max(x1_a, x1_b)
    y_top = max(y1_a, y1_b)
    x_right = min(x2_a, x2_b)
    y_bottom = min(y2_a, y2_b)

    if x_right < x_left or y_bottom < y_top: return 0.0
    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box_a_area = (x2_a - x1_a) * (y2_a - y1_a)
    if box_a_area == 0: return 0.0
    return intersection_area / box_a_area

def main():
    global CURRENT_LEVEL_PERCENT
    
    # Check device
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Model laden op: {device.upper()}...")
    
    model = YOLO(MODEL_PATH)
    
    cap = cv2.VideoCapture(RTSP_URL)
    
    prev_time = time.time()
    frame_duration = 1.0 / TARGET_FPS

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

        # Voer detectie uit met GPU
        results = model(frame, stream=True, verbose=False, device=device)

        face_box = None
        beer_box = None

        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                coords = box.xyxy[0].cpu().numpy()

                # --- AANGEPASTE CONFIDENCE CHECK ---
                current_threshold = STANDARD_CONFIDENCE_THRESHOLD
                if cls_id == CLASS_ID_BEER:
                    current_threshold = BEER_CONFIDENCE_THRESHOLD
                
                if conf < current_threshold:
                    continue
                # --- EINDE AANGEPASTE CHECK ---

                # Teken de boxen en sla coördinaten op
                if cls_id == CLASS_ID_FACE:
                    face_box = coords
                    color = (255, 0, 0)
                    label = f"Gezicht ({conf:.2f})"
                    cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), color, 2)
                    cv2.putText(frame, label, (int(coords[0]), int(coords[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                elif cls_id == CLASS_ID_BEER:
                    beer_box = coords
                    color = (0, 255, 0)
                    label = f"Bier ({conf:.2f})"
                    cv2.rectangle(frame, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), color, 2)
                    cv2.putText(frame, label, (int(coords[0]), int(coords[1])-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # ... (Rest van de Logica en UI-tekening blijft hetzelfde) ...

        # Logic
        is_drinking = False
        if face_box is not None and beer_box is not None:
            overlap = calculate_intersection_ratio(beer_box, face_box)
            cv2.putText(frame, f"Overlap: {overlap:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            if overlap > OVERLAP_THRESHOLD:
                is_drinking = True
        
        # Bier Update
        if is_drinking:
            cv2.putText(frame, "DRINKEN...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
            if CURRENT_LEVEL_PERCENT > 0:
                CURRENT_LEVEL_PERCENT -= (DRINK_DECAY_RATE_PER_SEC * dt)
                if CURRENT_LEVEL_PERCENT < 0: CURRENT_LEVEL_PERCENT = 0

        # UI Tekenen
        bar_h = 200
        fill_height = int((CURRENT_LEVEL_PERCENT / 100.0) * bar_h)
        cv2.rectangle(frame, (10, 100), (40, 300), (50, 50, 50), -1) 
        cv2.rectangle(frame, (10, 100 + bar_h - fill_height), (40, 300), (0, 165, 255), -1)
        cl_over = (CURRENT_LEVEL_PERCENT / 100.0) * BEER_CAPACITY_CL
        cv2.putText(frame, f"{int(CURRENT_LEVEL_PERCENT)}%", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{cl_over:.1f}cl", (10, 320), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # FPS Teller
        real_fps = 1 / dt if dt > 0 else 0
        cv2.putText(frame, f"FPS: {real_fps:.1f}", (frame.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        cv2.imshow('Bier Monitor GPU', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'): break
        elif key == ord('r'): CURRENT_LEVEL_PERCENT = 100.0

        # FPS Limiter
        process_time = time.time() - loop_start_time
        wait_time = frame_duration - process_time
        if wait_time > 0:
            time.sleep(wait_time)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()