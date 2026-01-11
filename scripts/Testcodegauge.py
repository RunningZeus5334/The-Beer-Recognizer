from machine import Pin, PWM
import time
import sys
import uselect

# ---------------- Onboard LED ----------------
led = Pin("LED", Pin.OUT)

def blink_ok(duration=0.08):
    led.off()
    time.sleep(duration)
    led.on()

# ---------------- Fuel Gauge Positions ----------------
FUEL_POSITIONS = {
    "FULL":  0,
    "HALFFULL": 83,
    "MID":   83,
    "EMPTY": 170,
}

FULL_DEG = 0
HALF_DEG = 83
EMPTY_DEG = 170

GAUGE_MIN_DEG = 0
GAUGE_MAX_DEG = 170

# ---------------- Servo Config ----------------
SERVO_GPIO = 0
PWM_FREQ = 50
MIN_US = 500
MAX_US = 2500


class Servo:
    def __init__(self, gpio: int):
        self.pwm = PWM(Pin(gpio))
        self.pwm.freq(PWM_FREQ)
        self.period_us = int(1_000_000 / PWM_FREQ)

    def _us_to_duty_u16(self, us: float) -> int:
        us = max(0, min(us, self.period_us))
        return int(us * 65535 / self.period_us)

    def write_angle(self, deg: float):
        deg = max(0.0, min(180.0, float(deg)))
        us = MIN_US + (deg / 180.0) * (MAX_US - MIN_US)
        self.pwm.duty_u16(self._us_to_duty_u16(us))


def state_prompt(angle: int):
    if angle == FULL_DEG:
        print("PROMPT: Fuel state = FULL (0)")
    elif angle == HALF_DEG:
        print("PROMPT: Fuel state = HALF FULL (83)")
    elif angle == EMPTY_DEG:
        print("PROMPT: Fuel state = EMPTY (170)")


def apply_numeric(angle: float, servo: Servo):
    if angle < GAUGE_MIN_DEG or angle > GAUGE_MAX_DEG:
        print(f"ERR: Out of range. Enter {GAUGE_MIN_DEG}..{GAUGE_MAX_DEG}.")
        return

    angle = int(angle)
    servo.write_angle(angle)
    print(f"OK: Set position -> {angle}")
    state_prompt(angle)
    blink_ok()


def apply_command(cmd: str, servo: Servo):
    cmd = cmd.strip()
    if not cmd:
        return

    up = cmd.upper().replace(" ", "")

    if up in ("H", "HELP", "?"):
        print("Commands:")
        print("  <number>     -> set position (0..170)")
        print("  FULL         -> set 0")
        print("  HALFFULL     -> set 83")
        print("  MID          -> set 83")
        print("  EMPTY        -> set 170")
        print("  STATUS       -> show config")
        return

    if up == "STATUS":
        print("Gauge range:", GAUGE_MIN_DEG, "..", GAUGE_MAX_DEG)
        print("Key positions: FULL=0, HALF=83, EMPTY=170")
        print("Servo GPIO:", SERVO_GPIO, "PWM:", PWM_FREQ, "Hz")
        return

    if up in FUEL_POSITIONS:
        angle = int(FUEL_POSITIONS[up])
        servo.write_angle(angle)
        print(f"OK: {up} -> {angle}")
        state_prompt(angle)
        blink_ok()
        return

    try:
        angle = float(cmd)
    except ValueError:
        print("ERR: Unknown command. Type HELP.")
        return

    apply_numeric(angle, servo)


def main():
    servo = Servo(SERVO_GPIO)

    # LED ON = program running
    led.on()

    # Optional boot position
    servo.write_angle(HALF_DEG)

    print("READY.")
    print("Enter a position 0..170, or FULL / HALFFULL / EMPTY. Type HELP for commands.")

    poller = uselect.poll()
    poller.register(sys.stdin, uselect.POLLIN)

    line_buf = ""

    try:
        while True:
            events = poller.poll(100)
            if events:
                ch = sys.stdin.read(1)
                if not ch:
                    time.sleep(0.01)
                    continue

                if ch == "\n":
                    apply_command(line_buf, servo)
                    line_buf = ""
                elif ch != "\r":
                    line_buf += ch

            time.sleep(0.01)

    finally:
        # This ALWAYS runs when the program stops
        led.off()
        try:
            servo.pwm.deinit()
        except Exception:
            pass
        print("STOPPED: LED off, servo released.")


main()