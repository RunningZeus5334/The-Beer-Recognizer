# Pico 2 W Fuel Gauge Servo Controller

Project Overview
----------------
This project implements a fuel gauge using a Raspberry Pi Pico 2 W and a micro servo motor (SG90 / S90G).
The servo represents a fuel gauge needle and is controlled through USB serial input from a terminal.

The system allows numeric positioning as well as named fuel states.
The onboard Pico LED is used as a status indicator to show when the program is running and when commands are received.

The design avoids Wi-Fi and networking to ensure reliability and simplicity.

Hardware Used
-------------
- Raspberry Pi Pico 2 W
- SG90 / S90G micro servo motor
- External 5V power supply for the servo (recommended)
- USB cable between Pico and PC

Wiring
------
Servo signal wire  -> Pico GP0
Servo ground wire  -> Pico GND
Servo power wire   -> External 5V (ground must be shared with Pico)
Status LED         -> Pico onboard LED

Do not power the servo from the Pico 3.3V pin.

Fuel Gauge Configuration
------------------------
The servo is limited to a fuel gauge range instead of the full 180 degrees.

Fuel state mapping:
FULL      = 0
HALF FULL = 83
EMPTY     = 170

Rules:
- Valid range is 0 to 170
- Values outside this range are rejected
- Limits are enforced in firmware to protect the mechanism

Serial Control Interface
------------------------
Commands are entered through the USB serial terminal.

Numeric Positions
-----------------
0
45
120
170

Rules:
- Must be between 0 and 170
- Values outside this range are rejected with an error message

Named Fuel States
-----------------
FULL
HALFFULL
MID
EMPTY

Utility Commands
----------------
HELP
STATUS

User Feedback
-------------
Terminal output:
- Confirms every valid command
- Displays a prompt when the gauge reaches FULL, HALF FULL, or EMPTY

Onboard LED behavior:
- LED ON continuously when the program is running
- LED blinks briefly when a valid command is received
- LED OFF when the program stops or crashes

Software Environment
--------------------
Pico:
- MicroPython
- Runs as main.py
- Uses non-blocking serial input
- Compatible with VS Code MicroPico, Thonny, and standard serial terminals

PC:
- No custom host software required
- Commands are typed directly into the terminal

Design Decisions
----------------
- USB serial used instead of Wi-Fi for reliability
- Non-blocking input to avoid REPL issues
- Strict gauge limits to protect hardware
- Named fuel states for clarity
- Clean shutdown handling (LED off, PWM released)

Current Status
--------------
Servo control working
Fuel gauge limits enforced
Named and numeric commands supported
Onboard LED status implemented
Clean startup and shutdown behavior

Possible Future Improvements
----------------------------
- Smooth needle movement
- Calibration mode
- Saving last position to flash memory
- Graphical PC interface
- Multiple gauge support