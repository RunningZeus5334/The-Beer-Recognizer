# The Beer Recognizer - Complete Setup and Usage Guide

## Table of Contents
1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Complete Setup Instructions](#complete-setup-instructions)
4. [How It Works](#how-it-works)
5. [Configuration Guide](#configuration-guide)
6. [Running the Application](#running-the-application)
7. [Troubleshooting](#troubleshooting)
8. [Project Structure](#project-structure)
9. [Hardware Setup](#hardware-setup)

---

## Overview

**The Beer Recognizer** is a proof-of-concept project that combines computer vision, embedded systems, and IoT to create an interactive beer monitoring system.

### What It Does
1. **Detects objects** in a video stream (webcam or RTSP) using a trained YOLO model
2. **Identifies face and beer** in the video frame
3. **Calculates drinking behavior** by detecting overlap between face and beer regions
4. **Tracks beer level** - decreases as drinking is detected
5. **Controls a servo motor** on a Raspberry Pi Pico that acts as a fuel gauge needle
6. **Displays results** in a GUI with real-time visualizations

### Technology Stack
- **Computer Vision:** YOLO (Ultralytics) for object detection
- **Machine Learning:** PyTorch for GPU acceleration
- **Video Processing:** OpenCV for frame capture and manipulation
- **Hardware Communication:** PySerial for USB communication with Pico
- **Embedded System:** Raspberry Pi Pico 2 W with MicroPython
- **Servo Control:** PWM-based servo positioning

---

## System Requirements

### Minimum Requirements
- **OS:** Linux, Windows, or macOS
- **Python:** 3.10 or higher
- **GPU:** NVIDIA GPU with CUDA support (optional but recommended)
- **RAM:** 8GB minimum (16GB+ recommended)
- **Disk Space:** 2GB for dependencies and models

### Hardware Requirements
- **Webcam:** USB webcam or built-in camera
- **Raspberry Pi Pico 2 W** (or compatible MicroPython board)
- **SG90/S90G Micro Servo Motor**
- **Servo Power Supply:** 5V external power (do NOT use Pico's 3.3V)
- **USB Cable:** For Pico connection
- **Jumper Wires:** For servo connections
- **3D Printed Parts:** (Optional) Gauge housing from 3Dmodel/faceplatemetnummerscombined.3mf

### Optional Hardware
- RTSP camera instead of USB webcam
- External display for GUI
- Prototyping breadboard for servo connections

---

## Complete Setup Instructions

### Step 1: Clone the Repository

```bash
git clone https://github.com/SamyPC/The-Beer-Recognizer.git
cd The-Beer-Recognizer
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

# Windows
python -m venv .venv
.venv\Scripts\activate
```

### Step 3: Install Python Dependencies

```bash
pip install -r requirements.txt
```

This installs:
- **torch** & **torchvision** - Deep learning framework with GPU support
- **ultralytics** - YOLO implementation
- **opencv-python** - Computer vision library
- **pyserial** - Serial communication with Pico
- **labelme** - Annotation tool (for dataset labeling)
- **numpy, matplotlib, pandas** - Data handling and visualization

### Step 4: Verify GPU Installation (if applicable)

```python
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU only')"
```

**Expected output:**
- With GPU: `CUDA available: True` + GPU name
- Without GPU: `CUDA available: False`

### Step 5: Prepare Raspberry Pi Pico

#### 5.1 Flash MicroPython (if not already done)

1. Download MicroPython UF2 file from https://micropython.org/download/rp2-pico/
2. Connect Pico to computer via USB while holding BOOTSEL button
3. A USB drive (RPI-RP2) will appear
4. Copy the UF2 file to the drive
5. Pico will reboot with MicroPython installed

#### 5.2 Upload Servo Control Script

**Option A: Using Thonny (Recommended)**
1. Install Thonny: https://thonny.org/
2. Select your Pico as the interpreter
3. Open `scripts/Testcodegauge.py`
4. Click "Save to Pico"

**Option B: Using mpremote**
```bash
pip install mpremote
mpremote connect /dev/ttyACM0 cp scripts/Testcodegauge.py :/main.py
```

**Option C: Using ampy**
```bash
pip install adafruit-ampy
ampy --port /dev/ttyACM0 put scripts/Testcodegauge.py main.py
```

### Step 6: Connect Hardware

```
Servo Connections (to Pico):
  Signal wire → GPIO 0 (GP0)
  Ground wire → GND (any GND pin)
  Power wire → External 5V power supply
  
IMPORTANT: Do NOT power servo from Pico's 3.3V pin
           Use external 5V supply with shared ground
```

**Wiring Diagram:**
```
Raspberry Pi Pico
┌─────────────────┐
│ GP0 ─────→ Signal (Orange/Yellow)
│ GND ─────→ Ground (Black) ──┐
│                             │
│                    ┌────────┴──────┐
│                    │              │
│               External 5V       Servo
│               Power Supply     (SG90)
│                    │              │
│                    └──────────────┘
```

### Step 7: Detect Serial Port

**Linux/macOS:**
```bash
ls /dev/ttyACM* /dev/ttyUSB*
# Pico typically appears as /dev/ttyACM0
```

**Windows:**
```powershell
Get-WmiObject Win32_SerialPort | Select-Object Name, DeviceID
# Pico typically appears as COM3, COM4, etc.
```

**Python (All platforms):**
```python
from serial.tools import list_ports
for port in list_ports.comports():
    print(f"{port.device}: {port.description}")
```

### Step 8: Verify Everything Works

```bash
# Test that Pico accepts commands
python3 << 'EOF'
import serial
import time

port = "/dev/ttyACM0"  # Change to your port
try:
    ser = serial.Serial(port, 115200, timeout=1)
    time.sleep(2)
    ser.write(b"90\n")  # Test: move servo to 90 degrees
    print("✓ Servo command sent successfully")
    ser.close()
except Exception as e:
    print(f"✗ Error: {e}")
EOF
```

---

## How It Works

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     HOST COMPUTER                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐         ┌─────────────────┐               │
│  │   Webcam     │────────▶│  OpenCV Video   │               │
│  │   (USB)      │         │   Capture       │               │
│  └──────────────┘         └────────┬────────┘               │
│                                    │                        │
│                            ┌───────▼────────┐               │
│                            │   YOLO Model   │               │
│                            │   (Inference)  │               │
│                            └───────┬────────┘               │
│                                    │                        │
│                            ┌───────▼────────┐               │
│                            │ Detection      │               │
│                            │ Processing:    │               │
│                            │ - Find faces   │               │
│                            │ - Find beers   │               │
│                            │ - Calculate    │               │
│                            │   overlap      │               │
│                            └───────┬────────┘               │
│                                    │                        │
│                            ┌───────▼────────┐               │
│                            │  Level Logic   │               │
│                            │ - Track %      │               │
│                            │ - Decay when   │               │
│                            │   drinking     │               │
│                            └───────┬────────┘               │
│                                    │                        │
│                            ┌───────▼────────┐               │
│                            │  Map % to      │               │
│                            │  Angle         │               │
│                            │  (0-170°)      │               │
│                            └───────┬────────┘               │
│                                    │                        │
│      ┌─────────────────────────────┴──────────────────┐    │
│      │              Serial (USB)                      │    │
│      └──────────────────────────────────────────────┬─┘    │
│                                                      │      │
└──────────────────────────────────────────────────────┼──────┘
                                                       │
                ┌──────────────────────────────────────┘
                │
                ▼
        ┌──────────────────┐
        │ Raspberry Pi     │
        │ Pico 2 W         │
        │                  │
        │ ┌──────────────┐ │
        │ │ MicroPython  │ │
        │ │ main.py      │ │
        │ │              │ │
        │ │ Reads angle  │ │
        │ │ from serial  │ │
        │ │              │ │
        │ │ PWM control  │ │
        │ └──────────────┘ │
        │                  │
        │ GPIO0 ──────────▶ Servo
        └──────────────────┘
```

### Detection Logic

#### 1. Object Detection (YOLO)
- Model processes each frame from webcam
- Outputs bounding boxes for detected objects
- Classes: `0 = beer`, `1 = face`
- Returns: box coordinates, confidence score, class ID

```
Input Frame (640x480)
        │
        ▼
  ┌─────────────┐
  │ YOLO Model  │
  └─────────────┘
        │
        ├─▶ Face detection: (x1, y1, x2, y2, conf=0.92)
        └─▶ Beer detection: (x1, y1, x2, y2, conf=0.87)
```

#### 2. Overlap Detection
- Calculates intersection area between beer and face bounding boxes
- Formula: `overlap_ratio = intersection_area / beer_box_area`
- If `overlap_ratio > OVERLAP_THRESHOLD` (0.2 by default), user is drinking

```
Face Box          Beer Box
┌─────────┐      ┌──────────┐
│         │      │ ┌──────┐ │
│ ┌──────┤◀─────▶│ │████  │ │  ◀── Intersection
│ │████  │      │ └──────┘ │
│ └──────┤      └──────────┘
│         │
└─────────┘

Overlap Ratio = (████ area) / (Total Beer Area) = 0.45
                → 0.45 > 0.2 → DRINKING!
```

#### 3. Level Tracking
- Starts at 100% (full beer)
- Decreases by `DRINK_DECAY_RATE_PER_SEC * elapsed_time` when drinking
- Minimum: 0%, Maximum: 100%

```python
# Pseudocode
if is_drinking and CURRENT_LEVEL_PERCENT > 0:
    CURRENT_LEVEL_PERCENT -= DRINK_DECAY_RATE_PER_SEC * delta_time
```

#### 4. Angle Mapping
- Converts percentage (0-100%) to servo angle (0-170°)
- Inverted mapping: 100% = 0° (full), 0% = 170° (empty)
- Formula: `angle = 170 - (percent / 100) * 170`

```
100% ─── 0°    (Full - needle at top)
 50% ─── 85°   (Half - needle at middle)
  0% ─── 170°  (Empty - needle at bottom)
```

#### 5. Serial Communication
- Sends angle as integer + newline: `"90\n"`
- Rate-limited: only sends if:
  - Angle change > `ANGLE_DELTA_THRESHOLD` (1° default), OR
  - Time since last send > `SEND_INTERVAL` (1 second default)
- Baud rate: 115200

```
Host ──[Serial]─▶ Pico
          │
          └─▶ "90\n"  (move to 90°)
             "0\n"   (move to 0°)
             "170\n" (move to 170°)
```

---

## Configuration Guide

### Beer_Detector.py Configuration

Located at the top of `scripts/Beer_Detector.py`, these variables control behavior:

```python
# Model & Input
DEFAULT_MODEL_PATH = Path("runs/detect/train_long/run1/weights/best.pt")
RTSP_URL = 0  # 0 = webcam, or "rtsp://..." for network camera

# Detection Classes
CLASS_ID_FACE = 1      # Model class ID for faces
CLASS_ID_BEER = 0      # Model class ID for beer

# Confidence Thresholds (0.0-1.0)
STANDARD_CONFIDENCE_THRESHOLD = 0.5   # For faces
BEER_CONFIDENCE_THRESHOLD = 0.35      # For beer (lower = more sensitive)

# Detection Logic
OVERLAP_THRESHOLD = 0.2      # Overlap required to detect drinking (0.0-1.0)
BEER_CAPACITY_CL = 30        # Beer capacity in centiliters (visual only)

# Level Tracking
CURRENT_LEVEL_PERCENT = 100.0        # Initial level
DRINK_DECAY_RATE_PER_SEC = 5.0      # % per second when drinking

# Serial Communication
SERIAL_BAUD = 115200
SERIAL_TIMEOUT = 1.0
# SERIAL_PORT = "/dev/ttyACM0"  # Auto-detected, can be set manually

# Display
WINDOW_NAME = "Bier Monitor GPU"
TARGET_FPS = 15
```

### Tuning Guide

**Increase Detection Sensitivity:**
```python
BEER_CONFIDENCE_THRESHOLD = 0.25   # Detect more beers
OVERLAP_THRESHOLD = 0.1            # Easier to trigger drinking
```

**Decrease False Positives:**
```python
STANDARD_CONFIDENCE_THRESHOLD = 0.6  # Only very confident faces
OVERLAP_THRESHOLD = 0.4              # Require more overlap
```

**Adjust Drinking Speed:**
```python
DRINK_DECAY_RATE_PER_SEC = 10.0   # Drinks faster
DRINK_DECAY_RATE_PER_SEC = 2.0    # Drinks slower
```

**Better Performance:**
```python
TARGET_FPS = 30          # Faster processing (needs stronger GPU)
SEND_INTERVAL = 0.5      # More frequent servo updates
```

---

## Running the Application

### Step 1: Start the Pico Script
Ensure `scripts/Testcodegauge.py` is uploaded and running on Pico.
You should see terminal output: `READY.` and `Enter a position 0..170...`

### Step 2: Run Beer_Detector

```bash
python scripts/Beer_Detector.py
```

### Step 3: Connect Hardware

In the GUI window that appears:
1. Click **"Connect Pico"** button (top right, blue)
   - Button turns green when connected
   - Status appears in console
2. Wait for successful connection message

### Step 4: Position Your Beer

1. Hold a beer bottle/glass in front of the webcam
2. Position your face so the camera can see both you and the beer
3. Watch the GUI for detection indicators:
   - Green box = beer detection
   - Blue box = face detection
   - Yellow text = overlap percentage
   - "DRINKEN..." text when drinking detected

### Step 5: Drink!

1. Bring beer to mouth (overlap increases)
2. Level percentage decreases in real-time
3. Servo moves to reflect current level
4. Watch gauge needle move as you drink!

### Controls

**Keyboard:**
- `q` - Quit application
- `r` - Reset beer level to 100%

**Mouse:**
- Click **"Connect Pico"** - Toggle Pico connection
- Click **"Reset Beer"** - Reset level to 100%

---

## Troubleshooting

### Issue: No Detections

**Symptom:** Boxes not appearing around beer/face

**Solutions:**
1. Check model loaded correctly - console should show model path
2. Lower confidence thresholds:
   ```python
   BEER_CONFIDENCE_THRESHOLD = 0.25
   STANDARD_CONFIDENCE_THRESHOLD = 0.4
   ```
3. Improve lighting - YOLO works better in bright conditions
4. Train model with more diverse data
5. Test model separately:
   ```python
   from ultralytics import YOLO
   model = YOLO("runs/detect/train_long/run1/weights/best.pt")
   results = model("test_image.jpg")
   ```

### Issue: Webcam Not Accessible

**Symptom:** `Cannot open webcam` or black video window

**Solutions:**
1. Check device is connected: `ls /dev/video*` (Linux)
2. Grant permissions:
   ```bash
   sudo usermod -a -G video $USER
   ```
3. Test OpenCV:
   ```python
   import cv2
   cap = cv2.VideoCapture(0)
   ret, frame = cap.read()
   print("Webcam works!" if ret else "No frames captured")
   ```
4. Try different camera index: `cv2.VideoCapture(1)` or `cv2.VideoCapture(2)`

### Issue: Serial Connection Failed

**Symptom:** Can't connect to Pico, red connection button

**Solutions:**
1. Verify Pico is connected:
   ```bash
   # Linux/macOS
   ls /dev/ttyACM*
   
   # Windows
   Get-WmiObject Win32_SerialPort | Select-Object Name
   ```
2. Check Pico script is running (you should see prompt in Pico terminal)
3. Manual port specification in code:
   ```python
   SERIAL_PORT = "/dev/ttyACM0"  # Change to your port
   ```
4. Test serial communication:
   ```python
   import serial
   ser = serial.Serial("/dev/ttyACM0", 115200, timeout=1)
   ser.write(b"90\n")  # Send angle
   ser.close()
   ```

### Issue: Servo Doesn't Move

**Symptom:** GUI connects but servo unresponsive

**Solutions:**
1. Check wiring:
   - Signal → GPIO 0
   - Ground → GND (Pico and power supply)
   - Power → External 5V
2. Test servo directly:
   ```python
   from machine import Pin, PWM
   pwm = PWM(Pin(0))
   pwm.freq(50)
   pwm.duty_u16(int(500/20000 * 65535))  # Test position
   ```
3. Verify Pico script receives commands - check Pico terminal output
4. Check servo isn't damaged - try with different servo if possible

### Issue: Slow Performance

**Symptom:** Low FPS, laggy GUI, delays

**Solutions:**
1. Reduce resolution:
   ```python
   TARGET_FPS = 10  # Lower target FPS
   ```
2. Use GPU:
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. Use lighter model: switch to YOLOv8n instead of YOLOv8s
4. Check system resources:
   ```bash
   nvidia-smi  # Check GPU usage
   top         # Check CPU usage
   ```

---

## Project Structure

```
The-Beer-Recognizer/
│
├── scripts/
│   ├── Beer_Detector.py           # Main application (HOST)
│   ├── Testcodegauge.py           # Servo controller (PICO)
│   ├── labelme_to_yolo.py         # LabelMe annotation converter
│   ├── prepare_yolo_dataset.py    # Dataset preparation tool
│   └── open_next_unlabeled.py     # Labeling helper
│
├── runs/
│   └── detect/
│       ├── train/                 # Quick training run
│       ├── train_long/
│       │   └── run1/
│       │       ├── weights/       # Trained models
│       │       │   ├── best.pt    # Best model (use this!)
│       │       │   └── last.pt
│       │       └── results.csv    # Training metrics
│       └── train2/                # Alternative training
│
├── dataset/
│   └── data.yaml                  # Dataset configuration
│       images/
│       │   ├── train/            # Training images
│       │   └── val/              # Validation images
│       └── labels/               # YOLO annotations (.txt)
│
├── 3Dmodel/
│   └── faceplatemetnummerscombined.3mf  # Gauge housing
│
├── documentation/
│   ├── ServounitDocumentation.md       # Servo details
│   └── SETUP_AND_USAGE_GUIDE.md        # This file
│
├── README.md                       # Project overview
├── requirements.txt                # Python dependencies
└── .gitignore
```

---

## Hardware Setup

### Raspberry Pi Pico 2 W Pinout

```
PICO TOP VIEW:
┌─────────────────────────┐
│  GP0 (Servo Signal) ─→ │  ← Important!
│  GP1                    │
│  GND ─→ (Servo Ground)  │  ← Shared with servo power
│  ...                    │
│  VSYS (5V Input) ────→  │  ← For servo power (via external supply)
│  GND ────────────────→  │  ← Servo power return
└─────────────────────────┘
```

### Servo Wiring (SG90/S90G)

```
Servo Connector (3 wires):
  Orange/Yellow → Signal (GPIO 0)
  Red          → Power (External 5V)
  Brown/Black  → Ground (GND)
```

### Power Supply Requirements

**External 5V Power Supply:**
- Voltage: 5V DC
- Current: 1-2A minimum (servo can draw 500mA+ under load)
- GND must be connected to Pico GND

**Bad Setup (Will Fail):**
```
❌ Servo power from Pico 3.3V
❌ Servo power from USB power
❌ Servo ground NOT connected to Pico GND
```

**Good Setup (Will Work):**
```
External Power Supply (5V)
       │
       ├─▶ Red wire → Servo power
       │
       ├─▶ Black wire → [Shared GND]
       │                    │
       │                    ├─▶ Pico GND
       │                    └─▶ Servo GND
       │
Pico GPIO0 ─▶ Orange wire (Servo signal)
```

### Servo Calibration

The servo angles are limited to a fuel gauge range (not full 0-180°):

```
FULL   = 0°    (Empty needle up, full beer)
HALF   = 83°   (Middle position)
EMPTY  = 170°  (Full needle down, empty beer)

Valid range: 0-170°
Outside range: Rejected with error message
```

If servo seems offset, you can adjust in `Testcodegauge.py`:
```python
FULL_DEG = 0
HALF_DEG = 83
EMPTY_DEG = 170
```

---

## Advanced Usage

### Using RTSP Stream Instead of Webcam

```python
RTSP_URL = "rtsp://192.168.1.100:554/stream"  # IP camera
```

### Training Your Own Model

```bash
python scripts/prepare_yolo_dataset.py \
  --sources labeled_images/ \
  --out dataset/ \
  --classes classes.txt

yolo detect train model=yolov8s.pt data=dataset/data.yaml epochs=100
```

### Labeling New Data

```bash
# Create LabelMe annotations
python scripts/open_next_unlabeled.py --dir your_images_folder

# Convert to YOLO format
python scripts/labelme_to_yolo.py --images-dir your_images_folder
```

### Monitoring Training

```bash
tensorboard --logdir runs/detect
```

---

## Performance Notes

- **GPU:** ~30 FPS with NVIDIA RTX GPU
- **CPU:** ~5 FPS on modern CPU
- **Model size:** ~22 MB (YOLOv8s)
- **Inference time:** ~30ms per frame (GPU)
- **Serial latency:** ~50ms to Pico

---

## License & Attribution

See README.md for project information.

For questions or issues, check the troubleshooting section or open an issue on GitHub.

---

**Last Updated:** January 2026
**Python Version:** 3.10+
**Status:** Proof of Concept
