- # The Beer Recognizer

Een proof-of-concept project dat een gezicht en een bierflesje/-glas detecteert met een getraind YOLO-model en op basis van het geschatte bierniveau een servo op een Raspberry Pi Pico aanstuurt.

**Status:** proof-of-concept — werkt op Linux met een CUDA-compatibele NVIDIA GPU.

**Snelle controle (voordat je start)**
- Zorg dat het getrainde model beschikbaar is en dat `MODEL_PATH` in `scripts/Beer_Detector.py` correct is ingesteld.
- Controleer je Pico verbinding: gebruik `ls /dev/ttyACM* /dev/ttyUSB*` en zet `SERIAL_PORT` in `scripts/Beer_Detector.py` op het juiste device.
- Upload of draai `code/gauge/Testcodegauge.py` op de Pico (Thonny of `mpremote`).

**Overzicht**
- **Beschrijving:** Dit project detecteert een gezicht en een bierflesje/glazen in een videofeed (webcam of RTSP) met een YOLO-model en stuurt op basis van de geschatte bierhoogte een hoek naar een Raspberry Pi Pico die een servo bestuurt (gauge).
- **Belangrijke bestanden:** [scripts/Beer_Detector.py](scripts/Beer_Detector.py), [code/gauge/Testcodegauge.py](code/gauge/Testcodegauge.py), [requirements.txt](requirements.txt), [classes.txt](classes.txt)

**Systeemvereisten (minimaal)**
- **OS:** Linux (ontwikkeld en getest op Linux)
- **GPU:** NVIDIA GPU met CUDA-ondersteuning (voor real-time detectie met GPU)
- **CUDA / cuDNN:** compatibel met je geïnstalleerde PyTorch-versie
- **Python:** 3.10+ (venv aanbevolen; repository bevat voorbeelden met 3.12)
- **Hardware:** Raspberry Pi Pico (of compatibele MicroPython-board) voor servo-besturing

**Opzet en installatie**
1. Maak en activeer een virtuele omgeving (aanbevolen):

```bash
python -m venv .venv
source .venv/bin/activate
```

2. Installeer Python-dependencies:

```bash
pip install -r requirements.txt
```

3. Controleer dat `pyserial` is geïnstalleerd (voor communicatie met de Pico). Als je later problemen krijgt met serial, controleer het device-pad (bv. `/dev/ttyACM0` of `/dev/ttyUSB0`).

**Pico (MicroPython) voorbereiden**
- Flasht of laad [code/gauge/Testcodegauge.py](code/gauge/Testcodegauge.py) op je Pico met Thonny, rshell, ampy of mpremote. Deze MicroPython script leest lijnen vanaf `stdin` (serial) met een numerieke hoek (0..180) en zet de servo daarop.

Voorbeeld met `mpremote` (vereist installatie op host):

```bash
mpremote connect /dev/ttyACM0 run code/gauge/Testcodegauge.py
```

Of open Thonny, selecteer je Pico en upload [code/gauge/Testcodegauge.py](code/gauge/Testcodegauge.py).

**Configuratie**
- Pas indien nodig in [scripts/Beer_Detector.py](scripts/Beer_Detector.py) de volgende variabelen aan bovenin het bestand:
	- `MODEL_PATH` – pad naar je getrainde model (bv. `runs/detect/.../best.pt`).
	- `RTSP_URL` – 0 voor lokale webcam of een RTSP URL.
	- `CLASS_ID_FACE` en `CLASS_ID_BEER` – check je model `classes.txt` en pas indices aan.
	- `SERIAL_PORT` – set naar het juiste device (bv. `/dev/ttyACM0`).

**Runnen van de detector**
1. Zorg dat de Pico is verbonden en dat `Testcodegauge.py` op de Pico draait (of dat de Pico een script heeft dat lijnen met hoeken accepteert via serial).
2. Start de detector op de host:

```bash
python scripts/Beer_Detector.py
```

3. De host-app zal detecties tonen in een venster en periodiek (of bij verandering) een hoek naar de Pico sturen. Gebruik `q` om te stoppen of `r` om het bierlevel naar 100% te resetten.

**Hoe het werkt (kort)**
- Het model (YOLO via `ultralytics`) detecteert objecten en retourneert bounding boxes met klasse-id en confidence.
- De code zoekt specifiek naar `FACE` en `BEER` boxes en berekent overlap (intersection over beer box area). Als overlap > `OVERLAP_THRESHOLD` wordt aangenomen dat er gedronken wordt en vermindert `CURRENT_LEVEL_PERCENT` over tijd met `DRINK_DECAY_RATE_PER_SEC`.
- `CURRENT_LEVEL_PERCENT` wordt gemapt naar een servohoek: 0% → 0°, 100% → 180°. De host stuurt de hoek (integer gevolgd door newline) via serial naar de Pico.
- De Pico (MicroPython) leest de hoek en zet de servopositie via PWM.

**Aanpassingen en tuning**
- Drempels: pas `STANDARD_CONFIDENCE_THRESHOLD`, `BEER_CONFIDENCE_THRESHOLD` en `OVERLAP_THRESHOLD` aan om minder/meer detecties te accepteren.
- `DRINK_DECAY_RATE_PER_SEC` regelt hoe snel het level daalt tijdens drinken.
- `SEND_INTERVAL` en `ANGLE_DELTA_THRESHOLD` in `Beer_Detector.py` voorkomen te frequente serial-sends.

**Fouten opsporen**
- Geen beeld: controleer `RTSP_URL` of webcam en dat `opencv` camera toegang heeft.
- Geen serial verbinding: controleer device met `ls /dev/ttyACM* /dev/ttyUSB*`. Pas `SERIAL_PORT` aan.
- Model laden faalt: controleer `MODEL_PATH` en dat de ultralytics-versie compatibel is.

**Opsommend overzicht van belangrijke bestanden**
- [scripts/Beer_Detector.py](scripts/Beer_Detector.py) : Host-detectie, level-logica, serial-communicatie
- [code/gauge/Testcodegauge.py](code/gauge/Testcodegauge.py) : MicroPython op Pico, ontvangt hoekwaarden en stuurt servo
- [requirements.txt](requirements.txt) : Python dependencies, inclusief `pyserial` en `ultralytics`

Als je wilt, kan ik:
- automatisch het juiste serial-device op jouw machine detecteren en `SERIAL_PORT` in [scripts/Beer_Detector.py](scripts/Beer_Detector.py) instellen, of
- een kort scriptsnippet toevoegen om `Testcodegauge.py` via `mpremote` te uploaden.

