# Label workflow for YOLOv8 training

This repository contains two image folders: `alfa-flesje/` and `me-drinking/`.
We create a Python 3.12 virtual environment and use `labelme` to annotate images,
then convert LabelMe JSONs to YOLO format (`.txt` files) using the provided script.

Quick steps

1. Activate the 3.12 venv:

```powershell
# from repository root
.\.venv312\Scripts\Activate.ps1
```

2. (Optional) Run `labelme` to annotate images. Example:

```powershell
labelme alfa-flesje\some_image.jpg
```

Label images and save — LabelMe creates `*.json` files next to images.

3. Convert LabelMe JSONs to YOLO txt files and generate `classes.txt`:

```powershell
python .\scripts\labelme_to_yolo.py --images-dir alfa-flesje --classes classes.txt
python .\scripts\labelme_to_yolo.py --images-dir me-drinking --classes classes.txt
```

This produces `image_name.txt` files alongside the images and updates `classes.txt`.

4. Prepare `data.yaml` for YOLOv8 training (example):

```yaml
train: /absolute/path/to/alfa-flesje
val: /absolute/path/to/me-drinking
names: classes.txt
```

Notes and alternatives
- If `labelme` GUI fails on your PC, use the browser-based `VIA` (VGG Image Annotator) at https://www.robots.ox.ac.uk/~vgg/software/via/ — export annotations and adapt to YOLO (the included script assumes LabelMe jsons).
- The venv is ` .venv312` and was created with system Python 3.12.

Files added
- `scripts/labelme_to_yolo.py`: conversion script.
- `requirements-venv312.txt`: installed packages list (in venv root).
