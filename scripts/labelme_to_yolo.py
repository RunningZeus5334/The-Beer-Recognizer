#!/usr/bin/env python3
"""
Convert LabelMe JSON annotations to YOLO format (one .txt per image).

Usage:
  python scripts/labelme_to_yolo.py --images-dir alfa-flesje --classes classes.txt

If `classes.txt` is present it should contain one class name per line.
If a label in the json isn't found in `classes.txt`, it will be appended.
"""
import argparse
import json
from pathlib import Path
import cv2


def load_classes(classes_path: Path):
    if classes_path and classes_path.exists():
        lines = [l.strip() for l in classes_path.read_text(encoding="utf-8").splitlines() if l.strip()]
        return lines
    return []


def save_classes(classes, classes_path: Path):
    classes_path.write_text("\n".join(classes), encoding="utf-8")


def json_to_yolo(json_path: Path, classes, write_txt=True):
    data = json.loads(json_path.read_text(encoding="utf-8"))
    image_path = json_path.with_suffix("")
    if "imagePath" in data and data["imagePath"]:
        image_path = json_path.with_name(data["imagePath"]) if not Path(data["imagePath"]).is_absolute() else Path(data["imagePath"])

    if not image_path.exists():
        # try common extensions
        for ext in ('.jpg', '.jpeg', '.png', '.bmp'):
            p = json_path.with_suffix(ext)
            if p.exists():
                image_path = p
                break

    if not image_path.exists():
        print(f"Image for {json_path} not found, skipping")
        return []

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]

    out_lines = []
    for shape in data.get('shapes', []):
        label = shape.get('label')
        pts = shape.get('points', [])
        if not pts:
            continue
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        box_w = x_max - x_min
        box_h = y_max - y_min

        # normalize
        x_center /= w
        y_center /= h
        box_w /= w
        box_h /= h

        if label not in classes:
            classes.append(label)

        cls_id = classes.index(label)
        out_lines.append(f"{cls_id} {x_center:.6f} {y_center:.6f} {box_w:.6f} {box_h:.6f}")

    if write_txt and out_lines:
        txt_path = image_path.with_suffix('.txt')
        txt_path.write_text("\n".join(out_lines), encoding="utf-8")

    return classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images-dir', '-i', required=True, help='Directory with images and LabelMe jsons')
    ap.add_argument('--classes', '-c', default='classes.txt', help='Path to classes.txt')
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    classes_path = Path(args.classes)
    classes = load_classes(classes_path)

    json_files = list(images_dir.rglob('*.json'))
    if not json_files:
        print('No LabelMe json files found under', images_dir)
        return

    for j in json_files:
        classes = json_to_yolo(j, classes)

    save_classes(classes, classes_path)
    print(f'Converted {len(json_files)} json files. Classes saved to {classes_path}')


if __name__ == '__main__':
    main()
