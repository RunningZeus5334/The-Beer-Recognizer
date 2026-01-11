#!/usr/bin/env python3
"""
Open the next unlabeled image in a directory with LabelMe.

Usage:
  python scripts/open_next_unlabeled.py --dir alfa-flesje

Finds the first image (jpg, png, bmp) without a same-named .json file
and opens it with the `labelme` command. Requires `labelme` to be on PATH
in the active venv.
"""
import argparse
import subprocess
from pathlib import Path

IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def find_next_unlabeled(dirpath: Path):
    for p in sorted(dirpath.rglob('*')):
        if p.suffix.lower() in IMAGE_EXTS:
            json_path = p.with_suffix('.json')
            if not json_path.exists():
                return p
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', '-d', required=True, help='Directory to search for unlabeled images')
    args = ap.parse_args()

    dirpath = Path(args.dir)
    if not dirpath.exists():
        print('Directory not found:', dirpath)
        return

    img = find_next_unlabeled(dirpath)
    if not img:
        print('No unlabeled images found in', dirpath)
        return

    print('Opening', img)
    try:
        subprocess.run(['labelme', str(img)], check=False)
    except FileNotFoundError:
        print('Could not find `labelme` on PATH. Activate the venv or install labelme in your environment.')


if __name__ == '__main__':
    main()
