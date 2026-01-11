#!/usr/bin/env python3
"""
Prepare a YOLO-format dataset (images/labels train/val) from source folders.

Usage:
  python scripts/prepare_yolo_dataset.py --sources alfa-flesje me-drinking faces --out dataset --classes classes.txt

The script copies images and corresponding `.txt` label files into
the `out/images/train`, `out/images/val`, `out/labels/train`, `out/labels/val` structure.
It stratifies the split so that images containing the positive class (default: 'bier')
are split proportionally between train/val.
"""
import argparse
from pathlib import Path
import shutil
import random
from typing import List


IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.bmp'}


def read_classes(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [l.strip() for l in path.read_text(encoding='utf-8').splitlines() if l.strip()]


def collect_labeled_images(src_dirs: List[Path]):
    imgs = []
    for d in src_dirs:
        if not d.exists():
            continue
        for p in d.rglob('*'):
            if p.suffix.lower() in IMAGE_EXTS:
                txt = p.with_suffix('.txt')
                if txt.exists():
                    imgs.append((p, txt))
    return imgs


def has_class(txt_path: Path, class_idx: int) -> bool:
    try:
        for line in txt_path.read_text(encoding='utf-8').splitlines():
            if not line.strip():
                continue
            if int(line.split()[0]) == class_idx:
                return True
    except Exception:
        return False
    return False


def copy_pair(img_path: Path, txt_path: Path, out_img: Path, out_txt: Path):
    out_img.parent.mkdir(parents=True, exist_ok=True)
    out_txt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(img_path, out_img)
    shutil.copy2(txt_path, out_txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--sources', '-s', nargs='+', required=True, help='Source folders to collect labeled images from')
    ap.add_argument('--out', '-o', default='dataset', help='Output dataset root')
    ap.add_argument('--classes', '-c', default='classes.txt', help='Path to classes.txt')
    ap.add_argument('--positive-class', default='bier', help='Class name to stratify on (defaults to "bier")')
    ap.add_argument('--train-ratio', type=float, default=0.8, help='Train split ratio')
    ap.add_argument('--seed', type=int, default=42, help='Random seed')
    ap.add_argument('--remap', nargs='*', help='Optional per-source remapping rules. Format: srcdir:old=new[,old2=new2]')
    args = ap.parse_args()

    random.seed(args.seed)
    src_dirs = [Path(s) for s in args.sources]
    out_root = Path(args.out)
    classes = read_classes(Path(args.classes))

    if not classes:
        print('Warning: classes.txt not found or empty — proceeding without class-based stratification')

    try:
        pos_idx = classes.index(args.positive_class) if classes else None
    except ValueError:
        pos_idx = None

    # parse remap rules early so stratification considers remapped class ids
    remap_rules = {}
    if args.remap:
        for rule in args.remap:
            if ':' not in rule:
                continue
            src, maps = rule.split(':', 1)
            mapping = {}
            for pair in maps.split(','):
                if '=' in pair:
                    a, b = pair.split('=')
                    mapping[int(a)] = int(b)
            remap_rules[Path(src).resolve()] = mapping

    pairs = collect_labeled_images(src_dirs)
    if not pairs:
        print('No labeled image(.txt) pairs found in sources:', src_dirs)
        return

    pos = []
    neg = []
    def mapping_for_img(img_path: Path):
        for src_path, m in remap_rules.items():
            try:
                if src_path in img_path.resolve().parents:
                    return m
            except Exception:
                continue
        return None

    for img, txt in pairs:
        mapping = mapping_for_img(img)
        # check with mapping applied
        if pos_idx is not None:
            # read txt and check if any class maps to pos_idx
            found = False
            for line in txt.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cls = int(parts[0])
                if mapping and cls in mapping:
                    cls = mapping[cls]
                if cls == pos_idx:
                    found = True
                    break
            if found:
                pos.append((img, txt))
            else:
                neg.append((img, txt))
        else:
            neg.append((img, txt))

    print(f'Found {len(pairs)} labeled images: {len(pos)} positive, {len(neg)} negative')

    # shuffle
    random.shuffle(pos)
    random.shuffle(neg)

    def split(lst):
        n = max(1, int(len(lst) * args.train_ratio))
        return lst[:n], lst[n:]

    pos_train, pos_val = split(pos)
    neg_train, neg_val = split(neg)

    train = pos_train + neg_train
    val = pos_val + neg_val
    random.shuffle(train)
    random.shuffle(val)

    # copy
    # parse remap rules
    remap_rules = {}
    if args.remap:
        for rule in args.remap:
            # rule format: srcdir:0=1,2=3
            if ':' not in rule:
                continue
            src, maps = rule.split(':', 1)
            mapping = {}
            for pair in maps.split(','):
                if '=' in pair:
                    a, b = pair.split('=')
                    mapping[int(a)] = int(b)
            remap_rules[Path(src).resolve()] = mapping

    def apply_and_copy(img, txt, dest_img, dest_txt):
        # remap based on the source dir rules (match by ancestor)
        mapping = None
        for src_path, m in remap_rules.items():
            try:
                if src_path in img.resolve().parents:
                    mapping = m
                    break
            except Exception:
                continue

        if mapping:
            # read txt and remap class ids
            lines = []
            for line in txt.read_text(encoding='utf-8').splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                cls = int(parts[0])
                if cls in mapping:
                    parts[0] = str(mapping[cls])
                lines.append(' '.join(parts))
            dest_txt.parent.mkdir(parents=True, exist_ok=True)
            dest_txt.write_text('\n'.join(lines), encoding='utf-8')
            shutil.copy2(img, dest_img)
        else:
            copy_pair(img, txt, dest_img, dest_txt)

    for img, txt in train:
        dest_img = out_root / 'images' / 'train' / img.name
        dest_txt = out_root / 'labels' / 'train' / txt.name
        apply_and_copy(img, txt, dest_img, dest_txt)

    for img, txt in val:
        dest_img = out_root / 'images' / 'val' / img.name
        dest_txt = out_root / 'labels' / 'val' / txt.name
        apply_and_copy(img, txt, dest_img, dest_txt)

    # write data.yaml
    data_yaml = out_root / 'data.yaml'
    names_line = "names: [" + ",".join(f"'{c}'" for c in classes) + "]"
    data_yaml.write_text('\n'.join([
        f"train: {str((out_root/'images'/'train').resolve())}",
        f"val: {str((out_root/'images'/'val').resolve())}",
        f"nc: {len(classes)}",
        names_line,
    ]), encoding='utf-8')

    print(f'Created dataset in {out_root} — train: {len(train)}, val: {len(val)}')
    print('Data yaml written to', data_yaml)


if __name__ == '__main__':
    main()
