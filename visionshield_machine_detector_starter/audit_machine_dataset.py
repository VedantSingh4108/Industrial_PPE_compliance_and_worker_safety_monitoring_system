from pathlib import Path
from collections import Counter, defaultdict
import yaml

ROOT = Path(r"C:\Users\vedan\Desktop\Python\visionshield_machine_detector_starter")
COMBINED = ROOT / "dataset" / "combined_machine_detector"
RAW = ROOT / "dataset" / "raw"

def read_yaml(path):
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"Could not read YAML {path}: {e}")
        return {}

def resolve_split(data, split):
    value = data.get(split)
    if not value:
        return None
    p = Path(str(value))
    if not p.is_absolute():
        p = (COMBINED / p).resolve()
    return p

def count_split_images_and_labels(data, split):
    img_root = resolve_split(data, split)
    print(f"\n[{split}] configured path: {img_root}")
    if not img_root or not img_root.exists():
        print("  Path missing. Check data.yaml paths manually.")
        return
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    images = sorted(p for p in img_root.rglob("*") if p.suffix.lower() in image_exts)
    if not images:
        print("  No images found under configured path.")
        return
    label_root = None
    # Ultralytics convention: .../images/... -> .../labels/...
    parts = list(img_root.parts)
    if "images" in parts:
        idx = parts.index("images")
        label_root = Path(*parts[:idx], "labels", *parts[idx + 1:])
    if label_root is None or not label_root.exists():
        print(f"  Could not infer labels folder from {img_root}")
        return
    class_counts = Counter()
    images_with_class = Counter()
    empty_labels = 0
    missing_labels = 0
    conveyor_only_or_not = Counter()
    for im in images:
        rel = im.relative_to(img_root)
        lab = label_root / rel.with_suffix(".txt")
        if not lab.exists():
            missing_labels += 1
            continue
        rows = [line.strip() for line in lab.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
        if not rows:
            empty_labels += 1
            continue
        per_image = set()
        for row in rows:
            cols = row.split()
            try:
                cid = int(float(cols[0]))
            except (ValueError, IndexError):
                continue
            class_counts[cid] += 1
            per_image.add(cid)
        for cid in per_image:
            images_with_class[cid] += 1
        conveyor_only_or_not["images_with_conveyor"] += int(1 in per_image)
        conveyor_only_or_not["images_without_conveyor_with_labels"] += int(1 not in per_image)
    print(f"  Images: {len(images)}")
    print(f"  Label rows per class: {dict(sorted(class_counts.items()))}")
    print(f"  Images per class: {dict(sorted(images_with_class.items()))}")
    print(f"  Empty label files: {empty_labels}")
    print(f"  Missing label files: {missing_labels}")
    print(f"  Images with conveyor class 1: {conveyor_only_or_not['images_with_conveyor']}")
    print(f"  Labeled images without conveyor class 1: {conveyor_only_or_not['images_without_conveyor_with_labels']}")

def main():
    print("READ-ONLY DATASET AUDIT — no files are changed")
    cfg = COMBINED / "data.yaml"
    if not cfg.exists():
        print(f"Missing dataset config: {cfg}")
        return
    data = read_yaml(cfg)
    print(f"Dataset config: {cfg}")
    print(f"Class names: {data.get('names')}")
    print(f"Train: {data.get('train')}")
    print(f"Val:   {data.get('val')}")
    print(f"Test:  {data.get('test')}")
    for split in ("train", "val", "test"):
        count_split_images_and_labels(data, split)
    print("\nRAW dataset folders:")
    if RAW.exists():
        for p in sorted(RAW.iterdir()):
            if p.is_dir():
                print(f"  {p.name}")
    else:
        print(f"  Missing raw folder: {RAW}")
    print("\nNext step: share this output before creating a revised dataset.")

if __name__ == "__main__":
    main()
