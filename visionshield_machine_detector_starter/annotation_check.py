"""Quickly check dataset image/label counts and malformed YOLO annotation rows."""
import argparse
from pathlib import Path

VALID_CLASS_IDS = {0, 1, 2}

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="datasets/machine_detector")
    p.add_argument("--splits", nargs="+", default=["train", "val"])
    args = p.parse_args()
    root = Path(args.root)
    total_issues = 0

    for split in args.splits:
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        images = []
        if image_dir.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp"):
                images.extend(image_dir.glob(ext))
        else:
            print(f"[WARN] Missing image directory: {image_dir}")

        missing_labels = []
        malformed = []
        class_counts = {0: 0, 1: 0, 2: 0}
        for image in images:
            label = label_dir / (image.stem + ".txt")
            if not label.exists():
                missing_labels.append(image.name)
                continue
            for line_no, line in enumerate(label.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                parts = line.split()
                try:
                    if len(parts) != 5:
                        raise ValueError("expected 5 columns")
                    cls = int(parts[0])
                    vals = [float(v) for v in parts[1:]]
                    if cls not in VALID_CLASS_IDS:
                        raise ValueError("class ID must be 0, 1, or 2")
                    if not all(0.0 <= v <= 1.0 for v in vals):
                        raise ValueError("normalized values must be in [0,1]")
                    if vals[2] <= 0 or vals[3] <= 0:
                        raise ValueError("box width/height must be positive")
                    class_counts[cls] += 1
                except Exception as e:
                    malformed.append(f"{label.name}:{line_no} ({e})")

        print(f"\n[{split}] images={len(images)} class instances={class_counts}")
        print(f"  images with missing label files: {len(missing_labels)}")
        print(f"  malformed annotation rows: {len(malformed)}")
        for x in missing_labels[:10]:
            print("  missing:", x)
        for x in malformed[:20]:
            print("  malformed:", x)
        total_issues += len(missing_labels) + len(malformed)

    if total_issues:
        print(f"\nCHECK REQUIRED: {total_issues} dataset issues found.")
    else:
        print("\nNo missing label files or malformed annotation rows found by this basic checker.")

if __name__ == "__main__":
    main()
