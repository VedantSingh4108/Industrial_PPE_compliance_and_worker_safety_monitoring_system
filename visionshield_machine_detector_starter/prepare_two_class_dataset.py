
from pathlib import Path
import shutil

source = Path("dataset/combined_machine_detector")
target = Path("dataset/combined_machine_detector_2class")
image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

if not (source / "data.yaml").exists():
    raise SystemExit(f"Missing source YAML: {source / 'data.yaml'}")

if target.exists():
    raise SystemExit(
        f"STOP: {target} already exists. "
        "Nothing was overwritten."
    )

try:
    for split in ("train", "valid", "test"):
        src_images = source / "images" / split
        src_labels = source / "labels" / split

        if not src_images.is_dir() or not src_labels.is_dir():
            raise RuntimeError(
                f"Missing folders for {split}:\n"
                f"{src_images}\n{src_labels}"
            )

        dst_images = target / "images" / split
        dst_labels = target / "labels" / split
        dst_images.mkdir(parents=True, exist_ok=True)
        dst_labels.mkdir(parents=True, exist_ok=True)

        images = [
            p for p in src_images.rglob("*")
            if p.is_file() and p.suffix.lower() in image_exts
        ]

        if not images:
            raise RuntimeError(f"No images found in {src_images}")

        for image in images:
            relative = image.relative_to(src_images)
            label = src_labels / relative.with_suffix(".txt")

            if not label.is_file():
                raise RuntimeError(f"Missing label: {label}")

            new_rows = []
            for line in label.read_text(encoding="utf-8").splitlines():
                parts = line.split()
                if not parts:
                    continue

                class_id = int(parts[0])

                if class_id == 0:          # forklift stays 0
                    new_rows.append(" ".join(parts))
                elif class_id == 1:        # conveyor removed
                    continue
                elif class_id == 2:        # production machine becomes 1
                    parts[0] = "1"
                    new_rows.append(" ".join(parts))
                else:
                    raise RuntimeError(
                        f"Unexpected class {class_id} in {label}"
                    )

            out_image = dst_images / relative
            out_label = dst_labels / relative.with_suffix(".txt")
            out_image.parent.mkdir(parents=True, exist_ok=True)
            out_label.parent.mkdir(parents=True, exist_ok=True)

            shutil.copy2(image, out_image)
            out_label.write_text(
                "\n".join(new_rows) + ("\n" if new_rows else ""),
                encoding="utf-8"
            )

    yaml = (
        f"path: {target.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/valid\n"
        "test: images/test\n"
        "nc: 2\n"
        "names:\n"
        "  0: forklift\n"
        "  1: production_machine\n"
    )
    (target / "data.yaml").write_text(yaml, encoding="utf-8")

except Exception:
    shutil.rmtree(target, ignore_errors=True)
    raise

print("SUCCESS: Separate two-class dataset created.")
print("Dataset:", target.resolve())
print("Class 0: forklift")
print("Class 1: production_machine")
print("Original dataset and checkpoint were not modified.")
