import os
import shutil
from pathlib import Path
from ultralytics import YOLO

# ============================================================
# VISIONSHIELD - REMOVE NO_BOOTS AND TRAIN 10-CLASS YOLO11n
# ============================================================

BASE = Path(r"C:\Users\vedan\Desktop\Python\VisionShield")

SOURCE = BASE / "datasets" / "visionshield_combined"
TARGET = BASE / "datasets" / "visionshield_10class"

# Original class 10 = no_boots
# We keep classes 0-9 unchanged.

NAMES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
]

print("=" * 70)
print("VISIONSHIELD - 10 CLASS DATASET PREPARATION")
print("=" * 70)

print("\nSource:")
print(SOURCE)

print("\nTarget:")
print(TARGET)

# ------------------------------------------------------------
# CREATE 10-CLASS DATASET
# ------------------------------------------------------------

for split in ["train", "val", "test"]:

    src_images = SOURCE / "images" / split
    src_labels = SOURCE / "labels" / split

    dst_images = TARGET / "images" / split
    dst_labels = TARGET / "labels" / split

    dst_images.mkdir(parents=True, exist_ok=True)
    dst_labels.mkdir(parents=True, exist_ok=True)

    image_files = list(src_images.glob("*"))

    print(f"\n{split.upper()}: {len(image_files)} images")

    removed_annotations = 0

    for image_path in image_files:

        # Copy image
        shutil.copy2(
            image_path,
            dst_images / image_path.name
        )

        # Corresponding label
        label_path = src_labels / (image_path.stem + ".txt")
        dst_label_path = dst_labels / (image_path.stem + ".txt")

        if not label_path.exists():
            continue

        new_lines = []

        with open(label_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:

            parts = line.strip().split()

            if len(parts) < 5:
                continue

            class_id = int(parts[0])

            # REMOVE no_boots (class 10)
            if class_id == 10:
                removed_annotations += 1
                continue

            # Keep classes 0-9
            if 0 <= class_id <= 9:
                new_lines.append(line)

        with open(dst_label_path, "w", encoding="utf-8") as f:
            f.writelines(new_lines)

    print("Removed no_boots annotations:", removed_annotations)


# ------------------------------------------------------------
# CREATE DATA.YAML
# ------------------------------------------------------------

yaml_path = TARGET / "data.yaml"

with open(yaml_path, "w", encoding="utf-8") as f:

    f.write(f"path: {TARGET.as_posix()}\n\n")
    f.write("train: images/train\n")
    f.write("val: images/val\n")
    f.write("test: images/test\n\n")
    f.write("names:\n")

    for i, name in enumerate(NAMES):
        f.write(f"  {i}: {name}\n")


print("\n" + "=" * 70)
print("10-CLASS DATASET READY")
print("=" * 70)

print("\nData YAML:")
print(yaml_path)

print("\nClasses:")
for i, name in enumerate(NAMES):
    print(f" {i}: {name}")


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

if __name__ == "__main__":

    import multiprocessing

    multiprocessing.freeze_support()

    print("\n" + "=" * 70)
    print("STARTING YOLO11n TRAINING")
    print("=" * 70)

    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(yaml_path),

        # Training
        epochs=20,

        # RTX 3050 4GB
        imgsz=640,
        batch=8,

        # GPU
        device=0,

        # Windows-safe DataLoader
        workers=0,

        # Early stopping
        patience=7,

        # Save checkpoints
        save=True,
        save_period=5,

        # Output
        project=str(BASE / "runs" / "detect"),
        name="visionshield_10class_yolo11n_50ep",

        # Pretrained YOLO11n
        pretrained=True,

        # Conservative augmentation
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        verbose=True
    )

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    print("\nBest model:")
    print(
        BASE /
        "runs" /
        "detect" /
        "visionshield_10class_yolo11n_50ep" /
        "weights" /
        "best.pt"
    )