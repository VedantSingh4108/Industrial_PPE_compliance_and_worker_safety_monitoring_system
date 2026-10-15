import os
import shutil
from pathlib import Path
from multiprocessing import freeze_support

from ultralytics import YOLO


# ============================================================
# VISIONSHIELD - 6 CLASS DATASET + YOLO11n TRAINING
# ============================================================
#
# ORIGINAL CLASSES:
#
# 0 helmet
# 1 gloves
# 2 vest
# 3 boots
# 4 goggles
# 5 none              -> REMOVE
# 6 Person            -> KEEP, REMAP TO 5
# 7 no_helmet         -> REMOVE
# 8 no_goggle         -> REMOVE
# 9 no_gloves         -> REMOVE
#
# FINAL:
#
# 0 helmet
# 1 gloves
# 2 vest
# 3 boots
# 4 goggles
# 5 Person
#
# ============================================================


BASE = Path(
    r"C:\Users\vedan\Desktop\Python\VisionShield"
)

SOURCE = (
    BASE
    / "datasets"
    / "visionshield_combined"
)

TARGET = (
    BASE
    / "datasets"
    / "visionshield_6class"
)


NAMES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "Person",
]


# Original -> new class mapping
#
# 0 -> 0 helmet
# 1 -> 1 gloves
# 2 -> 2 vest
# 3 -> 3 boots
# 4 -> 4 goggles
# 5 -> REMOVE none
# 6 -> 5 Person
# 7 -> REMOVE no_helmet
# 8 -> REMOVE no_goggle
# 9 -> REMOVE no_gloves

CLASS_MAP = {
    0: 0,
    1: 1,
    2: 2,
    3: 3,
    4: 4,
    6: 5,
}


def prepare_dataset():

    print("=" * 70)
    print("VISIONSHIELD - 6 CLASS DATASET PREPARATION")
    print("=" * 70)

    print("\nSOURCE:")
    print(SOURCE)

    print("\nTARGET:")
    print(TARGET)

    if not SOURCE.exists():
        raise FileNotFoundError(
            f"Source dataset does not exist:\n{SOURCE}"
        )

    # --------------------------------------------------------
    # REMOVE OLD TARGET
    # --------------------------------------------------------

    if TARGET.exists():

        print("\nRemoving previous 6-class dataset...")

        shutil.rmtree(TARGET)

    TARGET.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # PROCESS TRAIN / VAL / TEST
    # --------------------------------------------------------

    total_images = 0
    total_labels = 0

    removed_none = 0
    removed_no_helmet = 0
    removed_no_goggle = 0
    removed_no_gloves = 0

    for split in ["train", "val", "test"]:

        src_images = SOURCE / "images" / split
        src_labels = SOURCE / "labels" / split

        dst_images = TARGET / "images" / split
        dst_labels = TARGET / "labels" / split

        dst_images.mkdir(
            parents=True,
            exist_ok=True
        )

        dst_labels.mkdir(
            parents=True,
            exist_ok=True
        )

        if not src_images.exists():

            print(
                f"\nWARNING: {split} images folder does not exist."
            )

            continue

        image_files = [
            p for p in src_images.iterdir()
            if p.is_file()
        ]

        print(
            f"\n{split.upper()}: "
            f"{len(image_files)} images"
        )

        for image_path in image_files:

            # ------------------------------------------------
            # COPY IMAGE
            # ------------------------------------------------

            shutil.copy2(
                image_path,
                dst_images / image_path.name
            )

            total_images += 1

            # ------------------------------------------------
            # FIND LABEL
            # ------------------------------------------------

            label_path = (
                src_labels
                / f"{image_path.stem}.txt"
            )

            dst_label_path = (
                dst_labels
                / f"{image_path.stem}.txt"
            )

            if not label_path.exists():

                # Create empty label file
                # so YOLO knows image has no objects.
                dst_label_path.touch()

                continue

            new_lines = []

            with open(
                label_path,
                "r",
                encoding="utf-8"
            ) as f:

                lines = f.readlines()

            for line in lines:

                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                try:
                    old_class = int(parts[0])
                except ValueError:
                    continue

                # --------------------------------------------
                # REMOVE unwanted classes
                # --------------------------------------------

                if old_class == 5:

                    removed_none += 1
                    continue

                if old_class == 7:

                    removed_no_helmet += 1
                    continue

                if old_class == 8:

                    removed_no_goggle += 1
                    continue

                if old_class == 9:

                    removed_no_gloves += 1
                    continue

                # --------------------------------------------
                # KEEP + REMAP
                # --------------------------------------------

                if old_class in CLASS_MAP:

                    new_class = CLASS_MAP[old_class]

                    # Replace only class ID.
                    new_line = (
                        str(new_class)
                        + " "
                        + " ".join(parts[1:])
                        + "\n"
                    )

                    new_lines.append(new_line)

                    total_labels += 1

            # ------------------------------------------------
            # WRITE NEW LABEL
            # ------------------------------------------------

            with open(
                dst_label_path,
                "w",
                encoding="utf-8"
            ) as f:

                f.writelines(new_lines)

    # --------------------------------------------------------
    # DATASET YAML
    # --------------------------------------------------------

    yaml_path = TARGET / "data.yaml"

    with open(
        yaml_path,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            f"path: {TARGET.as_posix()}\n\n"
        )

        f.write(
            "train: images/train\n"
        )

        f.write(
            "val: images/val\n"
        )

        f.write(
            "test: images/test\n\n"
        )

        f.write("names:\n")

        for i, name in enumerate(NAMES):

            f.write(
                f"  {i}: {name}\n"
            )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("6-CLASS DATASET READY")
    print("=" * 70)

    print(
        f"\nTotal images: {total_images}"
    )

    print(
        f"Total kept annotations: {total_labels}"
    )

    print(
        f"Removed 'none':       {removed_none}"
    )

    print(
        f"Removed 'no_helmet':  {removed_no_helmet}"
    )

    print(
        f"Removed 'no_goggle':  {removed_no_goggle}"
    )

    print(
        f"Removed 'no_gloves':  {removed_no_gloves}"
    )

    print("\nFINAL CLASSES:")

    for i, name in enumerate(NAMES):

        print(
            f"  {i}: {name}"
        )

    print("\nYAML:")

    print(yaml_path)

    return yaml_path


def train_model(yaml_path):

    print("\n" + "=" * 70)
    print("STARTING YOLO11n - 6 CLASS TEST TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD PRETRAINED YOLO11n
    # --------------------------------------------------------

    model = YOLO("yolo11n.pt")

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    results = model.train(

        data=str(yaml_path),

        # --------------------------------------------
        # LOW EPOCHS FOR FIRST TEST
        # --------------------------------------------

        epochs=5,

        # RTX 3050 4GB
        imgsz=640,
        batch=8,

        # GPU
        device=0,

        # IMPORTANT ON WINDOWS
        workers=0,

        # Stop early if validation stops improving
        patience=5,

        # Save checkpoints
        save=True,
        save_period=5,

        # Output
        project=str(
            BASE / "runs" / "detect"
        ),

        name="visionshield_6class_yolo11n_10ep",

        # Pretrained YOLO11n
        pretrained=True,

        # Conservative augmentation
        degrees=5.0,
        translate=0.1,
        scale=0.5,
        fliplr=0.5,

        # Verbose output
        verbose=True,
    )

    # --------------------------------------------------------
    # FINISHED
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("TRAINING FINISHED")
    print("=" * 70)

    best_path = (
        BASE
        / "runs"
        / "detect"
        / "visionshield_6class_yolo11n_10ep"
        / "weights"
        / "best.pt"
    )

    last_path = (
        BASE
        / "runs"
        / "detect"
        / "visionshield_6class_yolo11n_10ep"
        / "weights"
        / "last.pt"
    )

    print("\nBEST MODEL:")
    print(best_path)

    print("\nLAST MODEL:")
    print(last_path)

    print("\nFINAL CLASSES:")

    for i, name in enumerate(NAMES):

        print(
            f"{i}: {name}"
        )

    return results



if __name__ == "__main__":

    # Prevent Windows multiprocessing/spawn problems.
    freeze_support()

    yaml_path = prepare_dataset()

    train_model(yaml_path)