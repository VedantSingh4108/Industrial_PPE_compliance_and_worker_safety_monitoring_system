
from pathlib import Path
from ultralytics import YOLO
import cv2
import csv

BASE = Path(r"C:\Users\vedan\Desktop\Python\VisionShield")

# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

MODEL_PATH = (
    BASE
    / "runs"
    / "detect"
    / "visionshield_10class_yolo11n_50ep-6"
    / "weights"
    / "best.pt"
)

# ------------------------------------------------------------
# TEST IMAGES
# ------------------------------------------------------------

IMAGE_FOLDER = (
    BASE
    / "datasets"
    / "visionshield_10class"
    / "images"
    / "test"
)

# ------------------------------------------------------------
# OUTPUT
# ------------------------------------------------------------

OUTPUT_FOLDER = (
    BASE
    / "batch_test_results"
)

CSV_PATH = OUTPUT_FOLDER / "detections.csv"

OUTPUT_FOLDER.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# SETTINGS
# ------------------------------------------------------------

CONFIDENCE = 0.25

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("=" * 70)
print("VISIONSHIELD - BATCH TEST")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nTest folder:")
print(IMAGE_FOLDER)

print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded.")

print("\nClasses:")

for class_id, class_name in model.names.items():
    print(f"  {class_id}: {class_name}")

# ------------------------------------------------------------
# FIND IMAGES
# ------------------------------------------------------------

image_files = []

for extension in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:

    image_files.extend(
        IMAGE_FOLDER.glob(extension)
    )

image_files = sorted(image_files)

print("\nImages found:", len(image_files))

if len(image_files) == 0:

    print("\nERROR: No images found.")

    input("\nPress Enter to exit...")
    raise SystemExit

# ------------------------------------------------------------
# CSV
# ------------------------------------------------------------

csv_file = open(
    CSV_PATH,
    "w",
    newline="",
    encoding="utf-8"
)

writer = csv.writer(csv_file)

writer.writerow([
    "image",
    "class",
    "confidence",
    "x1",
    "y1",
    "x2",
    "y2"
])

# ------------------------------------------------------------
# BATCH DETECTION
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("STARTING BATCH DETECTION")
print("=" * 70)

total_detections = 0

class_counts = {
    name: 0
    for name in model.names.values()
}

for index, image_path in enumerate(image_files, start=1):

    print(
        f"\n[{index}/{len(image_files)}] "
        f"{image_path.name}"
    )

    # Run prediction
    results = model.predict(
        source=str(image_path),
        conf=CONFIDENCE,
        device=0,
        verbose=False
    )

    result = results[0]

    # --------------------------------------------------------
    # SAVE ANNOTATED IMAGE
    # --------------------------------------------------------

    annotated = result.plot()

    output_path = OUTPUT_FOLDER / image_path.name

    cv2.imwrite(
        str(output_path),
        annotated
    )

    # --------------------------------------------------------
    # READ DETECTIONS
    # --------------------------------------------------------

    if result.boxes is None:
        continue

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        writer.writerow([
            image_path.name,
            class_name,
            f"{confidence:.4f}",
            f"{x1:.1f}",
            f"{y1:.1f}",
            f"{x2:.1f}",
            f"{y2:.1f}"
        ])

        total_detections += 1

        class_counts[class_name] += 1

        print(
            f"    {class_name:<15} "
            f"{confidence:.2f}"
        )

# ------------------------------------------------------------
# CLOSE CSV
# ------------------------------------------------------------

csv_file.close()

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 70)
print("BATCH TEST COMPLETE")
print("=" * 70)

print("\nImages tested:")
print(len(image_files))

print("\nTotal detections:")
print(total_detections)

print("\nDetection counts:")
print("-" * 40)

for class_name, count in class_counts.items():

    print(
        f"{class_name:<15}: {count}"
    )

print("\n" + "=" * 70)

print("\nAnnotated images:")
print(OUTPUT_FOLDER)

print("\nCSV:")
print(CSV_PATH)

print("\n" + "=" * 70)

input("\nPress Enter to exit...")

