
from pathlib import Path
from ultralytics import YOLO
import cv2

# ============================================================
# VISIONSHIELD - TEST 10-CLASS YOLO11n MODEL
# ============================================================

BASE = Path(r"C:\Users\vedan\Desktop\Python\VisionShield")

MODEL_PATH = (
    BASE
    / "runs"
    / "detect"
    / "visionshield_10class_yolo11n_50ep-6"
    / "weights"
    / "best.pt"
)

# ------------------------------------------------------------
# CHANGE THIS IMAGE TO WHATEVER YOU WANT TO TEST
# ------------------------------------------------------------

IMAGE_PATH = BASE / "images.jpeg"

# Output image
OUTPUT_PATH = BASE / "test_10class_result_1.jpeg"

# Detection confidence
CONFIDENCE = 0.25

# ------------------------------------------------------------
# LOAD MODEL
# ------------------------------------------------------------

print("=" * 60)
print("VISIONSHIELD - 10 CLASS MODEL TEST")
print("=" * 60)

print("\nLoading model:")
print(MODEL_PATH)

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully.")

print("\nClasses:")
print(model.names)

# ------------------------------------------------------------
# RUN DETECTION
# ------------------------------------------------------------

print("\nRunning detection...")

results = model.predict(
    source=str(IMAGE_PATH),
    conf=CONFIDENCE,
    device=0,
    verbose=False
)

result = results[0]

# ------------------------------------------------------------
# PRINT DETECTIONS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("DETECTIONS")
print("=" * 60)

if result.boxes is None or len(result.boxes) == 0:

    print("No detections found.")

else:

    for box in result.boxes:

        class_id = int(box.cls[0])
        confidence = float(box.conf[0])

        class_name = model.names[class_id]

        print(
            f"{class_name:<15} "
            f"confidence={confidence:.3f}"
        )

# ------------------------------------------------------------
# SAVE RESULT
# ------------------------------------------------------------

annotated = result.plot()

cv2.imwrite(
    str(OUTPUT_PATH),
    annotated
)

print("\n" + "=" * 60)
print("TEST COMPLETE")
print("=" * 60)

print("\nOutput saved to:")
print(OUTPUT_PATH)

print("\nPress any key to close.")

cv2.imshow("VisionShield - 10 Class Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

