
from pathlib import Path
from ultralytics import YOLO


BASE = Path(r"C:\Users\vedan\Desktop\Python\VisionShield")

MODEL = (
    BASE
    / "runs"
    / "detect"
    / "visionshield_10class_yolo11n_50ep-6"
    / "weights"
    / "best.pt"
)

DATA = (
    BASE
    / "datasets"
    / "visionshield_10class"
    / "data.yaml"
)

PROJECT = BASE / "runs" / "detect"


print("=" * 70)
print("VISIONSHIELD - TEST SET EVALUATION")
print("=" * 70)

print("\nModel:")
print(MODEL)

print("\nDataset:")
print(DATA)


if not MODEL.exists():
    print("\nERROR: Model not found:")
    print(MODEL)
    raise SystemExit

if not DATA.exists():
    print("\nERROR: data.yaml not found:")
    print(DATA)
    raise SystemExit


print("\nLoading model...")

model = YOLO(str(MODEL))

print("Model loaded successfully.")


print("\n" + "=" * 70)
print("RUNNING TEST SET")
print("=" * 70)

results = model.val(
    data=str(DATA),

    # IMPORTANT:
    # evaluate the TEST split
    split="test",

    imgsz=640,
    batch=8,
    device=0,

    workers=0,

    # Save confusion matrix and plots
    plots=True,

    # Save results
    project=str(PROJECT),
    name="visionshield_10class_test"
)



print("\n" + "=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(f"\nmAP50:     {results.box.map50:.4f}")
print(f"mAP50-95:  {results.box.map:.4f}")

print("\nPer-class mAP50:")

names = model.names

for i, value in enumerate(results.box.ap50):

    name = names.get(i, str(i))

    print(
        f"{i:2d}  "
        f"{name:<12} "
        f"mAP50 = {value:.4f}"
    )




OUTPUT = (
    PROJECT
    / "visionshield_10class_test"
)

print("\n" + "=" * 70)
print("TEST COMPLETE")
print("=" * 70)

print("\nResults saved to:")
print(OUTPUT)

print("\nLook for:")
print("  confusion_matrix.png")
print("  confusion_matrix_normalized.png")
print("  results.png")

print("\nDone.")

