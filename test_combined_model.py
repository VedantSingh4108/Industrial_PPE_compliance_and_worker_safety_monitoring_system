from ultralytics import YOLO
import cv2
import os
from tkinter import Tk, filedialog

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_PATH = r"C:\Users\vedan\Desktop\Python\VisionShield\runs\detect\visionshield_combined_yolo11n_20ep\weights\best.pt"

model = YOLO(MODEL_PATH)

print("Model loaded successfully.")
print("Classes:")
print(model.names)

# --------------------------------------------------
# SELECT IMAGE
# --------------------------------------------------

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select test image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not image_path:
    print("No image selected.")
    exit()

# --------------------------------------------------
# DETECTION
# --------------------------------------------------

results = model(
    image_path,
    conf=0.25,
    verbose=False
)

result = results[0]

# --------------------------------------------------
# DRAW
# --------------------------------------------------

annotated = result.plot()

# --------------------------------------------------
# SAVE
# --------------------------------------------------

folder = os.path.dirname(image_path)
filename = os.path.basename(image_path)
name, ext = os.path.splitext(filename)

output_path = os.path.join(
    folder,
    name + "_combined_model.jpg"
)

cv2.imwrite(output_path, annotated)

# --------------------------------------------------
# PRINT RESULTS
# --------------------------------------------------

print("\n======================================")
print("DETECTIONS")
print("======================================")

if result.boxes is not None and len(result.boxes) > 0:

    for box, cls, conf in zip(
        result.boxes.xyxy,
        result.boxes.cls,
        result.boxes.conf
    ):

        class_id = int(cls)
        class_name = model.names[class_id]
        confidence = float(conf)

        print(
            f"{class_name:15s} confidence={confidence:.3f}"
        )

else:
    print("No objects detected.")

print("======================================")
print(f"Output saved to:\n{output_path}")

# --------------------------------------------------
# SHOW
# --------------------------------------------------

cv2.imshow(
    "VisionShield - Combined YOLO11n",
    annotated
)

print("\nPress any key to close.")

cv2.waitKey(0)
cv2.destroyAllWindows()