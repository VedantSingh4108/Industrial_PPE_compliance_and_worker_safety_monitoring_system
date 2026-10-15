import cv2
from tkinter import Tk, filedialog
from ultralytics import YOLO
import os

# --------------------------------------------------
# MODEL
# --------------------------------------------------

MODEL_PATH = r"C:\Users\vedan\Desktop\Python\VisionShield\runs\detect\yolo11n_baseline\weights\best.pt"

model = YOLO(MODEL_PATH)

# --------------------------------------------------
# PICK IMAGE
# --------------------------------------------------

root = Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select an image",
    filetypes=[
        ("Image files", "*.jpg *.jpeg *.png *.bmp"),
        ("All files", "*.*")
    ]
)

root.destroy()

if not image_path:
    print("No image selected.")
    exit()

print("Selected:", image_path)

# --------------------------------------------------
# YOLO DETECTION
# --------------------------------------------------

results = model(
    image_path,
    conf=0.25,
    verbose=False
)

result = results[0]

# --------------------------------------------------
# DRAW ALL DETECTIONS
# --------------------------------------------------

annotated_image = result.plot()

# --------------------------------------------------
# SAVE OUTPUT
# --------------------------------------------------

folder = os.path.dirname(image_path)
filename = os.path.basename(image_path)

name, extension = os.path.splitext(filename)

output_path = os.path.join(
    folder,
    name + "_YOLO_detected.jpg"
)

cv2.imwrite(output_path, annotated_image)

# --------------------------------------------------
# PRINT DETECTIONS
# --------------------------------------------------

print("\n==============================")
print("YOLO DETECTIONS")
print("==============================")

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
            f"{class_name}: {confidence:.2f}"
        )

else:

    print("No objects detected.")

print("==============================")
print(f"\nOutput saved to:\n{output_path}")

# --------------------------------------------------
# SHOW IMAGE
# --------------------------------------------------

cv2.imshow(
    "VisionShield - YOLO Detection",
    annotated_image
)

print("\nPress any key to close.")

cv2.waitKey(0)
cv2.destroyAllWindows() 