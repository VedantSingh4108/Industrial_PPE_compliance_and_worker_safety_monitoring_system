
import sys
import cv2
import math
from ultralytics import YOLO

# Existing VisionShield machine detector — unchanged
MACHINE_MODEL = (
    r"runs\detect\runs\detect"
    r"\visionshield_machine_2class_aug40\weights\best.pt"
)

# Pretrained COCO detector; class 0 is "person"
PERSON_MODEL = "yolo11n.pt"

# Prototype threshold: normalized image gap, NOT real-world distance.
# Tune this after testing your camera/video.
PROXIMITY_THRESHOLD = 0.06
MACHINE_CONFIDENCE = 0.25
PERSON_CONFIDENCE = 0.30

machine_model = YOLO(MACHINE_MODEL)
person_model = YOLO(PERSON_MODEL)

# Usage: python worker_machine_proximity.py 0
# Or pass a video path as the argument.
source = sys.argv[1] if len(sys.argv) > 1 else "0"
if source.isdigit():
    source = int(source)

cap = cv2.VideoCapture(source)
if not cap.isOpened():
    raise SystemExit(f"Could not open video source: {source}")

print("VisionShield worker-machine proximity started.")
print("Press Q in the video window to quit.")

while True:
    ok, frame = cap.read()
    if not ok:
        break

    height, width = frame.shape[:2]
    diagonal = math.hypot(width, height)

    # Detect forklifts and production machines
    machine_results = machine_model.predict(
        frame, conf=MACHINE_CONFIDENCE, verbose=False
    )[0]

    # Detect people using COCO class 0
    person_results = person_model.predict(
        frame, classes=[0], conf=PERSON_CONFIDENCE,
        imgsz=640, verbose=False
    )[0]

    machines = []
    people = []

    if machine_results.boxes is not None:
        for box in machine_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_id = int(box.cls[0])
            label = machine_model.names[cls_id]
            machines.append((x1, y1, x2, y2, label))
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (255, 180, 0), 2)
            cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 180, 0), 2)

    if person_results.boxes is not None:
        for box in person_results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            people.append((x1, y1, x2, y2))
            cv2.rectangle(frame, (x1, y1), (x2, y2),
                          (0, 220, 0), 2)
            cv2.putText(frame, "person", (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 220, 0), 2)

    warnings = 0

    for px1, py1, px2, py2 in people:
        for mx1, my1, mx2, my2, label in machines:
            # Minimum gap between the two rectangles.
            dx = max(mx1 - px2, px1 - mx2, 0)
            dy = max(my1 - py2, py1 - my2, 0)
            normalized_gap = math.hypot(dx, dy) / diagonal

            if normalized_gap < PROXIMITY_THRESHOLD:
                warnings += 1
                cv2.line(
                    frame,
                    ((px1 + px2) // 2, (py1 + py2) // 2),
                    ((mx1 + mx2) // 2, (my1 + my2) // 2),
                    (0, 0, 255), 2
                )
                cv2.putText(
                    frame, "PROXIMITY WARNING",
                    (20, 40 + (warnings - 1) * 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75,
                    (0, 0, 255), 2
                )

    if warnings == 0:
        cv2.putText(frame, "No proximity alert",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0, 220, 0), 2)

    cv2.imshow("VisionShield - Worker Machine Proximity", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
