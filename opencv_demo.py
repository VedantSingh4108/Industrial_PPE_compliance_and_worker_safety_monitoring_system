
import cv2
import time
from pathlib import Path
from ultralytics import YOLO


BASE = Path(r"C:\Users\vedan\Desktop\Python\VisionShield")

MODEL_PATH = (
    BASE
    / "runs"
    / "detect"
    / "visionshield_10class_yolo11n_50ep-6"
    / "weights"
    / "best.pt"
)


CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "none",
    6: "Person",
    7: "no_helmet",
    8: "no_goggle",
    9: "no_gloves",
}

PERSON_ID = 6

PPE_IDS = {
    "helmet": 0,
    "gloves": 1,
    "vest": 2,
    "boots": 3,
    "goggles": 4,
}

VIOLATION_IDS = {
    "no_helmet": 7,
    "no_goggle": 8,
    "no_gloves": 9,
}


CONF = 0.15

# Minimum confidence for PPE to count as present.
# Kept relatively low because validation showed weak
# detection for some PPE classes.
PPE_CONF = {
    "helmet": 0.25,
    "goggles": 0.15,
    "gloves": 0.15,
    "boots": 0.15,
    "vest": 0.20,
}

# Expand worker box when checking whether PPE belongs to worker.
# This is important because boots can be slightly outside
# the Person bounding box.
PERSON_EXPAND_X = 0.12
PERSON_EXPAND_Y = 0.12


print("=" * 70)
print("VISIONSHIELD - LIVE PPE TEST")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    print("\nERROR: Model not found!")
    print("Check MODEL_PATH in this script.")
    input("\nPress ENTER to exit...")
    raise SystemExit

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully.")
print("Classes:")
for cid, name in CLASS_NAMES.items():
    print(f"  {cid}: {name}")


cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("\nERROR: Could not open webcam.")
    input("Press ENTER to exit...")
    raise SystemExit

# Camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("\nCamera started.")
print("Press Q to quit.")


def intersection_over_area(inner_box, outer_box):
    """
    Returns what percentage of inner_box lies inside outer_box.
    """

    ix1, iy1, ix2, iy2 = inner_box
    ox1, oy1, ox2, oy2 = outer_box

    x1 = max(ix1, ox1)
    y1 = max(iy1, oy1)
    x2 = min(ix2, ox2)
    y2 = min(iy2, oy2)

    if x2 <= x1 or y2 <= y1:
        return 0.0

    intersection = (x2 - x1) * (y2 - y1)

    inner_area = max(
        1,
        (ix2 - ix1) * (iy2 - iy1)
    )

    return intersection / inner_area


def expand_box(box, frame_width, frame_height):
    """
    Expand a Person box so PPE slightly outside the worker
    bounding box can still be associated with that worker.
    """

    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    dx = int(w * PERSON_EXPAND_X)
    dy = int(h * PERSON_EXPAND_Y)

    x1 = max(0, x1 - dx)
    y1 = max(0, y1 - dy)
    x2 = min(frame_width - 1, x2 + dx)
    y2 = min(frame_height - 1, y2 + dy)

    return (x1, y1, x2, y2)


def center_inside(box, person_box):
    """
    Checks whether the center of a detection lies inside
    the worker box.
    """

    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    px1, py1, px2, py2 = person_box

    return (
        px1 <= cx <= px2
        and
        py1 <= cy <= py2
    )


def assign_ppe_to_worker(ppe_box, worker_box):
    """
    Determines whether a PPE detection belongs to a worker.

    Uses BOTH:
      1. center point
      2. overlap

    This prevents random PPE detections elsewhere in the image
    from making a worker incorrectly SAFE.
    """

    expanded = expand_box(
        worker_box,
        FRAME_W,
        FRAME_H
    )

    if center_inside(ppe_box, expanded):
        return True

    overlap = intersection_over_area(
        ppe_box,
        expanded
    )

    return overlap >= 0.20


def draw_text(img, text, position, color, scale=0.6, thickness=2):
    """
    Draw readable text with a black background.
    """

    x, y = position

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    cv2.rectangle(
        img,
        (x - 4, y - th - baseline - 4),
        (x + tw + 4, y + 4),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        img,
        text,
        (x, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )



previous_time = time.time()

while True:

    ret, frame = cap.read()

    if not ret:
        print("Failed to read camera frame.")
        break

    FRAME_H, FRAME_W = frame.shape[:2]


    results = model.predict(
        source=frame,
        conf=CONF,
        imgsz=640,
        device=0,
        verbose=False
    )

    result = results[0]

    workers = []
    ppe_detections = []
    violation_detections = []

    if result.boxes is not None:

        for box in result.boxes:

            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])

            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )

            detection = {
                "class_id": cls_id,
                "confidence": confidence,
                "box": (x1, y1, x2, y2),
                "name": CLASS_NAMES.get(
                    cls_id,
                    str(cls_id)
                )
            }

            # Person
            if cls_id == PERSON_ID:

                if confidence >= CONF:

                    workers.append(detection)

            # Positive PPE
            elif cls_id in PPE_IDS.values():

                name = CLASS_NAMES[cls_id]

                if confidence >= PPE_CONF[name]:

                    ppe_detections.append(detection)

            # Explicit violations
            elif cls_id in VIOLATION_IDS.values():

                if confidence >= CONF:

                    violation_detections.append(detection)


    safe_count = 0
    unsafe_count = 0

    worker_results = []

    for worker_index, worker in enumerate(workers):

        worker_box = worker["box"]

        # PPE belonging to this worker
        detected_ppe = {
            "helmet": None,
            "goggles": None,
            "gloves": None,
            "boots": None,
            "vest": None,
        }


        for ppe in ppe_detections:

            name = ppe["name"]

            if name not in detected_ppe:
                continue

            if assign_ppe_to_worker(
                ppe["box"],
                worker_box
            ):

                # Keep highest confidence detection
                if (
                    detected_ppe[name] is None
                    or
                    ppe["confidence"]
                    >
                    detected_ppe[name]["confidence"]
                ):
                    detected_ppe[name] = ppe

        

        worker_violations = set()

        for violation in violation_detections:

            if assign_ppe_to_worker(
                violation["box"],
                worker_box
            ):

                worker_violations.add(
                    violation["name"]
                )


        helmet_ok = detected_ppe["helmet"] is not None
        goggles_ok = detected_ppe["goggles"] is not None
        gloves_ok = detected_ppe["gloves"] is not None
        boots_ok = detected_ppe["boots"] is not None
        vest_ok = detected_ppe["vest"] is not None


        missing = []

        if not helmet_ok:
            missing.append("helmet")

        if not goggles_ok:
            missing.append("goggles")

        if not gloves_ok:
            missing.append("gloves")

        if not boots_ok:
            missing.append("boots")

        if not vest_ok:
            missing.append("vest")

        if "no_helmet" in worker_violations:
            if "helmet" not in missing:
                missing.append("helmet")

        if "no_goggle" in worker_violations:
            if "goggles" not in missing:
                missing.append("goggles")

        if "no_gloves" in worker_violations:
            if "gloves" not in missing:
                missing.append("gloves")

        unsafe = len(missing) > 0

        if unsafe:
            unsafe_count += 1
        else:
            safe_count += 1

        worker_results.append({
            "worker": worker,
            "ppe": detected_ppe,
            "missing": missing,
            "unsafe": unsafe,
            "violations": worker_violations
        })


    for ppe in ppe_detections:

        x1, y1, x2, y2 = ppe["box"]

        name = ppe["name"]
        conf = ppe["confidence"]

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 120, 0),
            2
        )

        label = f"{name} {conf:.2f}"

        draw_text(
            frame,
            label,
            (x1, max(25, y1)),
            (255, 120, 0),
            0.50,
            1
        )


    for index, data in enumerate(worker_results):

        worker = data["worker"]

        x1, y1, x2, y2 = worker["box"]

        unsafe = data["unsafe"]
        missing = data["missing"]

        if unsafe:

            box_color = (0, 0, 255)
            status_text = "UNSAFE"

        else:

            box_color = (0, 200, 0)
            status_text = "SAFE"


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )


        header = (
            f"Worker {index + 1} | "
            f"{status_text}"
        )

        draw_text(
            frame,
            header,
            (x1 + 5, max(30, y1 - 8)),
            box_color,
            0.70,
            2
        )


        if unsafe:

            missing_text = (
                "Missing: "
                + ", ".join(missing)
            )

            draw_text(
                frame,
                missing_text,
                (x1 + 5, y1 + 35),
                (0, 0, 255),
                0.52,
                2
            )

        else:

            draw_text(
                frame,
                "All required PPE detected",
                (x1 + 5, y1 + 35),
                (0, 200, 0),
                0.52,
                2
            )

    current_time = time.time()

    fps = 1.0 / max(
        current_time - previous_time,
        0.001
    )

    previous_time = current_time


    panel_width = 390
    panel_height = 145

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (10, 10),
        (panel_width, panel_height),
        (0, 0, 0),
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.72,
        frame,
        0.28,
        0
    )

    draw_text(
        frame,
        f"FPS: {fps:.1f}",
        (25, 40),
        (255, 255, 255),
        0.60,
        2
    )

    draw_text(
        frame,
        f"Workers: {len(workers)}",
        (25, 70),
        (255, 255, 255),
        0.60,
        2
    )

    draw_text(
        frame,
        f"SAFE: {safe_count}",
        (25, 100),
        (0, 220, 0),
        0.60,
        2
    )

    draw_text(
        frame,
        f"UNSAFE: {unsafe_count}",
        (25, 130),
        (0, 0, 255),
        0.60,
        2
    )


    if len(workers) == 0:

        global_status = "NO WORKER DETECTED"
        global_color = (0, 165, 255)

    elif unsafe_count > 0:

        global_status = "STATUS: UNSAFE"
        global_color = (0, 0, 255)

    else:

        global_status = "STATUS: SAFE"
        global_color = (0, 200, 0)

    draw_text(
        frame,
        global_status,
        (25, FRAME_H - 25),
        global_color,
        0.85,
        3
    )


    cv2.imshow(
        "VisionShield - PPE Safety Monitor",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("\nVisionShield stopped.")
