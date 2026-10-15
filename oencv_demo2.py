import cv2
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# VISIONSHIELD - 6 CLASS PPE DETECTION
# PER-WORKER PPE ASSOCIATION
# ============================================================
#
# Classes:
#
# 0 = helmet
# 1 = gloves
# 2 = vest
# 3 = boots
# 4 = goggles
# 5 = Person
#
# SAFETY:
#
# Each Person is evaluated independently.
#
# SAFE:
#   helmet + goggles + gloves + vest + boots
#
# UNSAFE:
#   missing ANY required PPE
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE = Path(
    r"C:\Users\vedan\Desktop\Python\VisionShield"
)

MODEL_PATH = (
    BASE
    / "runs"
    / "detect"
    / "visionshield_6class_yolo11n_10ep"
    / "weights"
    / "best.pt"
)


# ============================================================
# INPUT
# ============================================================

# Image testing
IMAGE_PATH = BASE / "images.jpeg"

# For webcam/video later:
# IMAGE_PATH = None


# ============================================================
# SETTINGS
# ============================================================

CONFIDENCE = 0.25
IMAGE_SIZE = 640

REQUIRED_PPE = [
    "helmet",
    "goggles",
    "gloves",
    "vest",
    "boots",
]


# How much to expand the Person box when associating PPE.
#
# This is important because:
# - helmet can be slightly above the Person box
# - goggles can be near the top of the box
# - boots can be near the bottom
#
PERSON_EXPAND_X = 0.12
PERSON_EXPAND_TOP = 0.25
PERSON_EXPAND_BOTTOM = 0.12


# Minimum overlap/association requirement
PPE_CENTER_REQUIRED = True


# ============================================================
# COLORS - BGR
# ============================================================

GREEN = (0, 255, 0)
RED = (0, 0, 255)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("VISIONSHIELD - 6 CLASS PPE DETECTOR")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )

model = YOLO(str(MODEL_PATH))

print("\nModel loaded successfully.")

print("\nClasses:")

for class_id, name in model.names.items():
    print(f"{class_id}: {name}")


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def expand_person_box(box, image_width, image_height):
    """
    Expand the Person bounding box.

    This allows PPE slightly outside the Person box
    to still belong to that worker.
    """

    x1, y1, x2, y2 = box

    width = x2 - x1
    height = y2 - y1

    expand_x = int(width * PERSON_EXPAND_X)
    expand_top = int(height * PERSON_EXPAND_TOP)
    expand_bottom = int(height * PERSON_EXPAND_BOTTOM)

    new_x1 = clamp(
        x1 - expand_x,
        0,
        image_width - 1
    )

    new_y1 = clamp(
        y1 - expand_top,
        0,
        image_height - 1
    )

    new_x2 = clamp(
        x2 + expand_x,
        0,
        image_width - 1
    )

    new_y2 = clamp(
        y2 + expand_bottom,
        0,
        image_height - 1
    )

    return (
        new_x1,
        new_y1,
        new_x2,
        new_y2
    )


def box_center(box):
    """
    Return center point of bounding box.
    """

    x1, y1, x2, y2 = box

    return (
        (x1 + x2) // 2,
        (y1 + y2) // 2
    )


def point_inside_box(point, box):
    """
    Check whether a point lies inside a bounding box.
    """

    px, py = point

    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


def intersection_area(box1, box2):
    """
    Calculate intersection area between two boxes.
    """

    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    ix1 = max(x1a, x1b)
    iy1 = max(y1a, y1b)

    ix2 = min(x2a, x2b)
    iy2 = min(y2a, y2b)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0

    return (
        (ix2 - ix1)
        *
        (iy2 - iy1)
    )


def association_score(ppe_box, person_box):
    """
    Calculate how strongly a PPE detection belongs
    to a particular person.

    Uses:
      1. PPE center inside expanded person box
      2. Intersection area
    """

    center = box_center(ppe_box)

    if point_inside_box(center, person_box):

        intersection = intersection_area(
            ppe_box,
            person_box
        )

        ppe_area = max(
            1,
            (ppe_box[2] - ppe_box[0])
            *
            (ppe_box[3] - ppe_box[1])
        )

        overlap = intersection / ppe_area

        # Center inside is the strongest signal.
        return 1.0 + overlap

    return 0.0


def draw_label(
    image,
    text,
    x,
    y,
    color
):
    """
    Draw readable label with background.
    """

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    y = max(
        th + baseline + 5,
        y
    )

    cv2.rectangle(
        image,
        (
            x,
            y - th - baseline - 8
        ),
        (
            x + tw + 8,
            y + 2
        ),
        color,
        -1
    )

    cv2.putText(
        image,
        text,
        (
            x + 4,
            y - 5
        ),
        font,
        scale,
        WHITE,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# IMAGE MODE
# ============================================================

if IMAGE_PATH is None:

    raise RuntimeError(
        "IMAGE_PATH is None. "
        "Use an image path for this version."
    )


print("\n" + "=" * 70)
print("IMAGE DETECTION")
print("=" * 70)

print("\nInput:")
print(IMAGE_PATH)

if not IMAGE_PATH.exists():

    raise FileNotFoundError(
        f"\nImage not found:\n{IMAGE_PATH}"
    )


image = cv2.imread(
    str(IMAGE_PATH)
)

if image is None:

    raise RuntimeError(
        "Could not read image."
    )


image_height, image_width = image.shape[:2]


# ============================================================
# RUN YOLO
# ============================================================

results = model.predict(
    source=image,
    conf=CONFIDENCE,
    imgsz=IMAGE_SIZE,
    verbose=False
)

result = results[0]


# ============================================================
# READ DETECTIONS
# ============================================================

detections = []


if result.boxes is not None:

    for box in result.boxes:

        class_id = int(
            box.cls[0].item()
        )

        confidence = float(
            box.conf[0].item()
        )

        class_name = model.names[
            class_id
        ]

        x1, y1, x2, y2 = map(
            int,
            box.xyxy[0].tolist()
        )

        detections.append(
            {
                "class": class_name,
                "confidence": confidence,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                )
            }
        )


# ============================================================
# SEPARATE PERSON AND PPE DETECTIONS
# ============================================================

persons = [
    d
    for d in detections
    if d["class"] == "Person"
]

ppe_detections = [
    d
    for d in detections
    if d["class"] in REQUIRED_PPE
]


print("\nDetections:")
print("-" * 60)

for detection in detections:

    print(
        f"{detection['class']:10s} "
        f"{detection['confidence']:.3f}"
    )


print("\nWorkers detected:", len(persons))


# ============================================================
# CREATE WORKER DATA
# ============================================================

workers = []


for worker_index, person in enumerate(persons):

    person_box = person["box"]

    expanded_box = expand_person_box(
        person_box,
        image_width,
        image_height
    )

    worker_ppe = {}

    # Start with everything missing.
    for ppe in REQUIRED_PPE:
        worker_ppe[ppe] = None

    # --------------------------------------------------------
    # ASSIGN PPE TO THIS PERSON
    # --------------------------------------------------------

    for ppe_detection in ppe_detections:

        ppe_name = ppe_detection["class"]

        ppe_box = ppe_detection["box"]

        score = association_score(
            ppe_box,
            expanded_box
        )

        if score <= 0:
            continue

        # If there are multiple detections of the
        # same PPE type, keep the strongest one.
        current = worker_ppe[ppe_name]

        if current is None:

            worker_ppe[ppe_name] = {
                "detection": ppe_detection,
                "score": score
            }

        elif score > current["score"]:

            worker_ppe[ppe_name] = {
                "detection": ppe_detection,
                "score": score
            }

    # --------------------------------------------------------
    # FIND MISSING PPE
    # --------------------------------------------------------

    detected_ppe = []

    missing_ppe = []

    for ppe in REQUIRED_PPE:

        if worker_ppe[ppe] is not None:

            detected_ppe.append(ppe)

        else:

            missing_ppe.append(ppe)

    # --------------------------------------------------------
    # SAFETY DECISION
    # --------------------------------------------------------

    if len(missing_ppe) == 0:

        status = "SAFE"

    else:

        status = "UNSAFE"

    workers.append(
        {
            "index": worker_index + 1,
            "person": person,
            "expanded_box": expanded_box,
            "ppe": worker_ppe,
            "detected": detected_ppe,
            "missing": missing_ppe,
            "status": status
        }
    )


# ============================================================
# PRINT WORKER RESULTS
# ============================================================

print("\n" + "=" * 70)
print("WORKER SAFETY RESULTS")
print("=" * 70)

for worker in workers:

    print(
        f"\nWorker {worker['index']}: "
        f"{worker['status']}"
    )

    for ppe in REQUIRED_PPE:

        if worker["ppe"][ppe] is not None:

            confidence = worker["ppe"][ppe][
                "detection"
            ]["confidence"]

            print(
                f"  {ppe:10s}: "
                f"OK ({confidence:.2f})"
            )

        else:

            print(
                f"  {ppe:10s}: MISSING"
            )


# ============================================================
# GLOBAL COUNTERS
# ============================================================

safe_count = sum(
    1
    for worker in workers
    if worker["status"] == "SAFE"
)

unsafe_count = sum(
    1
    for worker in workers
    if worker["status"] == "UNSAFE"
)


# ============================================================
# DRAW PERSON + PPE
# ============================================================

annotated = image.copy()


for worker in workers:

    person = worker["person"]

    x1, y1, x2, y2 = person["box"]

    status = worker["status"]

    # --------------------------------------------------------
    # PERSON COLOR
    # --------------------------------------------------------

    if status == "SAFE":

        person_color = GREEN

    else:

        person_color = RED

    # --------------------------------------------------------
    # PERSON BOX
    # --------------------------------------------------------

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        person_color,
        3
    )

    # --------------------------------------------------------
    # WORKER LABEL
    # --------------------------------------------------------

    worker_label = (
        f"Worker {worker['index']} | "
        f"{status}"
    )

    draw_label(
        annotated,
        worker_label,
        x1,
        max(30, y1),
        person_color
    )

    # --------------------------------------------------------
    # PPE BOXES BELONGING TO THIS WORKER
    # --------------------------------------------------------

    for ppe in REQUIRED_PPE:

        item = worker["ppe"][ppe]

        if item is None:
            continue

        detection = item["detection"]

        px1, py1, px2, py2 = (
            detection["box"]
        )

        confidence = detection["confidence"]

        # PPE boxes are blue.
        cv2.rectangle(
            annotated,
            (px1, py1),
            (px2, py2),
            BLUE,
            2
        )

        label = (
            f"{ppe} "
            f"{confidence:.2f}"
        )

        draw_label(
            annotated,
            label,
            px1,
            max(25, py1),
            BLUE
        )


# ============================================================
# DRAW MISSING PPE NEXT TO EACH WORKER
# ============================================================

for worker in workers:

    x1, y1, x2, y2 = worker["person"]["box"]

    missing = worker["missing"]

    if not missing:
        continue

    # Place text inside/near the lower part of worker box.
    text_y = min(
        image_height - 20,
        y2 - 10
    )

    missing_text = (
        "MISSING: "
        + ", ".join(missing)
    )

    # Black background for readability.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.55
    thickness = 2

    (tw, th), baseline = cv2.getTextSize(
        missing_text,
        font,
        scale,
        thickness
    )

    box_x1 = max(
        0,
        x1
    )

    box_y1 = max(
        0,
        text_y - th - 12
    )

    box_x2 = min(
        image_width - 1,
        box_x1 + tw + 10
    )

    box_y2 = min(
        image_height - 1,
        text_y + 5
    )

    cv2.rectangle(
        annotated,
        (box_x1, box_y1),
        (box_x2, box_y2),
        BLACK,
        -1
    )

    cv2.putText(
        annotated,
        missing_text,
        (
            box_x1 + 5,
            text_y
        ),
        font,
        scale,
        RED,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# TOP DASHBOARD
# ============================================================

panel_height = 125

cv2.rectangle(
    annotated,
    (0, 0),
    (
        image_width,
        panel_height
    ),
    BLACK,
    -1
)


# ------------------------------------------------------------
# WORKER COUNTS
# ------------------------------------------------------------

cv2.putText(
    annotated,
    f"Workers: {len(workers)}",
    (20, 30),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    WHITE,
    2,
    cv2.LINE_AA
)


cv2.putText(
    annotated,
    f"Safe: {safe_count}",
    (20, 60),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    GREEN,
    2,
    cv2.LINE_AA
)


cv2.putText(
    annotated,
    f"Unsafe: {unsafe_count}",
    (20, 90),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    RED,
    2,
    cv2.LINE_AA
)


# ============================================================
# OVERALL STATUS
# ============================================================

if len(workers) == 0:

    overall_status = "NO WORKER"

elif unsafe_count > 0:

    overall_status = "UNSAFE"

else:

    overall_status = "SAFE"


overall_color = (
    GREEN
    if overall_status == "SAFE"
    else RED
)


cv2.putText(
    annotated,
    f"STATUS: {overall_status}",
    (
        max(
            300,
            image_width - 350
        ),
        50
    ),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    overall_color,
    3,
    cv2.LINE_AA
)


# ============================================================
# SAVE RESULT
# ============================================================

output_path = (
    IMAGE_PATH.parent
    /
    (
        IMAGE_PATH.stem
        +
        "_6class_per_worker_result.jpg"
    )
)


success = cv2.imwrite(
    str(output_path),
    annotated
)

if not success:

    raise RuntimeError(
        "Could not save output image."
    )


print("\n" + "=" * 70)

print("FINAL RESULT")

print("=" * 70)

print(
    f"Workers : {len(workers)}"
)

print(
    f"Safe    : {safe_count}"
)

print(
    f"Unsafe  : {unsafe_count}"
)

print(
    f"Status  : {overall_status}"
)

print("\nOutput saved to:")

print(output_path)


# ============================================================
# SHOW RESULT
# ============================================================

window_name = (
    "VisionShield - "
    "6 Class Per-Worker PPE Detection"
)

cv2.namedWindow(
    window_name,
    cv2.WINDOW_NORMAL
)

cv2.imshow(
    window_name,
    annotated
)

print(
    "\nPress any key to close."
)

cv2.waitKey(0)

cv2.destroyAllWindows()