import cv2
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# VISIONSHIELD - LIVE WEBCAM
# 6-CLASS PER-WORKER PPE DETECTION
# ============================================================
#
# Classes:
# 0 = helmet
# 1 = gloves
# 2 = vest
# 3 = boots
# 4 = goggles
# 5 = Person
#
# SAFE:
#   Worker has all 5 required PPE items.
#
# UNSAFE:
#   Worker is missing one or more PPE items.
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
# CAMERA
# ============================================================

CAMERA_INDEX = 0

# Lower = faster.
# 640 is a good starting point for your RTX 3050.
IMAGE_SIZE = 640

CONFIDENCE = 0.25


# ============================================================
# REQUIRED PPE
# ============================================================

REQUIRED_PPE = [
    "helmet",
    "goggles",
    "gloves",
    "vest",
    "boots",
]


# ============================================================
# PERSON BOX EXPANSION
# ============================================================

PERSON_EXPAND_X = 0.12
PERSON_EXPAND_TOP = 0.25
PERSON_EXPAND_BOTTOM = 0.12


# ============================================================
# COLORS - BGR
# ============================================================

GREEN = (0, 255, 0)
RED = (0, 0, 255)
BLUE = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("VISIONSHIELD - LIVE WEBCAM PPE DETECTION")
print("=" * 70)

print("\nLoading model:")
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


def expand_person_box(
    box,
    image_width,
    image_height
):
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
    x1, y1, x2, y2 = box

    return (
        (x1 + x2) // 2,
        (y1 + y2) // 2
    )


def point_inside_box(
    point,
    box
):
    px, py = point
    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


def intersection_area(
    box1,
    box2
):
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


def association_score(
    ppe_box,
    person_box
):
    """
    Determines whether a PPE detection belongs
    to a particular worker.
    """

    center = box_center(ppe_box)

    if not point_inside_box(
        center,
        person_box
    ):
        return 0.0

    intersection = intersection_area(
        ppe_box,
        person_box
    )

    ppe_width = max(
        1,
        ppe_box[2] - ppe_box[0]
    )

    ppe_height = max(
        1,
        ppe_box[3] - ppe_box[1]
    )

    ppe_area = (
        ppe_width
        *
        ppe_height
    )

    overlap = (
        intersection
        /
        ppe_area
    )

    return 1.0 + overlap


def draw_text(
    image,
    text,
    position,
    color,
    scale=0.55,
    thickness=2
):
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        BLACK,
        thickness + 3,
        cv2.LINE_AA
    )

    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# OPEN WEBCAM
# ============================================================

print("\nOpening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam.\n"
        "Try CAMERA_INDEX = 1."
    )


# Request reasonable resolution.
cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


print("\nWebcam started.")
print("Press Q to quit.")
print("Press S to save the current frame.")


# ============================================================
# FPS
# ============================================================

fps = 0.0

previous_time = cv2.getTickCount()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    success, frame = cap.read()

    if not success:

        print(
            "\nCould not read frame."
        )

        break


    frame_height, frame_width = (
        frame.shape[:2]
    )


    # ========================================================
    # YOLO
    # ========================================================

    results = model.predict(
        source=frame,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )

    result = results[0]


    # ========================================================
    # READ DETECTIONS
    # ========================================================

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


    # ========================================================
    # SEPARATE PERSON / PPE
    # ========================================================

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


    # ========================================================
    # WORKER ANALYSIS
    # ========================================================

    workers = []


    for worker_index, person in enumerate(persons):

        person_box = person["box"]

        expanded_box = expand_person_box(
            person_box,
            frame_width,
            frame_height
        )


        # --------------------------------------------
        # PPE assigned to this worker
        # --------------------------------------------

        worker_ppe = {}

        for ppe in REQUIRED_PPE:

            worker_ppe[ppe] = None


        # --------------------------------------------
        # Associate PPE
        # --------------------------------------------

        for ppe_detection in ppe_detections:

            ppe_name = (
                ppe_detection["class"]
            )

            ppe_box = (
                ppe_detection["box"]
            )

            score = association_score(
                ppe_box,
                expanded_box
            )

            if score <= 0:
                continue


            current = (
                worker_ppe[ppe_name]
            )


            if current is None:

                worker_ppe[ppe_name] = {
                    "detection":
                        ppe_detection,
                    "score":
                        score
                }

            elif score > current["score"]:

                worker_ppe[ppe_name] = {
                    "detection":
                        ppe_detection,
                    "score":
                        score
                }


        # --------------------------------------------
        # Missing PPE
        # --------------------------------------------

        missing = []

        for ppe in REQUIRED_PPE:

            if worker_ppe[ppe] is None:

                missing.append(ppe)


        # --------------------------------------------
        # Safety
        # --------------------------------------------

        if len(missing) == 0:

            status = "SAFE"

        else:

            status = "UNSAFE"


        workers.append(
            {
                "index":
                    worker_index + 1,

                "person":
                    person,

                "ppe":
                    worker_ppe,

                "missing":
                    missing,

                "status":
                    status
            }
        )


    # ========================================================
    # COUNTERS
    # ========================================================

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


    # ========================================================
    # OVERALL STATUS
    # ========================================================

    if len(workers) == 0:

        overall_status = "NO WORKER"

        overall_color = WHITE

    elif unsafe_count > 0:

        overall_status = "UNSAFE"

        overall_color = RED

    else:

        overall_status = "SAFE"

        overall_color = GREEN


    # ========================================================
    # DRAW WORKERS
    # ========================================================

    for worker in workers:

        person = worker["person"]

        x1, y1, x2, y2 = (
            person["box"]
        )

        status = worker["status"]


        # --------------------------------------------
        # Worker color
        # --------------------------------------------

        if status == "SAFE":

            worker_color = GREEN

        else:

            worker_color = RED


        # --------------------------------------------
        # Person box
        # --------------------------------------------

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            worker_color,
            3
        )


        # --------------------------------------------
        # Worker label
        # --------------------------------------------

        draw_text(
            frame,
            (
                f"Worker {worker['index']} | "
                f"{status}"
            ),
            (
                x1,
                max(30, y1 - 10)
            ),
            worker_color,
            0.65,
            2
        )


        # ====================================================
        # PPE BOXES
        # ====================================================

        for ppe in REQUIRED_PPE:

            item = worker["ppe"][ppe]

            if item is None:
                continue


            detection = item[
                "detection"
            ]

            px1, py1, px2, py2 = (
                detection["box"]
            )

            confidence = (
                detection["confidence"]
            )


            # PPE box
            cv2.rectangle(
                frame,
                (px1, py1),
                (px2, py2),
                BLUE,
                2
            )


            # PPE label
            draw_text(
                frame,
                (
                    f"{ppe} "
                    f"{confidence:.2f}"
                ),
                (
                    px1,
                    max(25, py1 - 8)
                ),
                BLUE,
                0.50,
                2
            )


        # ====================================================
        # MISSING PPE
        # ====================================================

        if worker["missing"]:

            missing_text = (
                "MISSING: "
                +
                ", ".join(
                    worker["missing"]
                )
            )

            # Put it near bottom of person box.
            text_x = x1

            text_y = min(
                frame_height - 15,
                y2 - 10
            )

            draw_text(
                frame,
                missing_text,
                (
                    text_x,
                    text_y
                ),
                RED,
                0.50,
                2
            )


    # ========================================================
    # TOP UI
    # ========================================================

    # Semi-transparent dashboard.
    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (0, 0),
        (
            frame_width,
            105
        ),
        BLACK,
        -1
    )

    frame = cv2.addWeighted(
        overlay,
        0.75,
        frame,
        0.25,
        0
    )


    # ========================================================
    # DASHBOARD TEXT
    # ========================================================

    draw_text(
        frame,
        f"Workers: {len(workers)}",
        (20, 30),
        WHITE,
        0.65,
        2
    )

    draw_text(
        frame,
        f"Safe: {safe_count}",
        (20, 60),
        GREEN,
        0.65,
        2
    )

    draw_text(
        frame,
        f"Unsafe: {unsafe_count}",
        (20, 90),
        RED,
        0.65,
        2
    )


    # ========================================================
    # OVERALL STATUS
    # ========================================================

    status_x = max(
        350,
        frame_width - 360
    )

    draw_text(
        frame,
        f"STATUS: {overall_status}",
        (
            status_x,
            55
        ),
        overall_color,
        0.90,
        3
    )


    # ========================================================
    # FPS CALCULATION
    # ========================================================

    current_time = cv2.getTickCount()

    elapsed = (
        current_time - previous_time
    ) / cv2.getTickFrequency()

    previous_time = current_time

    if elapsed > 0:

        current_fps = 1.0 / elapsed

        fps = (
            0.90 * fps
            +
            0.10 * current_fps
        )


    draw_text(
        frame,
        f"FPS: {fps:.1f}",
        (
            frame_width - 130,
            frame_height - 20
        ),
        WHITE,
        0.50,
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "VisionShield - Live PPE Detection",
        frame
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    # Q = quit
    if key == ord("q"):

        break


    # S = save screenshot
    if key == ord("s"):

        output_path = (
            BASE
            /
            "webcam_capture.jpg"
        )

        cv2.imwrite(
            str(output_path),
            frame
        )

        print(
            f"\nScreenshot saved: "
            f"{output_path}"
        )


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print("\nWebcam closed.")
print("VisionShield stopped.")