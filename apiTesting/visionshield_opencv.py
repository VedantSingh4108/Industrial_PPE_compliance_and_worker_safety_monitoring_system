
import cv2
import time
import requests
from pathlib import Path
from ultralytics import YOLO


# ============================================================
# VISIONSHIELD
# OPENCV WEBCAM + YOLO + WORKER TRACKING + FASTAPI
# ============================================================
#
# MODEL:
#   0 = helmet
#   1 = gloves
#   2 = vest
#   3 = boots
#   4 = goggles
#   5 = Person
#
# BEHAVIOUR:
#
#   - Open webcam directly with OpenCV
#   - Track individual workers using YOLO ByteTrack
#   - Determine PPE for each individual worker
#   - If worker is unsafe continuously for 15 seconds:
#         send ONE API request
#
#   - If still unsafe:
#         another API request after another 15 seconds
#
#   Example:
#
#       unsafe starts
#       0 sec
#       15 sec -> API CALL #1
#       30 sec -> API CALL #2
#       45 sec -> API CALL #3
#
#   NOT:
#
#       30 FPS -> 30 API calls/sec
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
# FASTAPI
# ============================================================

API_BASE_URL = "http://127.0.0.1:8000"

VIOLATION_ENDPOINT = (
    API_BASE_URL
    + "/api/violations"
)

HEALTH_ENDPOINT = (
    API_BASE_URL
    + "/api/health"
)


# ============================================================
# CAMERA
# ============================================================

CAMERA_INDEX = 0

IMAGE_SIZE = 640

CONFIDENCE = 0.20

PERSON_CONFIDENCE = 0.25


# ============================================================
# PPE
# ============================================================

REQUIRED_PPE = {
    "helmet",
    "goggles",
    "gloves",
    "vest",
    "boots",
}


PPE_ORDER = [
    "helmet",
    "goggles",
    "gloves",
    "vest",
    "boots",
]


# ============================================================
# API TIMING
# ============================================================

# IMPORTANT:
#
# A violation is sent every 15 seconds PER WORKER.
#
VIOLATION_INTERVAL = 15.0


# ============================================================
# REQUEST TIMEOUT
# ============================================================

API_TIMEOUT = 1.5


# ============================================================
# TRACKING
# ============================================================

TRACKER_CONFIG = "bytetrack.yaml"


# ============================================================
# DISPLAY
# ============================================================

WINDOW_NAME = "VisionShield - PPE Safety System"


# ============================================================
# COLOURS
# ============================================================

WHITE = (255, 255, 255)

BLACK = (0, 0, 0)

RED = (0, 0, 255)

GREEN = (0, 255, 0)

BLUE = (255, 0, 0)

YELLOW = (0, 255, 255)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("VISIONSHIELD - OPENCV WEBCAM SYSTEM")
print("=" * 70)

print("\nLoading model:")
print(MODEL_PATH)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nModel not found:\n{MODEL_PATH}"
    )


model = YOLO(
    str(MODEL_PATH)
)


print("\nModel loaded successfully.")

print("\nClasses:")

for class_id, name in model.names.items():

    print(
        f"{class_id}: {name}"
    )


# ============================================================
# FASTAPI HEALTH CHECK
# ============================================================

def check_api():

    try:

        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=API_TIMEOUT
        )

        return response.status_code == 200

    except requests.RequestException:

        return False


print("\nChecking FastAPI...")

api_online = check_api()

if api_online:

    print("FastAPI: ONLINE")

else:

    print("FastAPI: OFFLINE")

print()


# ============================================================
# CAMERA
# ============================================================

print("Opening webcam...")

cap = cv2.VideoCapture(
    CAMERA_INDEX
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# Try to use a reasonable webcam resolution.

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    1280
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    720
)


# ============================================================
# WORKER STATE
# ============================================================
#
# worker_states:
#
# {
#     worker_id: {
#
#         "unsafe_since": timestamp,
#
#         "last_api_call": timestamp,
#
#         "api_calls": number,
#
#         "missing_ppe": set(),
#
#         "detected_ppe": set(),
#
#         "last_seen": timestamp
#     }
# }
#
# ============================================================

worker_states = {}


# ============================================================
# GLOBAL API CALL COUNTER
# ============================================================

total_api_calls = 0


# ============================================================
# FPS
# ============================================================

fps = 0.0

fps_timer = time.time()

fps_frames = 0


# ============================================================
# API ERROR DISPLAY
# ============================================================

last_api_error = ""

last_api_check = 0.0

API_CHECK_INTERVAL = 3.0


# ============================================================
# HELPER FUNCTIONS
# ============================================================


def box_center(box):

    x1, y1, x2, y2 = box

    return (
        int((x1 + x2) / 2),
        int((y1 + y2) / 2)
    )


def point_inside_box(point, box):

    px, py = point

    x1, y1, x2, y2 = box

    return (
        x1 <= px <= x2
        and
        y1 <= py <= y2
    )


def draw_text(
    image,
    text,
    position,
    color=WHITE,
    scale=0.6,
    thickness=2
):

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


def draw_label(
    image,
    text,
    x,
    y,
    color=BLUE
):

    (tw, th), _ = cv2.getTextSize(
        text,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        2
    )

    cv2.rectangle(
        image,
        (x, y - th - 8),
        (x + tw + 8, y),
        color,
        -1
    )

    draw_text(
        image,
        text,
        (x + 4, y - 5),
        WHITE,
        0.55,
        2
    )


# ============================================================
# SEND VIOLATION TO FASTAPI
# ============================================================

def send_violation(
    worker_id,
    duration,
    missing_ppe,
    detected_ppe,
    workers_snapshot
):

    global total_api_calls
    global api_online
    global last_api_error

    payload = {

        "timestamp": time.time(),

        "worker_id": worker_id,

        "violation": True,

        "duration_seconds": round(
            duration,
            2
        ),

        "worker_count": len(
            workers_snapshot
        ),

        "unsafe_count": sum(
            1
            for worker in workers_snapshot
            if worker["status"] == "UNSAFE"
        ),

        "missing_ppe": sorted(
            list(missing_ppe)
        ),

        "detected_ppe": sorted(
            list(detected_ppe)
        ),

        "message": (
            f"PPE violation detected "
            f"for Worker {worker_id}"
        ),

        "workers": workers_snapshot,
    }

    try:

        response = requests.post(
            VIOLATION_ENDPOINT,
            json=payload,
            timeout=API_TIMEOUT
        )

        if response.status_code in (
            200,
            201
        ):

            api_online = True

            total_api_calls += 1

            last_api_error = ""

            print()
            print("=" * 70)
            print("PPE VIOLATION API SENT")
            print("=" * 70)

            print(
                f"Worker: {worker_id}"
            )

            print(
                f"Unsafe duration: "
                f"{duration:.2f}s"
            )

            print(
                f"Missing PPE: "
                f"{sorted(missing_ppe)}"
            )

            print(
                f"API status: "
                f"{response.status_code}"
            )

            print(
                f"Total API calls: "
                f"{total_api_calls}"
            )

            print("=" * 70)
            print()

            return True

        else:

            api_online = True

            last_api_error = (
                f"HTTP {response.status_code}"
            )

            print(
                f"\nViolation API error: "
                f"HTTP {response.status_code}"
            )

            print(
                response.text
            )

            return False

    except requests.RequestException as error:

        api_online = False

        last_api_error = str(error)

        print(
            "\nViolation API connection error:"
        )

        print(
            error
        )

        return False


# ============================================================
# MAIN LOOP
# ============================================================

print("=" * 70)
print("VISIONSHIELD WEBCAM STARTED")
print("=" * 70)

print(
    "\nPress Q to quit."
)

print(
    "Violation API interval:",
    VIOLATION_INTERVAL,
    "seconds"
)

print()


while True:

    # --------------------------------------------------------
    # READ CAMERA
    # --------------------------------------------------------

    success, frame = cap.read()

    if not success:

        print(
            "Could not read webcam frame."
        )

        break


    current_time = time.time()


    # --------------------------------------------------------
    # PERIODIC API HEALTH CHECK
    # --------------------------------------------------------

    if (
        current_time
        - last_api_check
        >= API_CHECK_INTERVAL
    ):

        api_online = check_api()

        last_api_check = current_time


    # --------------------------------------------------------
    # YOLO TRACKING
    # --------------------------------------------------------
    #
    # persist=True keeps track IDs between frames.
    #
    # ByteTrack gives us:
    #
    # Worker 1
    # Worker 2
    # Worker 3
    #
    # instead of treating everybody as one worker.
    #

    results = model.track(
        source=frame,
        persist=True,
        tracker=TRACKER_CONFIG,
        conf=CONFIDENCE,
        imgsz=IMAGE_SIZE,
        verbose=False
    )


    result = results[0]


    # --------------------------------------------------------
    # DETECTIONS
    # --------------------------------------------------------

    persons = []

    ppe_detections = []


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


            detection_box = (
                x1,
                y1,
                x2,
                y2
            )


            # ------------------------------------------------
            # PERSON
            # ------------------------------------------------

            if class_name == "Person":

                if confidence < PERSON_CONFIDENCE:

                    continue


                track_id = None


                if box.id is not None:

                    track_id = int(
                        box.id[0].item()
                    )


                if track_id is None:

                    continue


                persons.append(
                    {
                        "id": track_id,

                        "box": detection_box,

                        "confidence": confidence,
                    }
                )


            # ------------------------------------------------
            # PPE
            # ------------------------------------------------

            elif class_name in REQUIRED_PPE:

                ppe_detections.append(
                    {
                        "class": class_name,

                        "box": detection_box,

                        "confidence": confidence,
                    }
                )


    # ========================================================
    # BUILD WORKER STATES
    # ========================================================

    current_workers = []


    for person in persons:

        worker_id = person["id"]

        person_box = person["box"]


        detected_ppe = set()


        # ----------------------------------------------------
        # ASSOCIATE PPE WITH THIS PERSON
        # ----------------------------------------------------
        #
        # We use the centre of the PPE box.
        #
        # If that centre is inside the worker's Person box,
        # the PPE belongs to that worker.
        #

        for ppe in ppe_detections:

            center = box_center(
                ppe["box"]
            )


            if point_inside_box(
                center,
                person_box
            ):

                detected_ppe.add(
                    ppe["class"]
                )


        missing_ppe = (
            REQUIRED_PPE
            - detected_ppe
        )


        if missing_ppe:

            status = "UNSAFE"

        else:

            status = "SAFE"


        # ----------------------------------------------------
        # GET / CREATE WORKER STATE
        # ----------------------------------------------------

        if worker_id not in worker_states:

            worker_states[worker_id] = {

                "unsafe_since": None,

                "last_api_call": None,

                "api_calls": 0,

                "missing_ppe": set(),

                "detected_ppe": set(),

                "last_seen": current_time,
            }


        state = worker_states[
            worker_id
        ]


        state["last_seen"] = (
            current_time
        )


        # ----------------------------------------------------
        # SAFE WORKER
        # ----------------------------------------------------

        if status == "SAFE":

            state["unsafe_since"] = None

            state["last_api_call"] = None

            state["missing_ppe"] = set()

            state["detected_ppe"] = (
                set(detected_ppe)
            )


            unsafe_duration = 0.0


        # ----------------------------------------------------
        # UNSAFE WORKER
        # ----------------------------------------------------

        else:

            # -----------------------------------------------
            # START NEW UNSAFE PERIOD
            # -----------------------------------------------

            if state["unsafe_since"] is None:

                state["unsafe_since"] = (
                    current_time
                )

                state["last_api_call"] = None


            unsafe_duration = (
                current_time
                - state["unsafe_since"]
            )


            state["missing_ppe"] = (
                set(missing_ppe)
            )

            state["detected_ppe"] = (
                set(detected_ppe)
            )


        # ----------------------------------------------------
        # SAVE WORKER FOR API
        # ----------------------------------------------------

        current_workers.append(
            {
                "worker_id": worker_id,

                "status": status,

                "duration_seconds": round(
                    unsafe_duration,
                    2
                ),

                "missing_ppe": sorted(
                    list(missing_ppe)
                ),

                "detected_ppe": sorted(
                    list(detected_ppe)
                ),
            }
        )


    # ========================================================
    # API SNAPSHOT
    # ========================================================

    worker_count = len(
        current_workers
    )


    unsafe_workers = [

        worker

        for worker in current_workers

        if worker["status"] == "UNSAFE"
    ]


    safe_workers = [

        worker

        for worker in current_workers

        if worker["status"] == "SAFE"
    ]


    unsafe_count = len(
        unsafe_workers
    )


    safe_count = len(
        safe_workers
    )


    # ========================================================
    # SEND VIOLATION EVERY 15 SECONDS
    # ========================================================
    #
    # IMPORTANT:
    #
    # This is PER WORKER.
    #
    # Worker 1 unsafe:
    #
    #   15 sec -> call
    #   30 sec -> call
    #   45 sec -> call
    #
    # Worker 2 unsafe:
    #
    #   15 sec -> call
    #   30 sec -> call
    #
    # Worker 1 and Worker 2 therefore have independent timers.
    #

    for worker in current_workers:

        if worker["status"] != "UNSAFE":

            continue


        worker_id = worker[
            "worker_id"
        ]


        state = worker_states[
            worker_id
        ]


        unsafe_duration = (
            current_time
            - state["unsafe_since"]
        )


        # ----------------------------------------------------
        # FIRST API CALL
        #
        # Only after 15 seconds.
        # ----------------------------------------------------

        if unsafe_duration < VIOLATION_INTERVAL:

            continue


        # ----------------------------------------------------
        # DETERMINE WHETHER AN API CALL IS DUE
        # ----------------------------------------------------

        if state["last_api_call"] is None:

            api_due = True

        else:

            time_since_last_call = (
                current_time
                - state["last_api_call"]
            )

            api_due = (
                time_since_last_call
                >= VIOLATION_INTERVAL
            )


        if not api_due:

            continue


        # ----------------------------------------------------
        # SEND API
        # ----------------------------------------------------

        success = send_violation(

            worker_id=worker_id,

            duration=unsafe_duration,

            missing_ppe=state[
                "missing_ppe"
            ],

            detected_ppe=state[
                "detected_ppe"
            ],

            workers_snapshot=current_workers,
        )


        # ----------------------------------------------------
        # IMPORTANT:
        #
        # Even if API is offline, don't retry every frame.
        #
        # We mark this interval as attempted.
        #
        # Next attempt happens after another 15 seconds.
        # ----------------------------------------------------

        state["last_api_call"] = (
            current_time
        )


        if success:

            state["api_calls"] += 1


    # ========================================================
    # GLOBAL STATUS
    # ========================================================

    if worker_count == 0:

        global_status = "NO WORKER"

    elif unsafe_count > 0:

        global_status = "UNSAFE"

    else:

        global_status = "SAFE"


    # ========================================================
    # DRAW PERSON / PPE BOXES
    # ========================================================

    annotated = frame.copy()


    # --------------------------------------------------------
    # DRAW PPE
    # --------------------------------------------------------

    for ppe in ppe_detections:

        x1, y1, x2, y2 = ppe[
            "box"
        ]

        class_name = ppe[
            "class"
        ]

        confidence = ppe[
            "confidence"
        ]


        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            BLUE,
            2
        )


        label = (
            f"{class_name} "
            f"{confidence:.2f}"
        )


        draw_label(
            annotated,
            label,
            x1,
            y1,
            BLUE
        )


    # --------------------------------------------------------
    # DRAW WORKERS
    # --------------------------------------------------------

    for person in persons:

        worker_id = person["id"]

        x1, y1, x2, y2 = (
            person["box"]
        )


        worker_info = next(
            (
                worker
                for worker
                in current_workers
                if worker["worker_id"]
                == worker_id
            ),
            None
        )


        if worker_info is None:

            continue


        status = worker_info[
            "status"
        ]


        if status == "SAFE":

            box_color = GREEN

        else:

            box_color = RED


        # ----------------------------------------------------
        # PERSON BOX
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            box_color,
            3
        )


        # ----------------------------------------------------
        # WORKER LABEL
        # ----------------------------------------------------

        worker_label = (
            f"Worker {worker_id}"
        )


        draw_label(
            annotated,
            worker_label,
            x1,
            y1,
            box_color
        )


        # ----------------------------------------------------
        # PPE STATUS INSIDE WORKER BOX
        # ----------------------------------------------------

        status_y = (
            y1 + 35
        )


        for ppe_name in PPE_ORDER:

            if ppe_name in worker_info[
                "detected_ppe"
            ]:

                text = (
                    f"{ppe_name}: OK"
                )

                text_color = GREEN

            else:

                text = (
                    f"{ppe_name}: MISSING"
                )

                text_color = RED


            draw_text(
                annotated,
                text,
                (x1 + 5, status_y),
                text_color,
                0.55,
                2
            )


            status_y += 24


        # ----------------------------------------------------
        # UNSAFE TIMER
        # ----------------------------------------------------

        if status == "UNSAFE":

            duration = worker_info[
                "duration_seconds"
            ]


            draw_text(
                annotated,
                f"Unsafe: {duration:.1f}s",
                (x1 + 5, y2 - 10),
                RED,
                0.65,
                2
            )


    # ========================================================
    # TOP STATUS PANEL
    # ========================================================

    panel_height = 220


    cv2.rectangle(
        annotated,
        (0, 0),
        (
            annotated.shape[1],
            panel_height
        ),
        BLACK,
        -1
    )


    # --------------------------------------------------------
    # GLOBAL STATUS
    # --------------------------------------------------------

    if global_status == "SAFE":

        status_color = GREEN

    elif global_status == "UNSAFE":

        status_color = RED

    else:

        status_color = WHITE


    draw_text(
        annotated,
        f"STATUS: {global_status}",
        (20, 55),
        status_color,
        1.15,
        3
    )


    # --------------------------------------------------------
    # WORKER COUNTS
    # --------------------------------------------------------

    draw_text(
        annotated,
        f"Workers: {worker_count}",
        (20, 105),
        WHITE,
        0.7,
        2
    )


    draw_text(
        annotated,
        f"Safe: {safe_count}",
        (270, 105),
        GREEN,
        0.7,
        2
    )


    draw_text(
        annotated,
        f"Unsafe: {unsafe_count}",
        (470, 105),
        RED,
        0.7,
        2
    )


    # --------------------------------------------------------
    # API STATUS
    # --------------------------------------------------------

    if api_online:

        api_text = "API: ONLINE"

        api_color = GREEN

    else:

        api_text = "API: OFFLINE"

        api_color = RED


    draw_text(
        annotated,
        api_text,
        (
            annotated.shape[1] - 300,
            50
        ),
        api_color,
        0.75,
        2
    )


    # --------------------------------------------------------
    # API CALL COUNT
    # --------------------------------------------------------

    draw_text(
        annotated,
        f"API calls: {total_api_calls}",
        (
            annotated.shape[1] - 300,
            100
        ),
        WHITE,
        0.7,
        2
    )


    # --------------------------------------------------------
    # API INTERVAL
    # --------------------------------------------------------

    draw_text(
        annotated,
        "Violation API interval: 15s",
        (20, 160),
        WHITE,
        0.65,
        2
    )


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    draw_text(
        annotated,
        f"FPS: {fps:.1f}",
        (
            annotated.shape[1] - 170,
            160
        ),
        WHITE,
        0.65,
        2
    )


    # ========================================================
    # FPS CALCULATION
    # ========================================================

    fps_frames += 1

    elapsed = (
        current_time
        - fps_timer
    )


    if elapsed >= 1.0:

        fps = (
            fps_frames
            / elapsed
        )

        fps_frames = 0

        fps_timer = current_time


    # ========================================================
    # DISPLAY
    # ========================================================

    cv2.imshow(
        WINDOW_NAME,
        annotated
    )


    # ========================================================
    # KEYBOARD
    # ========================================================

    key = cv2.waitKey(1) & 0xFF


    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()


print()
print("=" * 70)
print("VISIONSHIELD STOPPED")
print("=" * 70)

print(
    f"Total API calls: {total_api_calls}"
)

print("=" * 70)
