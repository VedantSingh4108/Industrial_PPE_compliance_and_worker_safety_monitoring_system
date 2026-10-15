
import cv2
import time
from pathlib import Path
from ultralytics import YOLO


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


INPUT_VIDEO = (
    BASE
    / "CINEMATIC CORPORATE VIDEO  Factory B ROLL.mp4"
)

# INPUT_VIDEO = (
#     BASE
#     / "Construction Work  Construction Site  Building  Free Stock Footage Video380.mp4"
# )


OUTPUT_VIDEO = (
    BASE
    / "visionshield_6class_video_output.mp4"
)

CLASS_NAMES = {
    0: "helmet",
    1: "gloves",
    2: "vest",
    3: "boots",
    4: "goggles",
    5: "Person",
}


PERSON_ID = 5


PPE_IDS = {
    "helmet": 0,
    "gloves": 1,
    "vest": 2,
    "boots": 3,
    "goggles": 4,
}



GENERAL_CONF = 0.15


PPE_CONF = {
    "helmet": 0.25,
    "gloves": 0.15,
    "vest": 0.20,
    "boots": 0.15,
    "goggles": 0.15,
}


IMAGE_SIZE = 640



PERSON_EXPAND_X = 0.12
PERSON_EXPAND_Y = 0.12

OVERLAP_THRESHOLD = 0.20



print("=" * 70)
print("VISIONSHIELD - 6 CLASS PPE VIDEO TEST")
print("=" * 70)


print("\nModel:")
print(MODEL_PATH)


print("\nInput video:")
print(INPUT_VIDEO)


print("\nOutput video:")
print(OUTPUT_VIDEO)



if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"\nMODEL NOT FOUND:\n{MODEL_PATH}"
    )


if not INPUT_VIDEO.exists():

    raise FileNotFoundError(
        f"\nVIDEO NOT FOUND:\n{INPUT_VIDEO}"
    )


print("\nLoading model...")

model = YOLO(
    str(MODEL_PATH)
)

print("Model loaded successfully.")


print("\nModel classes:")

for class_id, name in model.names.items():

    print(
        f"{class_id}: {name}"
    )


cap = cv2.VideoCapture(
    str(INPUT_VIDEO)
)


if not cap.isOpened():

    raise RuntimeError(
        "Could not open input video."
    )



fps = cap.get(
    cv2.CAP_PROP_FPS
)

if fps <= 0:

    fps = 25.0


frame_width = int(
    cap.get(
        cv2.CAP_PROP_FRAME_WIDTH
    )
)


frame_height = int(
    cap.get(
        cv2.CAP_PROP_FRAME_HEIGHT
    )
)


total_frames = int(
    cap.get(
        cv2.CAP_PROP_FRAME_COUNT
    )
)


duration = (
    total_frames / fps
    if fps > 0
    else 0
)


print("\nVideo information:")
print(
    f"Resolution : {frame_width} x {frame_height}"
)
print(
    f"FPS        : {fps:.2f}"
)
print(
    f"Frames     : {total_frames}"
)
print(
    f"Duration   : {duration:.1f} seconds"
)

fourcc = cv2.VideoWriter_fourcc(
    *"mp4v"
)


writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps,
    (
        frame_width,
        frame_height
    )
)


if not writer.isOpened():

    cap.release()

    raise RuntimeError(
        "Could not create output video."
    )



def intersection_over_area(
    inner_box,
    outer_box
):

    ix1, iy1, ix2, iy2 = inner_box

    ox1, oy1, ox2, oy2 = outer_box


    x1 = max(
        ix1,
        ox1
    )

    y1 = max(
        iy1,
        oy1
    )

    x2 = min(
        ix2,
        ox2
    )

    y2 = min(
        iy2,
        oy2
    )


    if x2 <= x1 or y2 <= y1:

        return 0.0


    intersection = (
        x2 - x1
    ) * (
        y2 - y1
    )


    inner_area = max(
        1,
        (
            ix2 - ix1
        ) * (
            iy2 - iy1
        )
    )


    return (
        intersection
        / inner_area
    )



def expand_person_box(
    box
):

    x1, y1, x2, y2 = box


    width = x2 - x1
    height = y2 - y1


    dx = int(
        width
        * PERSON_EXPAND_X
    )

    dy = int(
        height
        * PERSON_EXPAND_Y
    )


    return (
        max(
            0,
            x1 - dx
        ),

        max(
            0,
            y1 - dy
        ),

        min(
            frame_width - 1,
            x2 + dx
        ),

        min(
            frame_height - 1,
            y2 + dy
        ),
    )


def center_inside(
    box,
    person_box
):

    x1, y1, x2, y2 = box


    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2


    px1, py1, px2, py2 = (
        person_box
    )


    return (
        px1 <= center_x <= px2
        and
        py1 <= center_y <= py2
    )



def belongs_to_worker(
    ppe_box,
    worker_box
):

    expanded = expand_person_box(
        worker_box
    )


    # First: center test
    if center_inside(
        ppe_box,
        expanded
    ):

        return True


    # Second: overlap test
    overlap = intersection_over_area(
        ppe_box,
        expanded
    )


    return (
        overlap >= OVERLAP_THRESHOLD
    )


def draw_label(
    image,
    text,
    x,
    y,
    color,
    scale=0.55,
    thickness=2
):

    font = (
        cv2.FONT_HERSHEY_SIMPLEX
    )


    (
        text_width,
        text_height
    ), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )


    # Keep label inside frame
    y = max(
        text_height + 5,
        y
    )


    # Black background
    cv2.rectangle(
        image,
        (
            x,
            y
            - text_height
            - baseline
            - 5
        ),
        (
            x
            + text_width
            + 8,
            y + 4
        ),
        (0, 0, 0),
        -1
    )


    cv2.putText(
        image,
        text,
        (
            x + 4,
            y
        ),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )


# ============================================================
# PROCESS VIDEO
# ============================================================

frame_number = 0

start_time = time.time()


print("\nProcessing video...")
print("Press Q to stop.\n")


while True:

    ret, frame = cap.read()


    if not ret:

        break


    frame_number += 1


    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = model.predict(
        source=frame,
        conf=GENERAL_CONF,
        imgsz=IMAGE_SIZE,
        verbose=False
    )


    result = results[0]


    workers = []

    ppe_detections = []


    # --------------------------------------------------------
    # READ DETECTIONS
    # --------------------------------------------------------

    if result.boxes is not None:

        for box in result.boxes:

            class_id = int(
                box.cls[0].item()
            )

            confidence = float(
                box.conf[0].item()
            )


            if class_id not in CLASS_NAMES:

                continue


            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist()
            )


            class_name = (
                CLASS_NAMES[class_id]
            )


            detection = {
                "class_id": class_id,
                "name": class_name,
                "confidence": confidence,
                "box": (
                    x1,
                    y1,
                    x2,
                    y2
                ),
            }


            # ------------------------------------------------
            # PERSON
            # ------------------------------------------------

            if class_id == PERSON_ID:

                workers.append(
                    detection
                )


            # ------------------------------------------------
            # PPE
            # ------------------------------------------------

            elif class_id in PPE_IDS.values():

                required_conf = PPE_CONF[
                    class_name
                ]


                if confidence >= required_conf:

                    ppe_detections.append(
                        detection
                    )


    # ========================================================
    # DRAW PPE DETECTIONS
    # ========================================================

    for ppe in ppe_detections:

        x1, y1, x2, y2 = (
            ppe["box"]
        )


        name = ppe["name"]

        confidence = (
            ppe["confidence"]
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            (255, 120, 0),
            2
        )


        draw_label(
            frame,
            f"{name} {confidence:.2f}",
            x1,
            y1,
            (255, 120, 0),
            0.48,
            1
        )


    # ========================================================
    # WORKER SAFETY
    # ========================================================

    safe_count = 0

    unsafe_count = 0


    for worker_index, worker in enumerate(
        workers
    ):

        worker_box = worker["box"]


        # ----------------------------------------------------
        # PPE FOR THIS WORKER
        # ----------------------------------------------------

        detected_ppe = {

            "helmet": None,

            "goggles": None,

            "gloves": None,

            "boots": None,

            "vest": None,
        }


        # ----------------------------------------------------
        # ASSIGN PPE TO WORKER
        # ----------------------------------------------------

        for ppe in ppe_detections:

            name = ppe["name"]


            if name not in detected_ppe:

                continue


            if belongs_to_worker(
                ppe["box"],
                worker_box
            ):

                current = (
                    detected_ppe[name]
                )


                # Keep highest confidence
                if (
                    current is None
                    or
                    ppe["confidence"]
                    >
                    current["confidence"]
                ):

                    detected_ppe[name] = (
                        ppe
                    )


        # ----------------------------------------------------
        # FIND MISSING PPE
        # ----------------------------------------------------

        missing = []


        for ppe_name in [
            "helmet",
            "goggles",
            "gloves",
            "boots",
            "vest",
        ]:

            if (
                detected_ppe[ppe_name]
                is None
            ):

                missing.append(
                    ppe_name
                )


        # ----------------------------------------------------
        # SAFETY DECISION
        # ----------------------------------------------------

        unsafe = (
            len(missing) > 0
        )


        if unsafe:

            unsafe_count += 1

            status = "UNSAFE"

            status_color = (
                0,
                0,
                255
            )

        else:

            safe_count += 1

            status = "SAFE"

            status_color = (
                0,
                220,
                0
            )


        # ----------------------------------------------------
        # WORKER BOX
        # ----------------------------------------------------

        x1, y1, x2, y2 = (
            worker_box
        )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            status_color,
            3
        )


        # ----------------------------------------------------
        # WORKER HEADER
        # ----------------------------------------------------

        draw_label(
            frame,
            f"Worker {worker_index + 1} | {status}",
            x1,
            y1,
            status_color,
            0.62,
            2
        )


        # ----------------------------------------------------
        # MISSING PPE
        # ----------------------------------------------------

        if unsafe:

            missing_text = (
                "Missing: "
                + ", ".join(missing)
            )


            draw_label(
                frame,
                missing_text,
                x1,
                y1 + 38,
                (0, 0, 255),
                0.48,
                2
            )


        else:

            draw_label(
                frame,
                "All required PPE detected",
                x1,
                y1 + 38,
                (0, 220, 0),
                0.48,
                2
            )


    # ========================================================
    # TOP INFORMATION PANEL
    # ========================================================

    overlay = frame.copy()


    cv2.rectangle(
        overlay,
        (10, 10),
        (300, 125),
        (0, 0, 0),
        -1
    )


    frame = cv2.addWeighted(
        overlay,
        0.70,
        frame,
        0.30,
        0
    )


    # --------------------------------------------------------
    # PROCESSING FPS
    # --------------------------------------------------------

    elapsed = (
        time.time()
        - start_time
    )


    processing_fps = (
        frame_number / elapsed
        if elapsed > 0
        else 0
    )


    # --------------------------------------------------------
    # WORKER COUNT
    # --------------------------------------------------------

    draw_label(
        frame,
        f"Workers: {len(workers)}",
        20,
        38,
        (255, 255, 255),
        0.55,
        2
    )


    # --------------------------------------------------------
    # SAFE COUNT
    # --------------------------------------------------------

    draw_label(
        frame,
        f"SAFE: {safe_count}",
        20,
        70,
        (0, 220, 0),
        0.55,
        2
    )


    # --------------------------------------------------------
    # UNSAFE COUNT
    # --------------------------------------------------------

    draw_label(
        frame,
        f"UNSAFE: {unsafe_count}",
        20,
        102,
        (0, 0, 255),
        0.55,
        2
    )


    # --------------------------------------------------------
    # GLOBAL STATUS
    # --------------------------------------------------------

    if len(workers) == 0:

        global_status = (
            "STATUS: NO WORKER"
        )

        global_color = (
            0,
            165,
            255
        )


    elif unsafe_count > 0:

        global_status = (
            "STATUS: UNSAFE"
        )

        global_color = (
            0,
            0,
            255
        )


    else:

        global_status = (
            "STATUS: SAFE"
        )

        global_color = (
            0,
            220,
            0
        )


    # --------------------------------------------------------
    # GLOBAL STATUS DISPLAY
    # --------------------------------------------------------

    draw_label(
        frame,
        global_status,
        20,
        frame_height - 20,
        global_color,
        0.80,
        3
    )


    # --------------------------------------------------------
    # CONSOLE PROGRESS
    # --------------------------------------------------------

    if total_frames > 0:

        progress = (
            frame_number
            / total_frames
        ) * 100

    else:

        progress = 0


    if frame_number % 30 == 0:

        print(
            f"\rFrame "
            f"{frame_number}/"
            f"{total_frames} "
            f"({progress:.1f}%) "
            f"| Workers: "
            f"{len(workers)} "
            f"| Safe: "
            f"{safe_count} "
            f"| Unsafe: "
            f"{unsafe_count} "
            f"| FPS: "
            f"{processing_fps:.1f}",
            end="",
            flush=True
        )


    # --------------------------------------------------------
    # SAVE FRAME
    # --------------------------------------------------------

    writer.write(
        frame
    )


    # --------------------------------------------------------
    # SHOW PREVIEW
    # --------------------------------------------------------

    cv2.imshow(
        "VisionShield - 6 Class Video Test",
        frame
    )


    key = (
        cv2.waitKey(1)
        & 0xFF
    )


    if key == ord("q"):

        print(
            "\n\nStopped by user."
        )

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

writer.release()

cv2.destroyAllWindows()


# ============================================================
# FINISHED
# ============================================================

print("\n")

print("=" * 70)

print("VIDEO PROCESSING COMPLETE")

print("=" * 70)


print("\nOutput saved to:")

print(
    OUTPUT_VIDEO
)


print("\nFrames processed:")

print(
    frame_number
)


print("\nDone.")
