
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

#INPUT_VIDEO = BASE / "Construction Work  Construction Site  Building  Free Stock Footage Video380.mp4"
INPUT_VIDEO = BASE / "CINEMATIC CORPORATE VIDEO  Factory B ROLL.mp4"
OUTPUT_VIDEO = BASE / "visionshield_output.mp4"


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

PPE_CONF = {
    "helmet": 0.25,
    "goggles": 0.15,
    "gloves": 0.15,
    "boots": 0.15,
    "vest": 0.20,
}

PERSON_EXPAND_X = 0.12
PERSON_EXPAND_Y = 0.12

IMG_SIZE = 640


print("=" * 70)
print("VISIONSHIELD - VIDEO PPE TEST")
print("=" * 70)

print("\nModel:")
print(MODEL_PATH)

print("\nInput:")
print(INPUT_VIDEO)

print("\nOutput:")
print(OUTPUT_VIDEO)

if not MODEL_PATH.exists():
    print("\nERROR: Model not found!")
    print(MODEL_PATH)
    raise SystemExit

if not INPUT_VIDEO.exists():
    print("\nERROR: Input video not found!")
    print(INPUT_VIDEO)
    print("\nPut your video here or change INPUT_VIDEO.")
    raise SystemExit


print("\nLoading model...")

model = YOLO(str(MODEL_PATH))

print("Model loaded.")


cap = cv2.VideoCapture(str(INPUT_VIDEO))

if not cap.isOpened():
    print("\nERROR: Could not open video.")
    raise SystemExit


fps_input = cap.get(cv2.CAP_PROP_FPS)

if fps_input <= 0:
    fps_input = 25.0

frame_width = int(
    cap.get(cv2.CAP_PROP_FRAME_WIDTH)
)

frame_height = int(
    cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
)

total_frames = int(
    cap.get(cv2.CAP_PROP_FRAME_COUNT)
)

duration = total_frames / fps_input


print("\nVideo information:")
print(f"Resolution : {frame_width} x {frame_height}")
print(f"FPS        : {fps_input:.2f}")
print(f"Frames     : {total_frames}")
print(f"Duration   : {duration:.1f} seconds")


fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(
    str(OUTPUT_VIDEO),
    fourcc,
    fps_input,
    (frame_width, frame_height)
)

if not writer.isOpened():
    print("\nERROR: Could not create output video.")
    cap.release()
    raise SystemExit


def intersection_over_area(inner_box, outer_box):

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


def expand_box(box):

    x1, y1, x2, y2 = box

    w = x2 - x1
    h = y2 - y1

    dx = int(w * PERSON_EXPAND_X)
    dy = int(h * PERSON_EXPAND_Y)

    return (
        max(0, x1 - dx),
        max(0, y1 - dy),
        min(frame_width - 1, x2 + dx),
        min(frame_height - 1, y2 + dy),
    )


def center_inside(box, person_box):

    x1, y1, x2, y2 = box

    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2

    px1, py1, px2, py2 = person_box

    return (
        px1 <= cx <= px2
        and
        py1 <= cy <= py2
    )


def belongs_to_worker(ppe_box, worker_box):

    expanded = expand_box(worker_box)

    # Center test
    if center_inside(
        ppe_box,
        expanded
    ):
        return True

    # Overlap test
    overlap = intersection_over_area(
        ppe_box,
        expanded
    )

    return overlap >= 0.20


def draw_label(
    image,
    text,
    x,
    y,
    color,
    scale=0.55,
    thickness=2
):

    font = cv2.FONT_HERSHEY_SIMPLEX

    (tw, th), baseline = cv2.getTextSize(
        text,
        font,
        scale,
        thickness
    )

    y = max(th + 5, y)

    cv2.rectangle(
        image,
        (x, y - th - baseline - 5),
        (x + tw + 8, y + 4),
        (0, 0, 0),
        -1
    )

    cv2.putText(
        image,
        text,
        (x + 4, y),
        font,
        scale,
        color,
        thickness,
        cv2.LINE_AA
    )

frame_number = 0

start_time = time.time()

print("\nProcessing video...")
print("Press Q in the preview window to stop.\n")


while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame_number += 1

    results = model.predict(
        source=frame,
        conf=CONF,
        imgsz=IMG_SIZE,
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
                ),
            }

            
            if cls_id == PERSON_ID:

                if confidence >= CONF:
                    workers.append(detection)

            elif cls_id in PPE_IDS.values():

                name = CLASS_NAMES[cls_id]

                if confidence >= PPE_CONF[name]:

                    ppe_detections.append(
                        detection
                    )

          
            elif cls_id in VIOLATION_IDS.values():

                if confidence >= CONF:

                    violation_detections.append(
                        detection
                    )

 
    for ppe in ppe_detections:

        x1, y1, x2, y2 = ppe["box"]

        name = ppe["name"]
        confidence = ppe["confidence"]

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

    safe_count = 0
    unsafe_count = 0

    for worker_index, worker in enumerate(workers):

        worker_box = worker["box"]

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

            if belongs_to_worker(
                ppe["box"],
                worker_box
            ):

                current = detected_ppe[name]

                if (
                    current is None
                    or
                    ppe["confidence"]
                    >
                    current["confidence"]
                ):

                    detected_ppe[name] = ppe

      
        worker_violations = set()

        for violation in violation_detections:

            if belongs_to_worker(
                violation["box"],
                worker_box
            ):

                worker_violations.add(
                    violation["name"]
                )

   
        missing = []

        if detected_ppe["helmet"] is None:
            missing.append("helmet")

        if detected_ppe["goggles"] is None:
            missing.append("goggles")

        if detected_ppe["gloves"] is None:
            missing.append("gloves")

        if detected_ppe["boots"] is None:
            missing.append("boots")

        if detected_ppe["vest"] is None:
            missing.append("vest")

        # Explicit violation classes override positive status.
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
            status = "UNSAFE"
            status_color = (0, 0, 255)
        else:
            safe_count += 1
            status = "SAFE"
            status_color = (0, 220, 0)

        x1, y1, x2, y2 = worker_box

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            status_color,
            3
        )


        header = (
            f"Worker {worker_index + 1} | {status}"
        )

        draw_label(
            frame,
            header,
            x1,
            y1,
            status_color,
            0.62,
            2
        )

  
        if unsafe:

            missing_text = (
                "Missing: "
                + ", ".join(missing)
            )

            draw_label(
                frame,
                missing_text,
                x1,
                y1 + 35,
                (0, 0, 255),
                0.48,
                2
            )

        else:

            draw_label(
                frame,
                "All required PPE detected",
                x1,
                y1 + 35,
                (0, 220, 0),
                0.48,
                2
            )

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

    elapsed = time.time() - start_time

    processing_fps = (
        frame_number / elapsed
        if elapsed > 0
        else 0
    )

    draw_label(
        frame,
        f"Workers: {len(workers)}",
        20,
        38,
        (255, 255, 255),
        0.55,
        2
    )

    draw_label(
        frame,
        f"SAFE: {safe_count}",
        20,
        70,
        (0, 220, 0),
        0.55,
        2
    )

    draw_label(
        frame,
        f"UNSAFE: {unsafe_count}",
        20,
        102,
        (0, 0, 255),
        0.55,
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
        global_color = (0, 220, 0)

    draw_label(
        frame,
        global_status,
        20,
        frame_height - 20,
        global_color,
        0.80,
        3
    )


    if total_frames > 0:

        progress = (
            frame_number / total_frames
        ) * 100

    else:

        progress = 0

    if frame_number % 30 == 0:

        print(
            f"\rFrame "
            f"{frame_number}/{total_frames} "
            f"({progress:.1f}%) "
            f"| Workers: {len(workers)} "
            f"| Safe: {safe_count} "
            f"| Unsafe: {unsafe_count}",
            end="",
            flush=True
        )

    writer.write(frame)

    cv2.imshow(
        "VisionShield - Video Test",
        frame
    )

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        print("\n\nStopped by user.")
        break



cap.release()
writer.release()
cv2.destroyAllWindows()

print("\n")
print("=" * 70)
print("VIDEO PROCESSING COMPLETE")
print("=" * 70)

print("\nOutput saved to:")
print(OUTPUT_VIDEO)

print("\nFrames processed:")
print(frame_number)

print("\nDone.")
