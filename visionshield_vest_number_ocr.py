"""VisionShield: PPE detection + more robust vest-number OCR prototype.

Uses the existing six-class PPE checkpoint, pretrained YOLO person detector,
and EasyOCR. Existing checkpoints are read only and are never modified.

Notes:
- OCR is heuristic; verify worker numbers manually before consequential use.
- This prototype reports a number for the largest detected person only.
- Repeated OCR readings are temporally voted to reduce one-frame misreads.
"""

import argparse
from collections import Counter, deque
from pathlib import Path
import re
import sys

import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_PPE_MODEL = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\runs\detect\visionshield_6class_yolo11n_10ep\weights\best.pt"
)
DEFAULT_PERSON_MODEL = "yolo11n.pt"
DEFAULT_VIDEO = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\CINEMATIC CORPORATE VIDEO  Factory B ROLL.mp4"
)


def parse_args():
    parser = argparse.ArgumentParser(description="VisionShield PPE + vest-number OCR")
    parser.add_argument("source", nargs="?", default=DEFAULT_VIDEO,
                        help="Video file path or camera index, e.g. 0")
    parser.add_argument("--ppe-model", default=DEFAULT_PPE_MODEL)
    parser.add_argument("--person-model", default=DEFAULT_PERSON_MODEL)
    parser.add_argument("--person-conf", type=float, default=0.35)
    parser.add_argument("--ppe-conf", type=float, default=0.25)
    parser.add_argument("--ocr-every", type=int, default=5,
                        help="Attempt OCR every N frames; smaller is more frequent")
    parser.add_argument("--ocr-min-conf", type=float, default=0.45,
                        help="Minimum EasyOCR confidence before a candidate is considered")
    return parser.parse_args()


def clamp_box(x1, y1, x2, y2, width, height):
    x1 = max(0, min(width - 1, int(x1)))
    y1 = max(0, min(height - 1, int(y1)))
    x2 = max(0, min(width, int(x2)))
    y2 = max(0, min(height, int(y2)))
    return x1, y1, x2, y2


def make_ocr_images(region):
    """Create several readable variants from a torso crop."""
    if region is None or region.size == 0:
        return []
    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
    # Increase contrast without aggressively destroying thin digit strokes.
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    scale = 4 if max(gray.shape) < 350 else 2
    enlarged = cv2.resize(gray, None, fx=scale, fy=scale,
                          interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(enlarged, (3, 3), 0)
    _, otsu = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(
        blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 31, 7
    )
    return [enlarged, otsu, 255 - otsu, adaptive, 255 - adaptive]


def find_vest_number(person_crop, reader, min_conf=0.45):
    """Return the best plausible digit string and confidence from a person crop."""
    if person_crop is None or person_crop.size == 0:
        return None, 0.0
    h, w = person_crop.shape[:2]
    if h < 35 or w < 20:
        return None, 0.0

    # Prioritize chest/upper torso, then widen to include front/back vest areas.
    regions = [
        person_crop[int(h * 0.18):int(h * 0.62), int(w * 0.08):int(w * 0.92)],
        person_crop[int(h * 0.10):int(h * 0.78), int(w * 0.03):int(w * 0.97)],
        person_crop[int(h * 0.08):int(h * 0.88), :],
    ]
    candidates = []
    for region_index, region in enumerate(regions):
        for image in make_ocr_images(region):
            try:
                results = reader.readtext(
                    image, allowlist="0123456789", detail=1, paragraph=False,
                    decoder="greedy", text_threshold=0.45,
                    low_text=0.25, link_threshold=0.35
                )
            except Exception:
                continue
            for _polygon, raw_text, confidence in results:
                digits = re.sub(r"\D", "", raw_text)
                confidence = float(confidence)
                # Worker numbers are usually short. Reject long strings that are
                # more likely to be unrelated text or multiple adjacent objects.
                if 1 <= len(digits) <= 4 and confidence >= min_conf:
                    # Prefer higher confidence and torso-focused regions.
                    adjusted = confidence + (0.025 if region_index == 0 else 0.0)
                    candidates.append((digits, confidence, adjusted))
        # If focused torso crop produced strong candidates, don't over-search.
        if candidates and max(c[1] for c in candidates) >= 0.82:
            break

    if not candidates:
        return None, 0.0
    # Combine repeated identical readings across image variants, weighting confidence.
    scores = {}
    best_conf = {}
    for digits, confidence, adjusted in candidates:
        scores[digits] = scores.get(digits, 0.0) + adjusted ** 2
        best_conf[digits] = max(best_conf.get(digits, 0.0), confidence)
    winner = max(scores, key=scores.get)
    return winner, best_conf[winner]


def draw_label(frame, text, x, y, color=(0, 255, 255)):
    h, w = frame.shape[:2]
    y = max(24, y)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
    left = max(0, min(w - 1, x))
    top = max(0, min(h - 1, y - th - 9))
    right = max(left + 1, min(w - 1, left + tw + 10))
    bottom = max(top + 1, min(h - 1, y + 3))
    cv2.rectangle(frame, (left, top), (right, bottom), (25, 25, 25), -1)
    cv2.putText(frame, text, (left + 5, y - 3), cv2.FONT_HERSHEY_SIMPLEX,
                0.65, color, 2, cv2.LINE_AA)


def main():
    args = parse_args()
    ppe_path = Path(args.ppe_model)
    if not ppe_path.is_file():
        raise FileNotFoundError(
            f"PPE checkpoint not found:\n{ppe_path}\n"
            "Use --ppe-model followed by the exact path to your existing best.pt."
        )

    print("Loading existing PPE model:", ppe_path)
    ppe_model = YOLO(str(ppe_path))
    print("Loading person detector:", args.person_model)
    person_model = YOLO(args.person_model)

    try:
        import easyocr
    except ImportError as exc:
        raise SystemExit("EasyOCR missing. Run: python -m pip install easyocr") from exc

    print("Loading OCR reader...")
    reader = easyocr.Reader(["en"], gpu=False)

    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video/camera source: {args.source}")

    # Test display capability before expensive frame processing, with a clear fix.
    window_name = "VisionShield - PPE + Vest Number OCR"
    try:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        test = np.zeros((32, 32, 3), dtype=np.uint8)
        cv2.imshow(window_name, test)
        cv2.waitKey(1)
        cv2.destroyWindow(window_name)
    except cv2.error as exc:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
        raise SystemExit(
            "OpenCV cannot create a live display window. In CMD run these commands, "
            "then restart this script:\n"
            "python -m pip uninstall -y opencv-python opencv-python-headless "
            "opencv-contrib-python opencv-contrib-python-headless\n"
            "python -m pip install opencv-python==4.11.0.86\n"
            "The model checkpoints were not changed.\n\n"
            f"OpenCV error: {exc}"
        ) from exc

    print("Live window ready. Press Q or Esc to quit.")
    frame_no = 0
    # A small rolling vote stabilizes OCR results across nearby frames.
    readings = deque(maxlen=9)
    shown_number = None
    shown_conf = 0.0
    last_ocr_frame = -10**9

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_no += 1
            original_frame = frame.copy()
            h, w = frame.shape[:2]

            # PPE model inference and annotation.
            ppe_result = ppe_model.predict(frame, conf=args.ppe_conf, verbose=False)[0]
            if ppe_result.boxes is not None:
                for box in ppe_result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls.item())
                    label = ppe_result.names.get(cls_id, str(cls_id))
                    confidence = float(box.conf.item())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 170, 0), 2)
                    cv2.putText(frame, f"{label} {confidence:.2f}",
                                (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (255, 170, 0), 2, cv2.LINE_AA)

            # Independent person detection (COCO class 0 = person).
            person_result = person_model.predict(
                original_frame, classes=[0], conf=args.person_conf, verbose=False
            )[0]
            people = []
            if person_result.boxes is not None:
                for box in person_result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
                    if x2 > x1 and y2 > y1:
                        people.append((x1, y1, x2, y2, float(box.conf.item())))

            selected = max(
                people, key=lambda p: (p[2] - p[0]) * (p[3] - p[1]), default=None
            )
            for x1, y1, x2, y2, confidence in people:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(frame, f"person {confidence:.2f}",
                            (x1, min(h - 5, max(18, y1 - 5))),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 2,
                            cv2.LINE_AA)

            if selected is not None:
                x1, y1, x2, y2, _ = selected
                interval = max(1, args.ocr_every)
                if frame_no == 1 or frame_no - last_ocr_frame >= interval:
                    crop = original_frame[y1:y2, x1:x2].copy()
                    candidate, candidate_conf = find_vest_number(
                        crop, reader, min_conf=args.ocr_min_conf
                    )
                    last_ocr_frame = frame_no
                    if candidate is not None:
                        readings.append((candidate, candidate_conf))
                        counts = Counter(number for number, _ in readings)
                        winner, count = counts.most_common(1)[0]
                        # Require repeated agreement or a very strong reading.
                        winner_conf = max(c for n, c in readings if n == winner)
                        if count >= 2 or winner_conf >= 0.88:
                            shown_number = winner
                            shown_conf = winner_conf
                    else:
                        # Don't immediately erase a stable number after one bad frame.
                        if len(readings) >= 5 and not any(n == shown_number for n, _ in readings):
                            shown_number = None
                            shown_conf = 0.0

                label = (f"Worker #{shown_number} ({shown_conf:.2f})"
                         if shown_number is not None else "Worker #Unknown")
                draw_label(frame, label, x1, max(30, y1 - 5))
            else:
                # Avoid carrying a previous person's number into an empty scene.
                shown_number = None
                shown_conf = 0.0
                readings.clear()

            try:
                cv2.imshow(window_name, frame)
            except cv2.error as exc:
                raise RuntimeError(
                    "OpenCV lost live-window support while processing. Reinstall "
                    "opencv-python==4.11.0.86 as shown in the startup message."
                ) from exc

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            # Prevent cleanup from masking the original error.
            pass

    print("Finished processing video.")


if __name__ == "__main__":
    main()
