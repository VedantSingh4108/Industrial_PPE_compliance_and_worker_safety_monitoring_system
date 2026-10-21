"""VisionShield live webcam PPE detection + vest-number OCR prototype.

Uses the existing six-class PPE checkpoint, a pretrained YOLO person detector,
and EasyOCR. Does not train, overwrite, or modify any model checkpoint.
Press Q or Esc in the camera window to quit.
"""
import argparse
import re
import sys
from collections import Counter, deque
from pathlib import Path

import cv2
from ultralytics import YOLO

DEFAULT_PPE_MODEL = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\runs\detect\visionshield_6class_yolo11n_10ep\weights\best.pt"
)
DEFAULT_PERSON_MODEL = "yolo11n.pt"


def parse_args():
    p = argparse.ArgumentParser(description="VisionShield live webcam PPE + vest OCR")
    p.add_argument("--camera", type=int, default=0, help="Webcam index (default: 0; try 1 if needed)")
    p.add_argument("--ppe-model", default=DEFAULT_PPE_MODEL)
    p.add_argument("--person-model", default=DEFAULT_PERSON_MODEL)
    p.add_argument("--person-conf", type=float, default=0.35)
    p.add_argument("--ppe-conf", type=float, default=0.25)
    p.add_argument("--ocr-every", type=int, default=8, help="Run OCR every N frames")
    p.add_argument("--width", type=int, default=1280)
    p.add_argument("--height", type=int, default=720)
    return p.parse_args()


def clamp_box(box, width, height):
    x1, y1, x2, y2 = map(int, box)
    return (max(0, min(width - 1, x1)), max(0, min(height - 1, y1)),
            max(0, min(width, x2)), max(0, min(height, y2)))


def read_vest_number(person_crop, reader):
    """Try several upper-body crops and image treatments; return best digit string/confidence."""
    if person_crop is None or person_crop.size == 0:
        return None, 0.0
    h, w = person_crop.shape[:2]
    if h < 35 or w < 25:
        return None, 0.0

    regions = [
        person_crop[int(h * .18):int(h * .66), int(w * .08):int(w * .92)],
        person_crop[int(h * .10):int(h * .78), int(w * .05):int(w * .95)],
        person_crop[:int(h * .72), :],
    ]
    best = (None, 0.0)
    for region in regions:
        if region.size == 0:
            continue
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        scale = 3 if max(gray.shape) < 450 else 2
        gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        blur = cv2.GaussianBlur(clahe, (3, 3), 0)
        _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for image in (gray, clahe, th, 255 - th):
            try:
                results = reader.readtext(image, allowlist="0123456789", detail=1, paragraph=False)
            except Exception:
                continue
            for _bbox, text, confidence in results:
                digits = re.sub(r"\D", "", text)
                confidence = float(confidence)
                # Most worker labels are short. Avoid treating long strings as IDs.
                if 1 <= len(digits) <= 4 and confidence > best[1]:
                    best = (digits, confidence)
        if best[1] >= 0.80:
            break
    # Conservative threshold: don't display weak guesses as an identity.
    if best[0] is not None and best[1] >= 0.45:
        return best
    return None, best[1]


def check_gui_support():
    info = cv2.getBuildInformation()
    gui_line = next((line.strip() for line in info.splitlines() if line.strip().startswith("GUI:")), "GUI information unavailable")
    try:
        cv2.namedWindow("VisionShield camera test", cv2.WINDOW_NORMAL)
        cv2.destroyWindow("VisionShield camera test")
    except cv2.error as e:
        raise RuntimeError(
            f"OpenCV cannot create desktop windows ({gui_line}).\n"
            "Fix OpenCV in this same Python environment with:\n"
            "python -m pip uninstall -y opencv-python opencv-python-headless opencv-contrib-python opencv-contrib-python-headless\n"
            "python -m pip install opencv-python==4.11.0.86\n"
            "Then verify with: python -c \"import cv2; print(cv2.__version__); print([x for x in cv2.getBuildInformation().splitlines() if 'GUI:' in x])\"\n"
            f"Original OpenCV error: {e}"
        ) from e


def open_camera(index, width, height):
    # DirectShow is usually the most reliable backend for Windows webcams.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap.release()
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(
            f"Could not open camera index {index}. Close apps using the camera, check Windows camera permissions, "
            f"then try --camera 1."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    ok, frame = cap.read()
    if not ok or frame is None:
        cap.release()
        raise RuntimeError(f"Camera {index} opened but did not provide a frame. Try --camera 1 or check camera permissions.")
    return cap


def main():
    args = parse_args()
    ppe_path = Path(args.ppe_model)
    if not ppe_path.is_file():
        raise FileNotFoundError(f"PPE checkpoint not found: {ppe_path}")

    print("Checking OpenCV live-window support...")
    check_gui_support()
    print("Loading existing PPE model:", ppe_path)
    ppe_model = YOLO(str(ppe_path))
    print("Loading person detector:", args.person_model)
    person_model = YOLO(args.person_model)
    try:
        import easyocr
    except ImportError as exc:
        raise SystemExit("EasyOCR is missing. Run: python -m pip install easyocr") from exc
    print("Loading OCR reader (first run may take a little longer)...")
    reader = easyocr.Reader(["en"], gpu=False)

    print(f"Opening live webcam index {args.camera}...")
    cap = open_camera(args.camera, args.width, args.height)
    history = deque(maxlen=7)
    cached_number, cached_conf = None, 0.0
    frame_no = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("No frame received from webcam; stopping.")
                break
            frame_no += 1
            original = frame.copy()
            h, w = frame.shape[:2]

            ppe_result = ppe_model.predict(frame, conf=args.ppe_conf, verbose=False)[0]
            if ppe_result.boxes is not None:
                for box in ppe_result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls_id = int(box.cls.item())
                    label = ppe_result.names.get(cls_id, str(cls_id))
                    conf = float(box.conf.item())
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 150, 0), 2)
                    cv2.putText(frame, f"{label} {conf:.2f}", (x1, max(18, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 150, 0), 2)

            people_result = person_model.predict(frame, classes=[0], conf=args.person_conf, verbose=False)[0]
            people = []
            if people_result.boxes is not None:
                for box in people_result.boxes:
                    coords = clamp_box(box.xyxy[0].tolist(), w, h)
                    x1, y1, x2, y2 = coords
                    if x2 > x1 and y2 > y1:
                        people.append((x1, y1, x2, y2, float(box.conf.item())))

            selected = max(people, key=lambda p: (p[2]-p[0])*(p[3]-p[1]), default=None)
            for x1, y1, x2, y2, conf in people:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
                cv2.putText(frame, f"person {conf:.2f}", (x1, max(18, y1 - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)

            if selected is not None:
                x1, y1, x2, y2, _ = selected
                if frame_no == 1 or frame_no % max(1, args.ocr_every) == 0:
                    number, confidence = read_vest_number(original[y1:y2, x1:x2].copy(), reader)
                    if number is not None:
                        history.append((number, confidence))
                        # Vote across recent successful readings to reduce flicker.
                        counts = Counter(n for n, _c in history)
                        winner, count = counts.most_common(1)[0]
                        matching_conf = [c for n, c in history if n == winner]
                        if count >= 2 or confidence >= 0.85:
                            cached_number = winner
                            cached_conf = sum(matching_conf) / len(matching_conf)
                    # Don't keep an old identity indefinitely if OCR repeatedly fails.
                    elif len(history) == 0:
                        cached_number, cached_conf = None, 0.0
                label = f"Worker #{cached_number} ({cached_conf:.2f})" if cached_number else "Worker #Unknown"
                top = max(0, y1 - 36)
                cv2.rectangle(frame, (x1, top), (min(w - 1, x1 + 330), y1), (20, 20, 20), -1)
                cv2.putText(frame, label, (x1 + 5, max(22, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

            cv2.putText(frame, "VisionShield LIVE CAMERA | Q/Esc: quit", (12, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            try:
                cv2.imshow("VisionShield - Live Camera (PPE + Vest OCR)", frame)
            except cv2.error as exc:
                raise RuntimeError(
                    "OpenCV lost live-window support while running. Reinstall opencv-python==4.11.0.86 "
                    "using the same 'python -m pip' interpreter.\n" + str(exc)
                ) from exc
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    print("VisionShield live camera stopped. Existing checkpoints were not modified.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print("\nERROR:", exc, file=sys.stderr)
        sys.exit(1)
