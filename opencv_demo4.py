import argparse
from collections import Counter, deque
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ============================================================
# VISIONSHIELD - LIVE WEBCAM & VIDEO PPE + WORKER ID DETECTION
# 6-CLASS PER-WORKER PPE DETECTION WITH PPE NUMBER IDENTIFICATION
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
# WORKER IDENTIFICATION & COMPLIANCE RULES:
#
# 1. IDENTIFIED WORKER:
#    - Number written on the worker's PPE kit (vest or helmet) is read via OCR.
#    - Worker is marked with their identified ID: e.g., "Worker #07".
#
# 2. UNKNOWN WORKER & VIOLATION:
#    - If no valid number tag is detected on the PPE kit, worker is tagged "Worker #UNKNOWN".
#    - An explicit "NO ID TAG" VIOLATION is issued.
#
# 3. PPE COMPLIANCE:
#    - Required items: helmet, goggles, gloves, vest, boots.
#    - Missing any item is flagged as a VIOLATION (e.g. "MISSING: gloves, vest").
#
# 4. OVERALL WORKER STATUS:
#    - SAFE:
#        Worker is identified (has valid PPE ID tag) AND has all 5 required PPE items.
#    - UNSAFE:
#        Worker is missing one or more PPE items OR has no recognized ID tag (UNKNOWN).
#
# ============================================================


# ============================================================
# PATHS
# ============================================================

BASE = Path(r"C:\Users\vedan\Desktop\Python\VisionShield")

DEFAULT_MODEL_PATH = (
    BASE
    / "runs"
    / "detect"
    / "visionshield_6class_yolo11n_10ep"
    / "weights"
    / "best.pt"
)


# ============================================================
# CAMERA & DETECTION CONFIGURATION
# ============================================================

CAMERA_INDEX = 0
IMAGE_SIZE = 640
CONFIDENCE = 0.25

# Required PPE items for full compliance
REQUIRED_PPE = [
    "helmet",
    "goggles",
    "gloves",
    "vest",
    "boots",
]

# OCR Configuration
OCR_INTERVAL = 8              # Run OCR every N frames for unverified workers (boosts FPS)
OCR_CONF_THRESHOLD = 0.60     # Minimum OCR confidence to consider a candidate (filters shirt wrinkles/shadows)
OCR_SINGLE_DIGIT_MIN_CONF = 0.75  # Stricter confidence for single digits (e.g. '1', '7' from folds)
OCR_VOTE_THRESHOLD = 2        # Number of consistent readings to lock worker identity
OCR_HIGH_CONF_LOCK = 0.78     # Single reading confidence to lock worker identity immediately

# Person box expansion factors for PPE association
PERSON_EXPAND_X = 0.12
PERSON_EXPAND_TOP = 0.25
PERSON_EXPAND_BOTTOM = 0.12


# ============================================================
# COLORS - BGR
# ============================================================

GREEN = (0, 255, 0)         # Safe / Compliant
RED = (0, 0, 255)           # Unsafe / Violation
BLUE = (255, 0, 0)          # PPE Detection Box
WHITE = (255, 255, 255)     # Text / Neutral
BLACK = (0, 0, 0)           # Text outline / Background
YELLOW = (0, 255, 255)      # Highlights / Unknown warning
CYAN = (255, 255, 0)        # Identified ID badge
DARK_GRAY = (24, 24, 24)    # UI Pill backgrounds
ORANGE = (0, 140, 255)      # Caution / Unknown worker


# ============================================================
# HELPER FUNCTIONS - GEOMETRY & PPE ASSOCIATION
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def expand_person_box(box, image_width, image_height):
    """Expand person box slightly to capture boots, helmet, etc."""
    x1, y1, x2, y2 = box
    width = x2 - x1
    height = y2 - y1

    expand_x = int(width * PERSON_EXPAND_X)
    expand_top = int(height * PERSON_EXPAND_TOP)
    expand_bottom = int(height * PERSON_EXPAND_BOTTOM)

    new_x1 = clamp(x1 - expand_x, 0, image_width - 1)
    new_y1 = clamp(y1 - expand_top, 0, image_height - 1)
    new_x2 = clamp(x2 + expand_x, 0, image_width - 1)
    new_y2 = clamp(y2 + expand_bottom, 0, image_height - 1)

    return (new_x1, new_y1, new_x2, new_y2)


def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) // 2, (y1 + y2) // 2)


def point_inside_box(point, box):
    px, py = point
    x1, y1, x2, y2 = box
    return (x1 <= px <= x2 and y1 <= py <= y2)


def intersection_area(box1, box2):
    x1a, y1a, x2a, y2a = box1
    x1b, y1b, x2b, y2b = box2

    ix1 = max(x1a, x1b)
    iy1 = max(y1a, y1b)
    ix2 = min(x2a, x2b)
    iy2 = min(y2a, y2b)

    if ix2 <= ix1 or iy2 <= iy1:
        return 0

    return (ix2 - ix1) * (iy2 - iy1)


def calculate_iou(box1, box2):
    """Intersection over Union between two bounding boxes."""
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    if area1 == 0 or area2 == 0:
        return 0.0
    inter = intersection_area(box1, box2)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def association_score(ppe_box, person_box):
    """
    Determines whether a PPE detection belongs to a particular worker.
    Same robust metric as opencv_demo3.py.
    """
    center = box_center(ppe_box)
    if not point_inside_box(center, person_box):
        return 0.0

    intersection = intersection_area(ppe_box, person_box)
    ppe_width = max(1, ppe_box[2] - ppe_box[0])
    ppe_height = max(1, ppe_box[3] - ppe_box[1])
    ppe_area = ppe_width * ppe_height
    overlap = intersection / ppe_area
    return 1.0 + overlap


# ============================================================
# DRAWING HELPERS
# ============================================================

def draw_text(image, text, position, color, scale=0.55, thickness=2):
    """Draw text with high-contrast black outline."""
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        BLACK,
        thickness + 3,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def draw_pill(image, text, x, y, bg_color, text_color, scale=0.60, thickness=2, pad=6):
    """Draw text inside a filled rounded/rectangular pill banner for crisp visibility."""
    h, w = image.shape[:2]
    (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x1 = clamp(x, 0, w - 1)
    y1 = clamp(y - th - pad * 2, 0, h - 1)
    x2 = clamp(x1 + tw + pad * 2, 0, w - 1)
    y2 = clamp(y + pad, 0, h - 1)

    cv2.rectangle(image, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(image, (x1, y1), (x2, y2), BLACK, 1)
    cv2.putText(
        image,
        text,
        (x1 + pad, y2 - pad - baseline // 2),
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        text_color,
        thickness,
        cv2.LINE_AA,
    )
    return (x1, y1, x2, y2)


# ============================================================
# OCR & PPE / PAPER NUMBER RECOGNITION
# ============================================================

def preprocess_ppe_crop(region):
    """
    Produce enhanced image variants preserving pencil/pen strokes and reflective vest markings.
    Direct color and CLAHE are optimal for OCR neural networks.
    """
    if region is None or region.size == 0:
        return []

    h, w = region.shape[:2]
    if h < 15 or w < 15:
        return []

    # 1. Direct Color Crop (EasyOCR CRAFT model works very well on raw color)
    variants = [region]

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # Upscale small crops for better character segmentation
    scale = 3 if max(h, w) < 220 else 2
    enlarged = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # 2. Contrast Limited Adaptive Histogram Equalization
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(enlarged)
    variants.append(clahe)

    return variants


def find_paper_or_badge_patches(region):
    """
    Detect bright/white paper sheets, ID badges, or reflective patches within a torso region.
    Prevents background shirt patterns, stripes, and folds from corrupting OCR.
    Returns list of: (patch_image, (rel_x, rel_y, rel_w, rel_h))
    """
    if region is None or region.size == 0:
        return []
    rh, rw = region.shape[:2]
    if rh < 40 or rw < 40:
        return []

    gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)

    # High-brightness threshold for white paper or reflective badge
    _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    # Morphological closing to fuse paper region containing dark digits
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patches = []
    min_area = int((rw * rh) * 0.03)  # At least 3% of torso
    max_area = int((rw * rh) * 0.95)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(max(1, bh))
        if 0.25 <= aspect <= 3.5 and bw >= 30 and bh >= 30:
            # Margin around paper patch
            mx = max(0, bx - 5)
            my = max(0, by - 5)
            mx2 = min(rw, bx + bw + 5)
            my2 = min(rh, by + bh + 5)
            patch = region[my:my2, mx:mx2]
            patches.append((patch, (mx, my, mx2 - mx, my2 - my)))

    # Sort largest patch first
    patches.sort(key=lambda p: p[1][2] * p[1][3], reverse=True)
    return patches


def read_ppe_number(crop, reader):
    """
    Extract worker number from a PPE crop using EasyOCR.
    Rejects vertical wrinkles, shirt stripes, and facial features via aspect ratio and geometry checks.
    Returns: (best_number_str, confidence) or (None, 0.0)
    """
    if crop is None or crop.size == 0 or reader is None:
        return None, 0.0

    best_candidate = None
    best_conf = 0.0

    images = preprocess_ppe_crop(crop)
    for img in images:
        try:
            results = reader.readtext(
                img,
                allowlist="0123456789",
                detail=1,
                paragraph=False,
                decoder="greedy",
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.30,
            )
        except Exception:
            continue

        for bbox, raw_text, conf in results:
            digits = re.sub(r"\D", "", str(raw_text))
            conf = float(conf)

            # Enforce minimum confidence to avoid shirt wrinkle hallucinations
            if not (1 <= len(digits) <= 5 and conf >= OCR_CONF_THRESHOLD):
                continue

            # Geometry check: reject razor-thin vertical wrinkles / shirt folds
            if bbox is not None and len(bbox) == 4:
                pts = np.array(bbox, dtype=np.int32)
                _bx, _by, bw, bh = cv2.boundingRect(pts)
                aspect = bw / float(max(1, bh))
                # Single digits like 1, 3, 5 must have sane aspect ratio and higher confidence
                if len(digits) == 1:
                    if conf < OCR_SINGLE_DIGIT_MIN_CONF or aspect < 0.18 or aspect > 2.8:
                        continue
                if bw < 8 or bh < 12:
                    continue

            if conf > best_conf:
                best_candidate = digits
                best_conf = conf

        # High confidence early-exit
        if best_conf >= OCR_HIGH_CONF_LOCK:
            break

    return best_candidate, best_conf


def get_worker_ppe_crops(frame, person_box, worker_ppe):
    """
    Extract regions of interest from the PPE kit or torso paper tag where worker numbers are placed:
    1. White paper / ID badge patch on chest (supports paper numbers held to chest!)
    2. Detected Vest crop & Vest chest patch
    3. Torso region (strictly starting below the chin/neck to exclude face, beard, and glasses)
    4. Helmet box (if detected)
    """
    frame_h, frame_w = frame.shape[:2]
    px1, py1, px2, py2 = person_box
    pw = px2 - px1
    ph = py2 - py1

    badge_crops = []
    fallback_crops = []

    # 1. Detected Vest crop
    vest_item = worker_ppe.get("vest")
    if vest_item is not None:
        vx1, vy1, vx2, vy2 = vest_item["detection"]["box"]
        vx1 = clamp(vx1, 0, frame_w - 1)
        vy1 = clamp(vy1, 0, frame_h - 1)
        vx2 = clamp(vx2, 0, frame_w)
        vy2 = clamp(vy2, 0, frame_h)
        if vx2 > vx1 and vy2 > vy1:
            vest_crop = frame[vy1:vy2, vx1:vx2]

            # Look for white paper patch or reflective number badge on the vest
            vest_patches = find_paper_or_badge_patches(vest_crop)
            for p_img, (rel_x, rel_y, rel_w, rel_h) in vest_patches:
                badge_crops.append((
                    "vest_badge",
                    p_img,
                    (vx1 + rel_x, vy1 + rel_y, vx1 + rel_x + rel_w, vy1 + rel_y + rel_h),
                ))

            # Chest / center patch of vest
            vh = vy2 - vy1
            vw = vx2 - vx1
            chest_crop = vest_crop[int(vh * 0.12):int(vh * 0.75), int(vw * 0.10):int(vw * 0.90)]
            if chest_crop.size > 0:
                fallback_crops.append(("vest_chest", chest_crop, (vx1, vy1, vx2, vy2)))

    # 2. Chest / Torso Region (STRICTLY starting below neck/chin to exclude eyes, glasses, mustache, beard!)
    torso_y1 = clamp(py1 + int(ph * 0.32), 0, frame_h - 1)
    torso_y2 = clamp(py1 + int(ph * 0.95), 0, frame_h)
    torso_x1 = clamp(px1 + int(pw * 0.05), 0, frame_w - 1)
    torso_x2 = clamp(px2 - int(pw * 0.05), 0, frame_w)

    if torso_y2 > torso_y1 and torso_x2 > torso_x1:
        torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]

        # DETECT PAPER SHEET / BADGE ON CHEST:
        paper_patches = find_paper_or_badge_patches(torso_crop)
        for p_img, (rel_x, rel_y, rel_w, rel_h) in paper_patches:
            badge_crops.append((
                "paper_badge",
                p_img,
                (torso_x1 + rel_x, torso_y1 + rel_y, torso_x1 + rel_x + rel_w, torso_y1 + rel_y + rel_h),
            ))

    # If badge patches are detected, return only them (fastest and cleanest)
    if badge_crops:
        return badge_crops

    # 3. Helmet Crop (some sites put worker numbers on helmets)
    helmet_item = worker_ppe.get("helmet")
    if helmet_item is not None:
        hx1, hy1, hx2, hy2 = helmet_item["detection"]["box"]
        hx1 = clamp(hx1, 0, frame_w - 1)
        hy1 = clamp(hy1, 0, frame_h - 1)
        hx2 = clamp(hx2, 0, frame_w)
        hy2 = clamp(hy2, 0, frame_h)
        if hx2 > hx1 and hy2 > hy1:
            helmet_crop = frame[hy1:hy2, hx1:hx2]
            fallback_crops.append(("helmet", helmet_crop, (hx1, hy1, hx2, hy2)))

    return fallback_crops[:2]


# ============================================================
# MULTI-WORKER TRACKER & IDENTITY MANAGER
# ============================================================

class TrackedWorker:
    """Maintains consistent identity, OCR voting history, and safety state across frames."""

    def __init__(self, track_id, box):
        self.track_id = track_id
        self.box = box
        self.lost_frames = 0
        self.ocr_attempts = 0
        self.last_ocr_frame = -999
        self.history = deque(maxlen=9)
        self.assigned_id = None
        self.assigned_conf = 0.0
        self.is_identified = False
        self.id_locked = False
        self.last_detected_tag_box = None
        self.tag_visible_this_frame = False

    def update_box(self, box):
        self.box = box
        self.lost_frames = 0
        self.tag_visible_this_frame = False

    def record_ocr_reading(self, number, conf, single_frame_mode=False):
        self.ocr_attempts += 1
        if number is not None and conf >= OCR_CONF_THRESHOLD:
            self.history.append((number, conf))
            self.tag_visible_this_frame = True

            # Majority voting across recent readings to filter single-frame noise
            counts = Counter(num for num, _ in self.history)
            winner, count = counts.most_common(1)[0]
            matching_confs = [c for num, c in self.history if num == winner]
            avg_conf = sum(matching_confs) / len(matching_confs)

            if single_frame_mode or count >= OCR_VOTE_THRESHOLD or conf >= OCR_HIGH_CONF_LOCK:
                if not self.id_locked or count >= OCR_VOTE_THRESHOLD:
                    self.assigned_id = winner
                    self.assigned_conf = avg_conf
                    self.is_identified = True
                    self.id_locked = True
        else:
            self.tag_visible_this_frame = False
            # When the badge is temporarily hidden / worker turns around,
            # KEEP the assigned ID stored for this person!


class WorkerTracker:
    """Matches detections to persistent worker tracks using IoU & center distance."""

    def __init__(self):
        self.tracks = []
        self.next_track_id = 1

    def update(self, detected_boxes):
        matched_tracks = {}
        unmatched_boxes = []

        # Greedy IoU matching
        used_tracks = set()
        for box_idx, dbox in enumerate(detected_boxes):
            best_iou = 0.0
            best_track = None
            for track in self.tracks:
                if track.track_id in used_tracks:
                    continue
                iou = calculate_iou(dbox, track.box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = track

            if best_track is not None and best_iou >= 0.25:
                best_track.update_box(dbox)
                matched_tracks[box_idx] = best_track
                used_tracks.add(best_track.track_id)
            else:
                unmatched_boxes.append((box_idx, dbox))

        # Centroid distance fallback for remaining boxes
        for box_idx, dbox in unmatched_boxes:
            dc = box_center(dbox)
            dw = dbox[2] - dbox[0]
            best_dist = 1e9
            best_track = None
            for track in self.tracks:
                if track.track_id in used_tracks:
                    continue
                tc = box_center(track.box)
                dist = np.hypot(dc[0] - tc[0], dc[1] - tc[1])
                if dist < best_dist:
                    best_dist = dist
                    best_track = track

            max_dist = max(130, int(dw * 0.9))
            if best_track is not None and best_dist < max_dist:
                best_track.update_box(dbox)
                matched_tracks[box_idx] = best_track
                used_tracks.add(best_track.track_id)
            else:
                # Create a new worker track
                new_track = TrackedWorker(self.next_track_id, dbox)
                self.next_track_id += 1
                self.tracks.append(new_track)
                matched_tracks[box_idx] = new_track
                used_tracks.add(new_track.track_id)

        # Increment lost frames for tracks not seen in this frame
        for track in self.tracks:
            if track.track_id not in used_tracks:
                track.lost_frames += 1

        # Purge stale tracks lost for more than 60 frames (~2 seconds)
        self.tracks = [t for t in self.tracks if t.lost_frames <= 60]

        return matched_tracks


# ============================================================
# CAMERA INITIALIZATION WITH DIRECTSHOW FALLBACK
# ============================================================

def open_capture_source(source, width=1280, height=720):
    """
    Open camera, video, or image source with DirectShow support for Windows reliability.
    Returns: (cap, is_image, static_image_frame)
    """
    is_cam_idx = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
    if is_cam_idx:
        idx = int(source)
        # DirectShow is fastest & most reliable on Windows webcams
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open webcam index {idx}.\n"
                f"Try --source 1 if camera 0 is busy, or check Windows camera permissions."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap, False, None

    # Check if file path
    src_path = Path(source)
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    # Check if static image file
    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    if src_path.suffix.lower() in image_extensions:
        img = cv2.imread(str(src_path))
        if img is None:
            raise RuntimeError(f"Could not decode image file: {src_path}")
        return None, True, img

    # Video file
    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {src_path}")
    return cap, False, None


# ============================================================
# MAIN APPLICATION
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="VisionShield: Live Webcam PPE Detection + PPE Number OCR"
    )
    parser.add_argument(
        "--source",
        default=str(CAMERA_INDEX),
        help="Camera index (e.g. 0) or path to video/image file (default: 0)",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_MODEL_PATH),
        help="Path to YOLO best.pt 6-class weights",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE,
        help="Confidence threshold for YOLO detections",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=IMAGE_SIZE,
        help="YOLO inference image size (default: 640)",
    )
    parser.add_argument(
        "--ocr-interval",
        type=int,
        default=OCR_INTERVAL,
        help="Run OCR every N frames for active workers (default: 6)",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional path to save annotated output (image or video file)",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before exiting",
    )
    parser.add_argument(
        "--no-window",
        action="store_true",
        help="Run without displaying GUI window (useful for headless/export)",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()

    print("=" * 72)
    print("VISIONSHIELD - LIVE PPE DETECTION WITH PPE NUMBER IDENTIFICATION")
    print("=" * 72)

    model_path = Path(args.model)
    print(f"\n[1/3] Loading YOLO Model:\n      {model_path}")

    if not model_path.exists():
        raise FileNotFoundError(
            f"\nModel checkpoint not found:\n{model_path}\n"
            "Please check the path or train weights."
        )

    model = YOLO(str(model_path))
    print("      Model loaded successfully.")
    print("      Detected Classes:")
    for cid, name in model.names.items():
        print(f"        {cid}: {name}")

    print("\n[2/3] Initializing EasyOCR Reader (GPU accelerated if available)...")
    use_gpu = torch.cuda.is_available()
    print(f"      CUDA Available: {use_gpu}")
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=use_gpu)
        print("      EasyOCR initialized successfully.")
    except Exception as exc:
        print(f"      WARNING: EasyOCR init failed ({exc}). Falling back to CPU.")
        reader = easyocr.Reader(["en"], gpu=False)

    print(f"\n[3/3] Opening source: {args.source}...")
    source_val = int(args.source) if args.source.isdigit() else args.source
    cap, is_image, static_image = open_capture_source(source_val)

    print("\n" + "=" * 72)
    print("SYSTEM READY. Controls:")
    print("  [Q] or [ESC] : Quit")
    print("  [S]          : Save current frame screenshot")
    print("  [R]          : Reset worker identities & OCR cache")
    print("  [P]          : Pause / Resume playback")
    print("=" * 72 + "\n")

    tracker = WorkerTracker()

    fps = 0.0
    previous_time = cv2.getTickCount()
    frame_number = 0
    paused = False

    video_writer = None
    if args.output and not is_image:
        output_p = Path(args.output)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        # Video resolution will be initialized on first frame
        print(f"      Saving output video to: {output_p}")

    while True:
        if not paused:
            if is_image:
                frame = static_image.copy()
                success = True
            else:
                success, frame = cap.read()
                if not success or frame is None:
                    # If reading video, loop back to start; if webcam, exit
                    if not (isinstance(source_val, int) or str(source_val).isdigit()):
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        continue
                    else:
                        print("\nCould not read frame from camera.")
                        break

            frame_number += 1
            frame_height, frame_width = frame.shape[:2]
            original_clean_frame = frame.copy()

            # ----------------------------------------------------
            # YOLO DETECTION
            # ----------------------------------------------------
            results = model.predict(
                source=frame,
                conf=args.conf,
                imgsz=args.imgsz,
                verbose=False,
            )
            result = results[0]

            detections = []
            if result.boxes is not None:
                for box in result.boxes:
                    class_id = int(box.cls[0].item())
                    confidence = float(box.conf[0].item())
                    class_name = model.names.get(class_id, str(class_id))
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    detections.append(
                        {
                            "class": class_name,
                            "confidence": confidence,
                            "box": (x1, y1, x2, y2),
                        }
                    )

            # Separate Person boxes from PPE items
            persons = [d for d in detections if d["class"] == "Person"]
            ppe_detections = [d for d in detections if d["class"] in REQUIRED_PPE]

            # ----------------------------------------------------
            # TRACK PERSONS ACROSS FRAMES
            # ----------------------------------------------------
            person_boxes = [p["box"] for p in persons]
            matched_tracks = tracker.update(person_boxes)

            # ----------------------------------------------------
            # WORKER & PPE ASSOCIATION + OCR IDENTIFICATION
            # ----------------------------------------------------
            workers_data = []

            for p_idx, person in enumerate(persons):
                person_box = person["box"]
                tracked_worker = matched_tracks.get(p_idx)

                expanded_box = expand_person_box(
                    person_box, frame_width, frame_height
                )

                # Initialize worker PPE dict
                worker_ppe = {ppe: None for ppe in REQUIRED_PPE}

                # Associate PPE items to this person
                for ppe_det in ppe_detections:
                    ppe_name = ppe_det["class"]
                    ppe_box = ppe_det["box"]
                    score = association_score(ppe_box, expanded_box)
                    if score <= 0:
                        continue

                    current = worker_ppe[ppe_name]
                    if current is None or score > current["score"]:
                        worker_ppe[ppe_name] = {
                            "detection": ppe_det,
                            "score": score,
                        }

                # ------------------------------------------------
                # OCR IDENTIFICATION ON PPE KIT
                # ------------------------------------------------
                number_found_this_frame = None
                number_conf_this_frame = 0.0

                if tracked_worker is not None:
                    # FPS Optimization: Skip heavy OCR if worker is already confirmed and locked!
                    needs_ocr = not tracked_worker.id_locked
                    due_interval = (frame_number - tracked_worker.last_ocr_frame >= args.ocr_interval)
                    box_h = person_box[3] - person_box[1]
                    can_ocr = box_h >= 100

                    if needs_ocr and due_interval and can_ocr:
                        tracked_worker.last_ocr_frame = frame_number

                        # Get candidate regions (Paper patch on chest, vest, torso, helmet)
                        ppe_crops = get_worker_ppe_crops(
                            original_clean_frame, person_box, worker_ppe
                        )

                        for _crop_type, crop_img, crop_box in ppe_crops:
                            num, conf = read_ppe_number(crop_img, reader)
                            if num is not None:
                                number_found_this_frame = num
                                number_conf_this_frame = conf
                                tracked_worker.last_detected_tag_box = crop_box
                                tracked_worker.tag_visible_this_frame = True
                                break

                        tracked_worker.record_ocr_reading(
                            number_found_this_frame,
                            number_conf_this_frame,
                            single_frame_mode=is_image,
                        )

                    # Determine worker identity tag
                    if tracked_worker.is_identified:
                        worker_id_tag = f"Worker #{tracked_worker.assigned_id}"
                        is_identified = True
                    else:
                        worker_id_tag = "Worker #UNKNOWN"
                        is_identified = False

                    display_track_id = tracked_worker.track_id
                    worker_conf = tracked_worker.assigned_conf
                    tag_box = tracked_worker.last_detected_tag_box if tracked_worker.tag_visible_this_frame else None
                    tag_visible = tracked_worker.tag_visible_this_frame
                else:
                    worker_id_tag = "Worker #UNKNOWN"
                    is_identified = False
                    display_track_id = p_idx + 1
                    worker_conf = 0.0
                    tag_box = None
                    tag_visible = False

                # ------------------------------------------------
                # PPE COMPLIANCE & VIOLATION ANALYSIS
                # ------------------------------------------------
                missing_ppe = [
                    ppe for ppe in REQUIRED_PPE if worker_ppe[ppe] is None
                ]

                violations = []

                # Rule 1: Unknown worker (missing PPE kit number tag) is a VIOLATION!
                if not is_identified:
                    violations.append("NO ID TAG")

                # Rule 2: Missing required PPE is a VIOLATION!
                for ppe in missing_ppe:
                    violations.append(f"NO {ppe.upper()}")

                # Safety Status: SAFE only if identified AND no missing PPE
                if len(violations) == 0:
                    status = "SAFE"
                else:
                    status = "UNSAFE"

                workers_data.append(
                    {
                        "track_id": display_track_id,
                        "person": person,
                        "worker_id_tag": worker_id_tag,
                        "is_identified": is_identified,
                        "worker_conf": worker_conf,
                        "ppe": worker_ppe,
                        "missing": missing_ppe,
                        "violations": violations,
                        "status": status,
                        "tag_box": tag_box,
                        "tag_visible": tag_visible,
                    }
                )

            # ----------------------------------------------------
            # COUNTERS
            # ----------------------------------------------------
            total_workers = len(workers_data)
            safe_count = sum(1 for w in workers_data if w["status"] == "SAFE")
            unsafe_count = sum(1 for w in workers_data if w["status"] == "UNSAFE")
            identified_count = sum(1 for w in workers_data if w["is_identified"])
            unknown_count = total_workers - identified_count

            if total_workers == 0:
                overall_status = "NO WORKER"
                overall_color = WHITE
            elif unsafe_count > 0:
                overall_status = "VIOLATION"
                overall_color = RED
            else:
                overall_status = "SAFE"
                overall_color = GREEN

            # ----------------------------------------------------
            # DRAW WORKERS & ANNOTATIONS
            # ----------------------------------------------------
            for w in workers_data:
                person = w["person"]
                x1, y1, x2, y2 = person["box"]
                status = w["status"]
                is_identified = w["is_identified"]
                worker_id_tag = w["worker_id_tag"]
                violations = w["violations"]

                # Worker Box Color
                worker_box_color = GREEN if status == "SAFE" else RED

                # Draw Person Bounding Box
                cv2.rectangle(frame, (x1, y1), (x2, y2), worker_box_color, 3)

                # Worker Header Pill Label
                if is_identified:
                    label_text = f"{worker_id_tag} ({w['worker_conf']:.2f}) | {status}"
                    pill_bg = (0, 110, 0) if status == "SAFE" else (0, 0, 140)
                else:
                    label_text = f"{worker_id_tag} | {status}"
                    pill_bg = (0, 0, 140)

                # Avoid overlap with top HUD dashboard (0 to 105px)
                if y1 < 125:
                    label_y = min(frame_height - 35, max(135, y1 + 32))
                else:
                    label_y = y1 - 8

                draw_pill(
                    frame,
                    label_text,
                    x1,
                    label_y,
                    pill_bg,
                    WHITE,
                    scale=0.58,
                    thickness=2,
                )

                # ------------------------------------------------
                # DRAW PPE BOXES
                # ------------------------------------------------
                for ppe in REQUIRED_PPE:
                    item = w["ppe"][ppe]
                    if item is None:
                        continue

                    detection = item["detection"]
                    px1, py1, px2, py2 = detection["box"]
                    confidence = detection["confidence"]

                    # PPE bounding box
                    cv2.rectangle(frame, (px1, py1), (px2, py2), BLUE, 2)

                    # PPE label
                    draw_text(
                        frame,
                        f"{ppe} {confidence:.2f}",
                        (px1, max(22, py1 - 6)),
                        BLUE,
                        0.48,
                        2,
                    )

                # Highlight detected paper patch or vest badge if identified and currently visible
                if is_identified:
                    tag_box = w.get("tag_box")
                    tag_visible = w.get("tag_visible", False)
                    badge_id = w["worker_id_tag"].split("#")[-1]
                    vest_item = w["ppe"].get("vest")
                    if tag_visible and tag_box is not None:
                        tx1, ty1, tx2, ty2 = tag_box
                        cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), CYAN, 2)
                        draw_pill(
                            frame,
                            f"ID: {badge_id}",
                            tx1,
                            max(120, ty1 - 6),
                            (0, 110, 110),
                            WHITE,
                            scale=0.52,
                            thickness=2,
                        )
                    elif tag_visible and vest_item is not None:
                        vx1, vy1, vx2, vy2 = vest_item["detection"]["box"]
                        badge_text = f"ID: {badge_id}"
                        draw_text(
                            frame,
                            badge_text,
                            (vx1 + 5, vy1 + 22),
                            CYAN,
                            0.60,
                            2,
                        )

                # ------------------------------------------------
                # DRAW VIOLATIONS (MISSING PPE + UNKNOWN TAG)
                # ------------------------------------------------
                if violations:
                    violation_text = "VIOLATION: " + ", ".join(violations)
                    # Check bottom clearance (reserve 35px for bottom helper bar)
                    if y2 + 30 < frame_height - 35:
                        vy = y2 + 25
                    else:
                        vy = max(140, y2 - 18)

                    draw_pill(
                        frame,
                        violation_text,
                        x1,
                        vy,
                        (0, 0, 150),
                        WHITE,
                        scale=0.48,
                        thickness=2,
                    )

            # ----------------------------------------------------
            # TOP HUD DASHBOARD
            # ----------------------------------------------------
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame_width, 105), DARK_GRAY, -1)
            frame = cv2.addWeighted(overlay, 0.78, frame, 0.22, 0)

            # Responsive HUD column positioning
            if frame_width < 900:
                col1_x = 15
                col2_x = int(frame_width * 0.33)
                col3_x = int(frame_width * 0.65)
                hud_scale = 0.52
                stat_scale = 0.68
            else:
                col1_x = 20
                col2_x = int(frame_width * 0.36)
                col3_x = max(int(frame_width * 0.72), frame_width - 320)
                hud_scale = 0.62
                stat_scale = 0.85

            # Left stats
            draw_text(frame, f"Workers: {total_workers}", (col1_x, 30), WHITE, hud_scale, 2)
            draw_text(frame, f"Safe: {safe_count}", (col1_x, 60), GREEN, hud_scale, 2)
            draw_text(frame, f"Violations: {unsafe_count}", (col1_x, 90), RED, hud_scale, 2)

            # Center stats (ID tag tracking)
            draw_text(
                frame,
                f"Identified IDs: {identified_count}",
                (col2_x, 42),
                CYAN,
                hud_scale,
                2,
            )
            draw_text(
                frame,
                f"Unknown Workers: {unknown_count}",
                (col2_x, 78),
                YELLOW if unknown_count > 0 else WHITE,
                hud_scale,
                2,
            )

            # Right Banner: Overall Status
            draw_text(
                frame,
                f"STATUS: {overall_status}",
                (col3_x, 58),
                overall_color,
                stat_scale,
                3 if stat_scale > 0.7 else 2,
            )

            # ----------------------------------------------------
            # FPS CALCULATION
            # ----------------------------------------------------
            current_time = cv2.getTickCount()
            elapsed = (current_time - previous_time) / cv2.getTickFrequency()
            previous_time = current_time

            if elapsed > 0:
                current_fps = 1.0 / elapsed
                fps = 0.90 * fps + 0.10 * current_fps

            draw_text(
                frame,
                f"FPS: {fps:.1f}",
                (frame_width - 130, frame_height - 20),
                WHITE,
                0.55,
                2,
            )

            # Bottom helper prompt
            draw_text(
                frame,
                "Q: Quit | S: Screenshot | R: Reset Tracker | P: Pause",
                (20, frame_height - 20),
                WHITE,
                0.48,
                1,
            )

            # ----------------------------------------------------
            # WRITE OUTPUT IF REQUESTED
            # ----------------------------------------------------
            if args.output:
                if is_image:
                    cv2.imwrite(str(args.output), frame)
                    print(f"\n[Annotated Image Saved] -> {args.output}")
                else:
                    if video_writer is None:
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        video_writer = cv2.VideoWriter(
                            str(args.output), fourcc, 20.0, (frame_width, frame_height)
                        )
                    video_writer.write(frame)

            # Check max frames
            if args.max_frames and frame_number >= args.max_frames:
                print(f"\nReached max-frames limit ({args.max_frames}). Stopping...")
                break

        # ----------------------------------------------------
        # SHOW FRAME (IF NOT HEADLESS)
        # ----------------------------------------------------
        if not args.no_window:
            cv2.imshow("VisionShield - Live PPE & Worker ID Detection", frame)
            key = cv2.waitKey(1) & 0xFF

            # Q or ESC = Quit
            if key in (ord("q"), ord("Q"), 27):
                print("\nQuit key received. Stopping...")
                break

            # S = Screenshot
            elif key in (ord("s"), ord("S")):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_path = BASE / f"worker_id_capture_{timestamp}.jpg"
                cv2.imwrite(str(output_path), frame)
                print(f"\n[Screenshot Saved] -> {output_path}")

            # R = Reset Tracker
            elif key in (ord("r"), ord("R")):
                tracker = WorkerTracker()
                print("\n[Tracker Reset] Worker IDs and OCR histories cleared.")

            # P = Pause / Resume
            elif key in (ord("p"), ord("P")):
                paused = not paused
                print(f"\n[{'Paused' if paused else 'Resumed'}]")
        else:
            # If headless and static image, process once and exit
            if is_image:
                break

    # --------------------------------------------------------
    # CLEANUP
    # --------------------------------------------------------
    if video_writer is not None:
        video_writer.release()
        print(f"[Output Video Saved] -> {args.output}")
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    print("\nSource closed.")
    print("VisionShield worker identification terminated successfully.")


if __name__ == "__main__":
    main()
