import argparse
from collections import Counter, deque
import math
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ============================================================
# VISIONSHIELD - VERSION 5
# INTEGRATED MULTI-WORKER PPE DETECTION + WORKER ID RECOGNITION
# + PRODUCTION MACHINE PROXIMITY & DANGER ZONE MONITORING
# ============================================================
#
# Modules:
# 1. PPE & WORKER DETECTION (6-Class Model):
#    - Classes: helmet, gloves, vest, boots, goggles, Person
# 2. WORKER ID IDENTIFICATION (EasyOCR + Persistent Memory):
#    - Reads number tags from worker vest/badge/paper with fast CLAHE
#    - Persistent Memory ("Lock & Store"): Remembers worker ID even when badge
#      is occluded, turned away, or temporarily obscured.
#    - Skips heavy OCR on identified workers to maintain high real-time FPS!
# 3. MACHINE DETECTION & PROXIMITY HAZARD ENGINE:
#    - Detects: forklift, production_machine, conveyor
#    - Computes dynamic exclusion hazard zones (polygons & radial halos)
#    - Calculates real-time Euclidean gap distance between workers and machinery
#    - Triggers "PROXIMITY DANGER" violations and draws alert connectors
# 4. COMPREHENSIVE INDUSTRIAL SAFETY STATUS:
#    - SAFE: Identified + All 5 PPE items + Safe distance from all machinery
#    - UNSAFE: Missing PPE, Unknown ID tag, or Proximity Danger
# ============================================================


# ============================================================
# PATHS & CONFIGURATION
# ============================================================

BASE = Path(r"C:\Users\vedan\Desktop\Python\VisionShield")

DEFAULT_PPE_MODEL = (
    BASE
    / "runs"
    / "detect"
    / "visionshield_6class_yolo11n_10ep"
    / "weights"
    / "best.pt"
)

# Machine detector weights
DEFAULT_MACHINE_MODEL = (
    BASE
    / "visionshield_machine_detector_starter"
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "visionshield_machine_2class_aug40"
    / "weights"
    / "best.pt"
)

FALLBACK_MACHINE_MODEL = (
    BASE
    / "visionshield_machine_detector_starter"
    / "runs"
    / "detect"
    / "runs"
    / "detect"
    / "visionshield_machine_yolo11n_15epochs"
    / "weights"
    / "best.pt"
)

DEFAULT_VIDEO = BASE / "CINEMATIC CORPORATE VIDEO  Factory B ROLL.mp4"

CAMERA_INDEX = 0
IMAGE_SIZE = 640
CONFIDENCE = 0.25
MACHINE_CONFIDENCE = 0.30

# Proximity threshold: normalized image gap (gap in pixels / frame diagonal)
DEFAULT_PROXIMITY_THRESHOLD = 0.08
DEFAULT_CAUTION_THRESHOLD = 0.15

# Required PPE items for full compliance
REQUIRED_PPE = [
    "helmet",
    "goggles",
    "gloves",
    "vest",
    "boots",
]

# OCR Configuration
OCR_INTERVAL = 4                  # Run OCR every N frames for unverified workers
OCR_CONF_THRESHOLD = 0.35         # Minimum OCR candidate confidence
OCR_SINGLE_DIGIT_MIN_CONF = 0.45  # Confirmed confidence for single digits (e.g. '4')
OCR_VOTE_THRESHOLD = 2            # Number of consistent readings to lock worker identity
OCR_HIGH_CONF_LOCK = 0.72         # Single reading confidence to lock worker identity immediately

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
YELLOW = (0, 255, 255)      # Highlights / Caution / Unknown warning
CYAN = (255, 255, 0)        # Identified ID badge
DARK_GRAY = (24, 24, 24)    # UI Pill backgrounds
ORANGE = (0, 140, 255)      # Caution / Machine Danger Zone
AMBER = (0, 185, 255)       # Machine Bounding Box


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
# MACHINE DANGER ZONES & PROXIMITY CALCULATIONS
# ============================================================

def compute_machine_hazard_zone(box, label, image_width, image_height, width_ratio=0.22, height_ratio=0.50):
    """
    Computes an exclusion hazard zone polygon for machinery:
    - For 'forklift': Mobile heavy vehicle with 360-degree radial perimeter around base.
    - For 'production_machine': Operational exclusion zone on the hazard side / front.
    """
    x1, y1, x2, y2 = box
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)

    clean_label = label.lower().replace(" ", "_")
    if "forklift" in clean_label:
        # Radial halo buffer around forklift base
        expand_w = int(bw * 0.25)
        expand_h = int(bh * 0.20)
        bx1 = clamp(x1 - expand_w, 0, image_width - 1)
        by1 = clamp(y1 + int(bh * 0.40), 0, image_height - 1)
        bx2 = clamp(x2 + expand_w, 0, image_width - 1)
        by2 = clamp(y2 + expand_h, 0, image_height - 1)
        return np.array([[bx1, by1], [bx2, by1], [bx2, by2], [bx1, by2]], dtype=np.int32)
    else:
        # Production machine: right/front operational hazard zone
        zw = max(25, int(bw * width_ratio))
        zh = max(25, int(bh * height_ratio))
        top = max(0, min(image_height - 1, y1 + int(bh * 0.25)))
        bottom = max(0, min(image_height - 1, top + zh))
        left = max(0, min(image_width - 1, x2))
        right = max(0, min(image_width - 1, x2 + zw))
        if right <= left or bottom <= top:
            return None
        return np.array([
            [left, top],
            [right, top + int((bottom - top) * 0.12)],
            [right, bottom],
            [left, bottom],
        ], dtype=np.int32)


def calculate_worker_machine_proximity(person_box, machine_box, diagonal):
    """
    Computes minimum gap distance and normalized distance between person and machine bounding boxes.
    """
    px1, py1, px2, py2 = person_box
    mx1, my1, mx2, my2 = machine_box

    dx = max(mx1 - px2, px1 - mx2, 0)
    dy = max(my1 - py2, py1 - my2, 0)
    gap_pixels = math.hypot(dx, dy)
    normalized_gap = gap_pixels / max(1.0, diagonal)

    return gap_pixels, normalized_gap


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
    Direct color and CLAHE are optimal for OCR neural networks while preserving high FPS.
    """
    if region is None or region.size == 0:
        return []

    h, w = region.shape[:2]
    if h < 15 or w < 15:
        return []

    # 1. Upscaled Color Crop (ensures CRAFT receptive field catches handwritten single digits)
    scale = 3 if max(h, w) < 220 else 2
    enlarged_bgr = cv2.resize(region, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    variants = [enlarged_bgr]

    gray = cv2.cvtColor(enlarged_bgr, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8)).apply(gray)
    variants.append(clahe)

    return variants


def find_paper_or_badge_patches(region):
    """
    Detect bright/white paper sheets, ID badges, or reflective patches within a torso region.
    Uses HSV color space to filter out walls, skin, and clothing.
    """
    if region is None or region.size == 0:
        return []
    rh, rw = region.shape[:2]
    if rh < 40 or rw < 40:
        return []

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    # White paper has low color saturation (< 60) and high brightness (> 130)
    paper_mask = (sat < 60) & (val > 130)
    paper_mask_uint8 = np.uint8(paper_mask * 255)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    closed = cv2.morphologyEx(paper_mask_uint8, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    patches = []
    min_area = int((rw * rh) * 0.04)  # At least 4% of torso
    max_area = int((rw * rh) * 0.90)

    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area or area > max_area:
            continue
        hull = cv2.convexHull(c)
        hull_area = cv2.contourArea(hull)
        if hull_area <= 0:
            continue
        solidity = area / float(hull_area)
        # Rectangular / polygonal paper sheet has high solidity (> 0.60)
        if solidity < 0.60:
            continue

        bx, by, bw, bh = cv2.boundingRect(c)
        aspect = bw / float(max(1, bh))
        if 0.30 <= aspect <= 3.0 and bw >= 40 and bh >= 40:
            mx = max(0, bx - 6)
            my = max(0, by - 6)
            mx2 = min(rw, bx + bw + 6)
            my2 = min(rh, by + bh + 6)
            patch = region[my:my2, mx:mx2]
            patches.append((patch, (mx, my, mx2 - mx, my2 - my)))

    patches.sort(key=lambda p: p[1][2] * p[1][3], reverse=True)
    return patches


def read_ppe_number(crop, reader):
    """
    Extract worker number from a PPE crop using EasyOCR.
    Rejects false zeroes from circular buttons/folds and filters shirt noise.
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
                text_threshold=0.25,
                low_text=0.15,
                link_threshold=0.25,
                mag_ratio=1.5,
            )
        except Exception:
            continue

        for bbox, raw_text, conf in results:
            digits = re.sub(r"\D", "", str(raw_text))
            conf = float(conf)

            # Rule 1: Length must be 1 to 4 digits
            if not (1 <= len(digits) <= 4 and conf >= OCR_CONF_THRESHOLD):
                continue

            # Rule 2: REJECT ALL PURE ZERO STRINGS ("0", "00", "000")
            # A lone zero is a circular button, collar curve, or fold.
            if digits.strip("0") == "":
                continue

            # Rule 3: Geometry check on character bounding box
            if bbox is not None and len(bbox) == 4:
                pts = np.array(bbox, dtype=np.int32)
                _bx, _by, bw, bh = cv2.boundingRect(pts)
                aspect = bw / float(max(1, bh))
                if len(digits) == 1:
                    if conf < OCR_SINGLE_DIGIT_MIN_CONF or aspect < 0.18 or aspect > 2.5:
                        continue
                    if bh < 16 or bw < 8:
                        continue
                if bw < 8 or bh < 12:
                    continue

            if conf > best_conf:
                best_candidate = digits
                best_conf = conf

        if best_conf >= OCR_HIGH_CONF_LOCK:
            break

    return best_candidate, best_conf


def get_worker_ppe_crops(frame, person_box, worker_ppe):
    """
    Extract regions of interest from the PPE kit or torso paper tag where worker numbers are placed:
    1. Detected white paper sheets held to chest/torso or vest badge patches
    2. Fallback chest region ONLY if vest is worn
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

            vest_patches = find_paper_or_badge_patches(vest_crop)
            for p_img, (rel_x, rel_y, rel_w, rel_h) in vest_patches:
                badge_crops.append((
                    "vest_badge",
                    p_img,
                    (vx1 + rel_x, vy1 + rel_y, vx1 + rel_x + rel_w, vy1 + rel_y + rel_h),
                ))

            vh = vy2 - vy1
            vw = vx2 - vx1
            chest_crop = vest_crop[int(vh * 0.15):int(vh * 0.75), int(vw * 0.10):int(vw * 0.90)]
            if chest_crop.size > 0:
                fallback_crops.append(("vest_chest", chest_crop, (vx1, vy1, vx2, vy2)))

    # 2. Chest / Torso Region (from below chin/neck down to lower torso)
    torso_y1 = clamp(py1 + int(ph * 0.32), 0, frame_h - 1)
    torso_y2 = clamp(py1 + int(ph * 0.96), 0, frame_h)
    torso_x1 = clamp(px1 + int(pw * 0.05), 0, frame_w - 1)
    torso_x2 = clamp(px2 - int(pw * 0.05), 0, frame_w)

    if torso_y2 > torso_y1 and torso_x2 > torso_x1:
        torso_crop = frame[torso_y1:torso_y2, torso_x1:torso_x2]

        paper_patches = find_paper_or_badge_patches(torso_crop)
        for p_img, (rel_x, rel_y, rel_w, rel_h) in paper_patches:
            badge_crops.append((
                "paper_badge",
                p_img,
                (torso_x1 + rel_x, torso_y1 + rel_y, torso_x1 + rel_x + rel_w, torso_y1 + rel_y + rel_h),
            ))

    # If badge patches were isolated, return them!
    if badge_crops:
        return badge_crops

    # Otherwise return fallback crops (e.g. vest chest)
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
        if number is not None and conf >= OCR_CONF_THRESHOLD and number.strip("0") != "":
            self.history.append((number, conf))
            self.tag_visible_this_frame = True

            # Majority voting across recent readings to filter single-frame noise
            counts = Counter(num for num, _ in self.history)
            winner, count = counts.most_common(1)[0]
            matching_confs = [c for num, c in self.history if num == winner]
            avg_conf = sum(matching_confs) / len(matching_confs)

            if single_frame_mode or count >= OCR_VOTE_THRESHOLD or conf >= OCR_HIGH_CONF_LOCK:
                self.assigned_id = winner
                self.assigned_conf = avg_conf
                self.is_identified = True
                if count >= 3 or conf >= OCR_HIGH_CONF_LOCK:
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
                dist = math.hypot(dc[0] - tc[0], dc[1] - tc[1])
                if dist < best_dist:
                    best_dist = dist
                    best_track = track

            max_dist = max(130, int(dw * 0.9))
            if best_track is not None and best_dist < max_dist:
                best_track.update_box(dbox)
                matched_tracks[box_idx] = best_track
                used_tracks.add(best_track.track_id)
            else:
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
# VIDEO CAPTURE INITIALIZATION
# ============================================================

def open_capture_source(source, width=1280, height=720):
    """
    Open camera, video, or image source with DirectShow support for Windows reliability.
    Returns: (cap, is_image, static_image_frame)
    """
    is_cam_idx = isinstance(source, int) or (isinstance(source, str) and source.isdigit())
    if is_cam_idx:
        idx = int(source)
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

    src_path = Path(source)
    if not src_path.exists():
        raise FileNotFoundError(f"Source file not found: {source}")

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
    if src_path.suffix.lower() in image_extensions:
        img = cv2.imread(str(src_path))
        if img is None:
            raise RuntimeError(f"Could not read image file: {src_path}")
        return None, True, img

    cap = cv2.VideoCapture(str(src_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video file: {src_path}")

    return cap, False, None


# ============================================================
# ARGUMENT PARSER
# ============================================================

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="VisionShield Demo 5: Worker PPE & ID Detection + Machine Proximity"
    )
    parser.add_argument(
        "--source",
        default=str(DEFAULT_VIDEO if DEFAULT_VIDEO.exists() else CAMERA_INDEX),
        help=f"Video file path or camera index (default: {DEFAULT_VIDEO.name if DEFAULT_VIDEO.exists() else '0'})",
    )
    parser.add_argument(
        "--model",
        default=str(DEFAULT_PPE_MODEL),
        help="Path to 6-class YOLO PPE model checkpoint (.pt)",
    )
    parser.add_argument(
        "--machine-model",
        default=str(DEFAULT_MACHINE_MODEL if DEFAULT_MACHINE_MODEL.exists() else FALLBACK_MACHINE_MODEL),
        help="Path to YOLO machine detector model checkpoint (.pt)",
    )
    parser.add_argument(
        "--disable-machine",
        action="store_true",
        help="Disable machine detection and proximity monitoring",
    )
    parser.add_argument(
        "--machine-conf",
        type=float,
        default=MACHINE_CONFIDENCE,
        help="Confidence threshold for machine detection (default: 0.30)",
    )
    parser.add_argument(
        "--proximity-threshold",
        type=float,
        default=DEFAULT_PROXIMITY_THRESHOLD,
        help="Normalized distance threshold for proximity violation (default: 0.08)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=CONFIDENCE,
        help="Confidence threshold for YOLO PPE detections",
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
        help="Run OCR every N frames for active unverified workers (default: 8)",
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


# ============================================================
# MAIN APPLICATION LOOP
# ============================================================

def main():
    args = parse_arguments()

    print("=" * 76)
    print("VISIONSHIELD DEMO 5")
    print("INTEGRATED WORKER PPE DETECTION, WORKER ID & MACHINE PROXIMITY HAZARDS")
    print("=" * 76)

    # 1. Load PPE Model
    model_path = Path(args.model)
    print(f"\n[1/4] Loading PPE YOLO Model:\n      {model_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"PPE model checkpoint not found: {model_path}")

    ppe_model = YOLO(str(model_path))
    print("      PPE model loaded successfully.")
    print("      Classes:", ppe_model.names)

    # 2. Load Machine Detector Model
    machine_model = None
    if not args.disable_machine:
        mach_path = Path(args.machine_model)
        if not mach_path.exists():
            print(f"\n[2/4] WARNING: Machine model not found at:\n      {mach_path}")
            print("      Checking fallback path...")
            if FALLBACK_MACHINE_MODEL.exists():
                mach_path = FALLBACK_MACHINE_MODEL
                print(f"      Using fallback: {mach_path}")
            else:
                print("      Machine model could not be located. Running in PPE-only mode.")

        if mach_path.exists():
            print(f"\n[2/4] Loading Machine Detector Model:\n      {mach_path}")
            machine_model = YOLO(str(mach_path))
            print("      Machine model loaded successfully.")
            print("      Machine Classes:", machine_model.names)
    else:
        print("\n[2/4] Machine detection disabled by user flag (--disable-machine).")

    # 3. EasyOCR Reader
    print("\n[3/4] Initializing EasyOCR Reader (GPU accelerated if available)...")
    use_gpu = torch.cuda.is_available()
    print(f"      CUDA Available: {use_gpu}")
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=use_gpu)
        print("      EasyOCR initialized successfully.")
    except Exception as exc:
        print(f"      WARNING: EasyOCR init failed ({exc}). Falling back to CPU.")
        reader = easyocr.Reader(["en"], gpu=False)

    # 4. Open Capture Source
    print(f"\n[4/4] Opening source: {args.source}...")
    source_val = int(args.source) if args.source.isdigit() else args.source
    cap, is_image, static_image = open_capture_source(source_val)

    print("\n" + "=" * 76)
    print("SYSTEM READY. Controls:")
    print("  [Q] or [ESC] : Quit")
    print("  [S]          : Save current frame screenshot")
    print("  [R]          : Reset worker identities & tracker")
    print("  [P]          : Pause / Resume playback")
    print("=" * 76 + "\n")

    tracker = WorkerTracker()
    video_writer = None
    frame_number = 0
    fps = 0.0
    previous_time = cv2.getTickCount()
    paused = False

    try:
        while True:
            if not paused:
                if is_image:
                    frame = static_image.copy()
                else:
                    ret, frame = cap.read()
                    if not ret:
                        print("\nEnd of video stream reached.")
                        break

                frame_number += 1
                frame_height, frame_width = frame.shape[:2]
                diagonal = math.hypot(frame_width, frame_height)
                original_clean_frame = frame.copy()

                # ----------------------------------------------------
                # 1. PPE & WORKER DETECTION INFERENCE
                # ----------------------------------------------------
                ppe_results = ppe_model.predict(
                    frame,
                    conf=args.conf,
                    imgsz=args.imgsz,
                    verbose=False,
                )[0]

                person_detections = []
                ppe_detections = []

                if ppe_results.boxes is not None:
                    for box in ppe_results.boxes:
                        class_id = int(box.cls[0].item())
                        class_name = ppe_model.names[class_id]
                        conf_score = float(box.conf[0].item())
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                        x1 = clamp(x1, 0, frame_width - 1)
                        y1 = clamp(y1, 0, frame_height - 1)
                        x2 = clamp(x2, 0, frame_width)
                        y2 = clamp(y2, 0, frame_height)

                        det = {
                            "class": class_name,
                            "class_id": class_id,
                            "box": (x1, y1, x2, y2),
                            "confidence": conf_score,
                        }

                        if class_name.lower() == "person":
                            person_detections.append(det)
                        elif class_name in REQUIRED_PPE:
                            ppe_detections.append(det)

                # ----------------------------------------------------
                # 2. MACHINE DETECTION & HAZARD ZONES
                # ----------------------------------------------------
                machines_data = []
                if machine_model is not None:
                    mach_results = machine_model.predict(
                        frame,
                        conf=args.machine_conf,
                        imgsz=args.imgsz,
                        verbose=False,
                    )[0]

                    if mach_results.boxes is not None:
                        for b in mach_results.boxes:
                            cid = int(b.cls[0].item())
                            m_name = machine_model.names.get(cid, f"machine_{cid}")
                            m_conf = float(b.conf[0].item())
                            mx1, my1, mx2, my2 = map(int, b.xyxy[0].tolist())

                            mx1 = clamp(mx1, 0, frame_width - 1)
                            my1 = clamp(my1, 0, frame_height - 1)
                            mx2 = clamp(mx2, 0, frame_width)
                            my2 = clamp(my2, 0, frame_height)

                            m_box = (mx1, my1, mx2, my2)
                            m_zone = compute_machine_hazard_zone(
                                m_box, m_name, frame_width, frame_height
                            )

                            machines_data.append({
                                "box": m_box,
                                "label": m_name,
                                "confidence": m_conf,
                                "zone": m_zone,
                            })

                # Draw machine danger exclusion zones with transparency
                zone_polys = [m["zone"] for m in machines_data if m["zone"] is not None]
                if zone_polys:
                    zone_overlay = frame.copy()
                    for poly in zone_polys:
                        cv2.fillPoly(zone_overlay, [poly], (0, 140, 255))
                    frame = cv2.addWeighted(zone_overlay, 0.22, frame, 0.78, 0)
                    for poly in zone_polys:
                        cv2.polylines(frame, [poly], True, (0, 140, 255), 2)

                # ----------------------------------------------------
                # 3. WORKER TRACKING & PPE ASSOCIATION
                # ----------------------------------------------------
                person_boxes = [p["box"] for p in person_detections]
                matched_tracks = tracker.update(person_boxes)

                workers_data = []

                for p_idx, person in enumerate(person_detections):
                    person_box = person["box"]
                    tracked_worker = matched_tracks.get(p_idx)

                    expanded_box = expand_person_box(
                        person_box, frame_width, frame_height
                    )

                    worker_ppe = {ppe: None for ppe in REQUIRED_PPE}

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
                        # Skip heavy OCR if worker is already confirmed and locked!
                        needs_ocr = not tracked_worker.id_locked
                        due_interval = (
                            frame_number - tracked_worker.last_ocr_frame >= args.ocr_interval
                        )
                        box_h = person_box[3] - person_box[1]
                        can_ocr = box_h >= 100

                        if needs_ocr and due_interval and can_ocr:
                            tracked_worker.last_ocr_frame = frame_number

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

                        if tracked_worker.is_identified:
                            worker_id_tag = f"Worker #{tracked_worker.assigned_id}"
                            is_identified = True
                        else:
                            worker_id_tag = "Worker #UNKNOWN"
                            is_identified = False

                        display_track_id = tracked_worker.track_id
                        worker_conf = tracked_worker.assigned_conf
                        tag_box = (
                            tracked_worker.last_detected_tag_box
                            if tracked_worker.tag_visible_this_frame
                            else None
                        )
                        tag_visible = tracked_worker.tag_visible_this_frame
                    else:
                        worker_id_tag = "Worker #UNKNOWN"
                        is_identified = False
                        display_track_id = p_idx + 1
                        worker_conf = 0.0
                        tag_box = None
                        tag_visible = False

                    # ------------------------------------------------
                    # 4. PPE COMPLIANCE & VIOLATION ANALYSIS
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

                    # ------------------------------------------------
                    # 5. MACHINE PROXIMITY EVALUATION
                    # ------------------------------------------------
                    proximity_alerts = []
                    px1, py1, px2, py2 = person_box
                    foot_point = ((px1 + px2) // 2, py2)

                    for m in machines_data:
                        m_box = m["box"]
                        m_label = m["label"]
                        m_zone = m["zone"]

                        # Check if worker's feet are inside machine exclusion zone
                        inside_zone = False
                        if m_zone is not None:
                            inside_zone = (
                                cv2.pointPolygonTest(
                                    m_zone.astype(np.float32), foot_point, False
                                )
                                >= 0
                            )

                        gap_px, norm_gap = calculate_worker_machine_proximity(
                            person_box, m_box, diagonal
                        )

                        is_danger = inside_zone or (norm_gap < args.proximity_threshold)
                        is_caution = (not is_danger) and (norm_gap < DEFAULT_CAUTION_THRESHOLD)

                        if is_danger:
                            proximity_alerts.append({
                                "machine": m,
                                "type": "DANGER",
                                "label": m_label,
                                "gap_px": gap_px,
                            })
                            violations.append(f"PROXIMITY: {m_label.upper()}")
                        elif is_caution:
                            proximity_alerts.append({
                                "machine": m,
                                "type": "CAUTION",
                                "label": m_label,
                                "gap_px": gap_px,
                            })

                    # Safety Status: SAFE only if identified + no missing PPE + no proximity danger
                    status = "SAFE" if len(violations) == 0 else "UNSAFE"

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
                            "proximity_alerts": proximity_alerts,
                        }
                    )

                # ----------------------------------------------------
                # 6. DRAW MACHINE BOUNDING BOXES & LABELS
                # ----------------------------------------------------
                for m in machines_data:
                    mx1, my1, mx2, my2 = m["box"]
                    m_label = m["label"]
                    m_conf = m["confidence"]

                    is_forklift = "forklift" in m_label.lower()
                    m_color = YELLOW if is_forklift else AMBER

                    cv2.rectangle(frame, (mx1, my1), (mx2, my2), m_color, 2)
                    draw_pill(
                        frame,
                        f"{m_label.upper()} {m_conf:.2f}",
                        mx1,
                        max(115, my1 - 6),
                        (0, 100, 150),
                        WHITE,
                        scale=0.52,
                        thickness=2,
                    )

                # ----------------------------------------------------
                # 7. DRAW PROXIMITY HAZARD LINES
                # ----------------------------------------------------
                hazard_count = 0
                for w in workers_data:
                    px1, py1, px2, py2 = w["person"]["box"]
                    p_foot = ((px1 + px2) // 2, py2)

                    for alert in w["proximity_alerts"]:
                        m = alert["machine"]
                        mx1, my1, mx2, my2 = m["box"]
                        m_center = ((mx1 + mx2) // 2, (my1 + my2) // 2)

                        if alert["type"] == "DANGER":
                            hazard_count += 1
                            # Danger warning line (solid bright red)
                            cv2.line(frame, p_foot, m_center, RED, 3)
                            cv2.circle(frame, p_foot, 6, RED, -1)

                            mid_x = (p_foot[0] + m_center[0]) // 2
                            mid_y = (p_foot[1] + m_center[1]) // 2
                            draw_pill(
                                frame,
                                f"! DANGER: {alert['label'].upper()} !",
                                mid_x - 70,
                                mid_y,
                                (0, 0, 160),
                                WHITE,
                                scale=0.48,
                                thickness=2,
                            )
                        elif alert["type"] == "CAUTION":
                            # Caution warning line (yellow)
                            cv2.line(frame, p_foot, m_center, YELLOW, 2)
                            cv2.circle(frame, p_foot, 4, YELLOW, -1)

                # ----------------------------------------------------
                # 8. DRAW WORKERS, IDENTIFIED TAGS & PPE BOXES
                # ----------------------------------------------------
                for w in workers_data:
                    person = w["person"]
                    x1, y1, x2, y2 = person["box"]
                    status = w["status"]
                    is_identified = w["is_identified"]
                    worker_id_tag = w["worker_id_tag"]
                    violations = w["violations"]

                    worker_box_color = GREEN if status == "SAFE" else RED
                    cv2.rectangle(frame, (x1, y1), (x2, y2), worker_box_color, 3)

                    # Worker Header Pill Label
                    if is_identified:
                        label_text = f"{worker_id_tag} ({w['worker_conf']:.2f}) | {status}"
                        pill_bg = (0, 110, 0) if status == "SAFE" else (0, 0, 140)
                    else:
                        label_text = f"{worker_id_tag} | {status}"
                        pill_bg = (0, 0, 140)

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

                    # Draw PPE boxes
                    for ppe in REQUIRED_PPE:
                        item = w["ppe"][ppe]
                        if item is None:
                            continue

                        detection = item["detection"]
                        px1, py1, px2, py2 = detection["box"]
                        confidence = detection["confidence"]

                        cv2.rectangle(frame, (px1, py1), (px2, py2), BLUE, 2)
                        draw_text(
                            frame,
                            f"{ppe} {confidence:.2f}",
                            (px1, max(22, py1 - 6)),
                            BLUE,
                            0.48,
                            2,
                        )

                    # Highlight detected paper patch or vest badge if currently visible
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

                    # Draw Violations Pill
                    if violations:
                        violation_text = "VIOLATION: " + ", ".join(violations)
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
                # 9. OVERALL STATUS & HUD DASHBOARD
                # ----------------------------------------------------
                total_workers = len(workers_data)
                safe_count = sum(1 for w in workers_data if w["status"] == "SAFE")
                unsafe_count = sum(1 for w in workers_data if w["status"] == "UNSAFE")
                identified_count = sum(1 for w in workers_data if w["is_identified"])
                unknown_count = total_workers - identified_count
                total_machines = len(machines_data)

                if total_workers == 0:
                    overall_status = "NO WORKER"
                    overall_color = WHITE
                elif hazard_count > 0:
                    overall_status = "PROXIMITY DANGER"
                    overall_color = RED
                elif unsafe_count > 0:
                    overall_status = "VIOLATION"
                    overall_color = RED
                else:
                    overall_status = "SAFE"
                    overall_color = GREEN

                # Draw Top HUD
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, 0), (frame_width, 105), DARK_GRAY, -1)
                frame = cv2.addWeighted(overlay, 0.80, frame, 0.20, 0)

                if frame_width < 1000:
                    col1_x = 15
                    col2_x = int(frame_width * 0.26)
                    col3_x = int(frame_width * 0.52)
                    col4_x = int(frame_width * 0.74)
                    hud_scale = 0.50
                    stat_scale = 0.66
                else:
                    col1_x = 20
                    col2_x = int(frame_width * 0.26)
                    col3_x = int(frame_width * 0.52)
                    col4_x = max(int(frame_width * 0.75), frame_width - 290)
                    hud_scale = 0.58
                    stat_scale = 0.82

                # Column 1: Worker & Compliance Counters
                draw_text(frame, f"Workers: {total_workers}", (col1_x, 30), WHITE, hud_scale, 2)
                draw_text(frame, f"Safe: {safe_count}", (col1_x, 60), GREEN, hud_scale, 2)
                draw_text(frame, f"Violations: {unsafe_count}", (col1_x, 90), RED, hud_scale, 2)

                # Column 2: ID Tag Tracking Counters
                draw_text(frame, f"Identified IDs: {identified_count}", (col2_x, 42), CYAN, hud_scale, 2)
                draw_text(
                    frame,
                    f"Unknown Tags: {unknown_count}",
                    (col2_x, 78),
                    YELLOW if unknown_count > 0 else WHITE,
                    hud_scale,
                    2,
                )

                # Column 3: Machinery & Proximity Counters
                draw_text(frame, f"Machines: {total_machines}", (col3_x, 42), AMBER, hud_scale, 2)
                draw_text(
                    frame,
                    f"Proximity Hazards: {hazard_count}",
                    (col3_x, 78),
                    RED if hazard_count > 0 else GREEN,
                    hud_scale,
                    2,
                )

                # Column 4: Overall Facility Status Banner
                draw_text(
                    frame,
                    f"STATUS: {overall_status}",
                    (col4_x, 58),
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

                draw_text(
                    frame,
                    "Q: Quit | S: Screenshot | R: Reset Tracker | P: Pause",
                    (20, frame_height - 20),
                    WHITE,
                    0.48,
                    1,
                )

                # ----------------------------------------------------
                # SAVE OUTPUT IF REQUESTED
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

                if args.max_frames and frame_number >= args.max_frames:
                    print(f"\nReached max-frames limit ({args.max_frames}). Stopping...")
                    break

            # --------------------------------------------------------
            # SHOW FRAME (IF NOT HEADLESS)
            # --------------------------------------------------------
            if not args.no_window:
                cv2.imshow(
                    "VisionShield Demo 5 - Worker PPE, IDs & Machine Proximity",
                    frame,
                )
                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), ord("Q"), 27):
                    print("\nQuit key received. Stopping...")
                    break
                elif key in (ord("s"), ord("S")):
                    timestamp = time.strftime("%Y%m%d_%H%M%S")
                    output_path = BASE / f"visionshield_demo5_capture_{timestamp}.jpg"
                    cv2.imwrite(str(output_path), frame)
                    print(f"\n[Screenshot Saved] -> {output_path}")
                elif key in (ord("r"), ord("R")):
                    tracker = WorkerTracker()
                    print("\n[Tracker Reset] Worker IDs and histories cleared.")
                elif key in (ord("p"), ord("P")):
                    paused = not paused
                    print(f"\n[{'Paused' if paused else 'Resumed'}]")
            else:
                if is_image:
                    break

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"[Output Video Saved] -> {args.output}")
        if cap is not None:
            cap.release()
        cv2.destroyAllWindows()
        print("\nVisionShield Demo 5 terminated successfully.")


if __name__ == "__main__":
    main()
