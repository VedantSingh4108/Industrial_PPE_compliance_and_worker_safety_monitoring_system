"""
VisionShield automatic worker-to-production-machine proximity prototype.
Prototype only: image-space zones are NOT physical distances or safety-rated boundaries.
"""
import argparse
from pathlib import Path
import cv2
import numpy as np
from ultralytics import YOLO

DEFAULT_MACHINE_MODEL = r"runs\detect\runs\detect\visionshield_machine_2class_aug40\weights\best.pt"
DEFAULT_PERSON_MODEL = "yolo11n.pt"
DEFAULT_VIDEO = r"C:\Users\vedan\Desktop\Python\VisionShield\CINEMATIC CORPORATE VIDEO  Factory B ROLL.mp4"

def args_parse():
    p = argparse.ArgumentParser()
    p.add_argument("source", nargs="?", default=DEFAULT_VIDEO, help="Video path or camera index, e.g. 0")
    p.add_argument("--machine-model", default=DEFAULT_MACHINE_MODEL)
    p.add_argument("--person-model", default=DEFAULT_PERSON_MODEL)
    p.add_argument("--machine-conf", type=float, default=0.35)
    p.add_argument("--person-conf", type=float, default=0.35)
    p.add_argument("--zone-width", type=float, default=0.18)
    p.add_argument("--zone-height", type=float, default=0.45)
    p.add_argument("--stable-frames", type=int, default=5)
    p.add_argument("--save", action="store_true")
    p.add_argument("--output", default="worker_machine_proximity_auto.mp4")
    return p.parse_args()

def class_name(names, i):
    if isinstance(names, dict):
        return str(names.get(i, i))
    if isinstance(names, (list, tuple)) and 0 <= i < len(names):
        return str(names[i])
    return str(i)

def right_zone(box, w, h, width_ratio, height_ratio):
    x1, y1, x2, y2 = box
    bw, bh = max(1, x2-x1), max(1, y2-y1)
    zw, zh = max(20, int(bw*width_ratio)), max(20, int(bh*height_ratio))
    top = max(0, min(h-1, y1 + int(bh*0.30)))
    bottom = max(0, min(h-1, top+zh))
    left = max(0, min(w-1, x2))
    right = max(0, min(w-1, x2+zw))
    if right <= left or bottom <= top:
        return None
    return np.array([[left, top],
                     [right, top + int((bottom-top)*0.15)],
                     [right, bottom],
                     [left, bottom]], dtype=np.int32)

def main():
    a = args_parse()
    machine_path = Path(a.machine_model)
    if not machine_path.is_file():
        raise FileNotFoundError(
            f"Machine checkpoint not found: {machine_path.resolve()}\n"
            "Use --machine-model with the correct existing best.pt path."
        )

    print("Loading machine model:", machine_path.resolve())
    machine_model = YOLO(str(machine_path))
    print("Loading person model:", a.person_model)
    person_model = YOLO(a.person_model)

    source = int(a.source) if a.source.isdigit() else a.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open source: {a.source}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 1:
        fps = 25.0
    fw, fh = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = None
    if a.save:
        out = Path(a.output).resolve()
        out.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (fw, fh))
        if not writer.isOpened():
            cap.release()
            raise RuntimeError(f"Could not create output video: {out}")
        print("Output video:", out)

    streaks = {}
    frame_count = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_count += 1
            h, w = frame.shape[:2]

            mr = machine_model.predict(frame, conf=a.machine_conf, verbose=False)[0]
            machines = []
            if mr.boxes is not None:
                for b in mr.boxes:
                    cid = int(b.cls.item())
                    name = class_name(mr.names, cid)
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    machines.append((x1, y1, x2, y2, name, float(b.conf.item())))

            # COCO class 0 is "person".
            pr = person_model.predict(frame, classes=[0], conf=a.person_conf, verbose=False)[0]
            people = []
            if pr.boxes is not None:
                for b in pr.boxes:
                    x1, y1, x2, y2 = map(int, b.xyxy[0].tolist())
                    people.append((x1, y1, x2, y2, float(b.conf.item())))

            zones = []
            for x1, y1, x2, y2, name, conf in machines:
                is_machine = name.lower().replace(" ", "_") == "production_machine"
                color = (255, 170, 0) if is_machine else (180, 180, 180)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"{name} {conf:.2f}", (x1, max(20, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
                if is_machine:
                    poly = right_zone((x1, y1, x2, y2), w, h, a.zone_width, a.zone_height)
                    if poly is not None:
                        zones.append(poly)

            for poly in zones:
                overlay = frame.copy()
                cv2.fillPoly(overlay, [poly], (0, 165, 255))
                cv2.addWeighted(overlay, 0.20, frame, 0.80, 0, frame)
                cv2.polylines(frame, [poly], True, (0, 165, 255), 2)

            new_streaks = {}
            alert = False
            for idx, (x1, y1, x2, y2, conf) in enumerate(people):
                foot = (int((x1+x2)/2), int(y2))
                inside = any(cv2.pointPolygonTest(poly.astype(np.float32), foot, False) >= 0
                             for poly in zones)
                streak = streaks.get(idx, 0) + 1 if inside else 0
                new_streaks[idx] = streak
                risk = inside and streak >= max(1, a.stable_frames)
                alert = alert or risk
                color = (0, 0, 255) if risk else (0, 255, 0)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.circle(frame, foot, 5, color, -1)
                cv2.putText(frame, f"person {conf:.2f}", (x1, max(20, y1-6)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
            streaks = new_streaks

            status = "POTENTIAL MACHINE RISK - VERIFY" if alert else "NO PROXIMITY ALERT"
            status_color = (0, 0, 255) if alert else (0, 200, 0)
            cv2.rectangle(frame, (0, 0), (w, 40), (20, 20, 20), -1)
            cv2.putText(frame, status, (12, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.imshow("VisionShield - Automatic Machine Zone", frame)
            if writer is not None:
                writer.write(frame)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()
    print(f"Processed {frame_count} frames.")
    if a.save:
        print("Saved:", Path(a.output).resolve())

if __name__ == "__main__":
    main()
