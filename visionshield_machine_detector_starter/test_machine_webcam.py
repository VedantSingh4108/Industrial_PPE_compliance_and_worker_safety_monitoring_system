import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser(description="Live webcam test for VisionShield machine detector.")
    p.add_argument("--weights", required=True)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--conf", type=float, default=0.25)
    p.add_argument("--imgsz", type=int, default=640)
    args = p.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {Path(args.weights).resolve()}")

    model = YOLO(args.weights)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {args.camera}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    print("Machine detector webcam test. Press Q to quit.")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Could not read frame from webcam.")
                break
            result = model.predict(frame, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
            annotated = result.plot()
            cv2.imshow("VisionShield - Machine Detector", annotated)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
