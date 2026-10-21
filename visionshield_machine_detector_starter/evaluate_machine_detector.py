import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    p = argparse.ArgumentParser(description="Evaluate VisionShield machine detector.")
    p.add_argument("--data", default="datasets/machine_detector/machine.yaml")
    p.add_argument("--weights", required=True)
    p.add_argument("--split", choices=["val", "test"], default="val")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    p.add_argument("--project", default="runs/detect")
    p.add_argument("--name", default="visionshield_machine_eval")
    args = p.parse_args()

    if not Path(args.data).exists():
        raise FileNotFoundError(f"Dataset YAML not found: {Path(args.data).resolve()}")
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {Path(args.weights).resolve()}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        device=args.device,
        plots=True,
        project=args.project,
        name=args.name,
        verbose=True,
    )
    print("\nEvaluation summary")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")
    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print("Review per-class metrics and confusion matrix in the generated validation folder.")

if __name__ == "__main__":
    main()
