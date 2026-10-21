import argparse
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description="Train VisionShield's separate machine detector.")
    parser.add_argument("--data", default="datasets/machine_detector/machine.yaml")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--device", default="0", help="Use 0 for first CUDA GPU, cpu for CPU, or auto if supported.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="visionshield_machine_yolo11n")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_path.resolve()}\n"
            "Copy machine.yaml.example to datasets/machine_detector/machine.yaml and update paths."
        )

    model = YOLO(args.model)
    results = model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        project=args.project,
        name=args.name,
        pretrained=True,
        plots=True,
        verbose=True,
        seed=42,
    )
    best = Path(args.project) / args.name / "weights" / "best.pt"
    print("\nTraining complete.")
    print(f"Best weights expected at: {best.resolve()}")
    print("Check the training output above if your Ultralytics version used a different output path.")

if __name__ == "__main__":
    main()
