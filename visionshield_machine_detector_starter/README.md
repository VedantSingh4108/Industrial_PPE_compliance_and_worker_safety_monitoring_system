# VisionShield Machine Detector — Starter Pack

This is a separate machine-detection training pipeline. It does not modify your existing six-class PPE model and does not include FastAPI.

Target classes (class IDs must stay in this order):
0. forklift
1. conveyor
2. industrial_machine

## 1. Dataset folder structure

Create this structure under your VisionShield project:

```text
VisionShield/
  datasets/
    machine_detector/
      images/
        train/
        val/
        test/       # optional but strongly recommended
      labels/
        train/
        val/
        test/       # optional
      machine.yaml
```

Each image needs a YOLO-format `.txt` annotation with one object per line:

`class_id x_center y_center width height`

Coordinates are normalized to 0–1. Draw a tight box around the visible machine. Do not label the same machine inconsistently across images. If an image contains multiple machines, annotate each visible instance.

Recommended initial data target:
- Aim for 300–500 labelled images per class if feasible; this is a starting target, not a guarantee.
- Include variation in distance, lighting, partial occlusion, machine appearance, and background.
- For a fixed camera, capture the real target scene as well as representative images from similar views.
- Split by video/session or capture sequence, not random adjacent frames. Near-duplicate frames across train/val/test will inflate metrics.
- Ensure every class appears in train and validation. Keep a held-out test set for final review.

If `industrial_machine` covers many visually different machine types, define its scope clearly before annotation. A single broad class can be hard to learn consistently; consider separate classes only if the project scope and data support them.

## 2. Install

Activate the same Python environment you use for VisionShield, then:

```bash
pip install ultralytics opencv-python pyyaml
```

## 3. Configure dataset

Copy `machine.yaml.example` to `datasets/machine_detector/machine.yaml` and update paths if needed. Example uses absolute Windows paths to avoid working-directory ambiguity.

## 4. Train

From the VisionShield project root:

```bash
python train_machine_detector.py --data datasets/machine_detector/machine.yaml
```

Default settings: YOLO11n pretrained weights, 80 epochs, image size 640, batch size 8, early stopping patience 15. Change these only after establishing a baseline. Use `--device 0` if a supported NVIDIA GPU is configured; use `--device cpu` for CPU (slower).

The output should be under:

`runs/detect/visionshield_machine_yolo11n/weights/best.pt`

The script prints the resolved path at the end. Do not overwrite the PPE model.

## 5. Evaluate

```bash
python evaluate_machine_detector.py --data datasets/machine_detector/machine.yaml --weights runs/detect/visionshield_machine_yolo11n/weights/best.pt
```

Use validation results during iteration. Use the held-out test split only for final evaluation. Record mAP50, mAP50-95, precision, recall, per-class metrics, inference speed, and examples of false positives/false negatives.

## 6. Test on webcam

```bash
python test_machine_webcam.py --weights runs/detect/visionshield_machine_yolo11n/weights/best.pt --camera 0
```

Press Q to exit. This only tests machine detection. It does not calculate real-world distance or trigger industrial safety actions.

## 7. First milestone / acceptance criteria

Before adding proximity:
1. All three classes have correctly labelled train and validation data.
2. Validation results are reviewed per class, not just overall mAP.
3. Test set includes real images from the fixed webcam scene.
4. Review false positives and missed detections, especially partially visible or visually similar machinery.
5. Measure webcam FPS and end-to-end detection latency on the target computer.
6. Do not claim distance-in-metres accuracy until calibration and distance ground-truth tests are performed.

## Important
A detector bounding box is not a machine's exact physical footprint. For proximity, machine geometry will need manual footprint/zone configuration or a separate geometry annotation/calibration step. The model is a prototype, not a certified machine-guarding system.
