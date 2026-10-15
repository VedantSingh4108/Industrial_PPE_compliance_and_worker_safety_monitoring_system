import os
import time
import torch

from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    ssd300_vgg16,
)
from torchvision.models import VGG16_Weights
from torch.utils.data import DataLoader
from torchmetrics.detection import MeanAveragePrecision

from faster_rcnn_dataset import ConstructionPPEDataset


# ============================================================
# CONFIG
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DATASET_ROOT = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\datasets\construction-ppe"
)

FRCNN_WEIGHTS = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\faster_rcnn_construction_ppe.pth"
)

SSD_WEIGHTS = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\ssd_construction_ppe.pth"
)

NUM_CLASSES = 12
CONF_THRESHOLD = 0.25


# ============================================================
# DATASET
# ============================================================

def collate_fn(batch):
    return tuple(zip(*batch))


VAL_IMAGES = os.path.join(
    DATASET_ROOT,
    "images",
    "val"
)

VAL_LABELS = os.path.join(
    DATASET_ROOT,
    "labels",
    "val"
)

val_dataset = ConstructionPPEDataset(
    VAL_IMAGES,
    VAL_LABELS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    collate_fn=collate_fn
)

print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)
print(f"Validation images : {len(val_dataset)}")
print(f"Device             : {DEVICE}")
print()


# ============================================================
# LOAD FASTER R-CNN
# ============================================================

def load_faster_rcnn():

    print("Loading Faster R-CNN...")

    model = fasterrcnn_resnet50_fpn(
        weights=None,
        weights_backbone=None,
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        FRCNN_WEIGHTS,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("Faster R-CNN loaded.")

    return model


# ============================================================
# LOAD SSD300
# ============================================================

def load_ssd():

    print("Loading SSD300...")

    model = ssd300_vgg16(
        weights=None,
        weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        SSD_WEIGHTS,
        map_location=DEVICE,
        weights_only=True
    )

    model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    print("SSD300 loaded.")

    return model


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate_model(model, model_name):

    print()
    print("=" * 60)
    print(f"EVALUATING {model_name}")
    print("=" * 60)

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox"
    )

    total_inference_time = 0.0
    image_count = 0

    # --------------------------------------------------------
    # GPU WARM-UP
    # --------------------------------------------------------

    print("Warming up GPU...")

    warmup_count = 0

    for images, targets in val_loader:

        images = [
            image.to(DEVICE)
            for image in images
        ]

        model(images)

        warmup_count += 1

        if warmup_count >= 3:
            break

    if DEVICE.type == "cuda":
        torch.cuda.synchronize()

    print("Warm-up complete.")
    print("Running evaluation...")

    # --------------------------------------------------------
    # EVALUATION LOOP
    # --------------------------------------------------------

    for idx, (images, targets) in enumerate(val_loader):

        images = [
            image.to(DEVICE)
            for image in images
        ]

        # ----------------------------------------------------
        # TIME INFERENCE
        # ----------------------------------------------------

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        start_time = time.perf_counter()

        outputs = model(images)

        if DEVICE.type == "cuda":
            torch.cuda.synchronize()

        end_time = time.perf_counter()

        total_inference_time += (
            end_time - start_time
        )

        image_count += len(images)

        # ----------------------------------------------------
        # PREDICTIONS
        # ----------------------------------------------------

        predictions = []

        for output in outputs:

            boxes = output["boxes"].detach().cpu()
            scores = output["scores"].detach().cpu()
            labels = output["labels"].detach().cpu()

            keep = scores >= CONF_THRESHOLD

            boxes = boxes[keep]
            scores = scores[keep]
            labels = labels[keep]

            predictions.append({
                "boxes": boxes,
                "scores": scores,
                "labels": labels
            })

        # ----------------------------------------------------
        # GROUND TRUTH
        # ----------------------------------------------------

        ground_truths = []

        for target in targets:

            boxes = target["boxes"].detach().cpu()
            labels = target["labels"].detach().cpu()

            ground_truths.append({
                "boxes": boxes,
                "labels": labels
            })

        # ----------------------------------------------------
        # UPDATE METRIC
        # ----------------------------------------------------

        metric.update(
            predictions,
            ground_truths
        )

        if (idx + 1) % 20 == 0:

            print(
                f"Processed "
                f"{idx + 1}/{len(val_loader)} images"
            )

    # ========================================================
    # COMPUTE METRICS
    # ========================================================

    results = metric.compute()

    map50 = results["map_50"].item()
    map5095 = results["map"].item()
    recall = results["mar_100"].item()

    # --------------------------------------------------------
    # SPEED
    # --------------------------------------------------------

    avg_inference_ms = (
        total_inference_time /
        image_count
    ) * 1000

    fps = 1000 / avg_inference_ms

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print()
    print("-" * 60)
    print(f"{model_name} RESULTS")
    print("-" * 60)

    print(f"mAP@50        : {map50:.4f}")
    print(f"mAP@50:95     : {map5095:.4f}")
    print(f"Recall        : {recall:.4f}")
    print(f"Avg inference : {avg_inference_ms:.2f} ms/image")
    print(f"FPS           : {fps:.2f}")

    print("-" * 60)

    return {
        "model": model_name,
        "map50": map50,
        "map5095": map5095,
        "recall": recall,
        "inference_ms": avg_inference_ms,
        "fps": fps
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # ========================================================
    # FASTER R-CNN
    # ========================================================

    faster_model = load_faster_rcnn()

    faster_results = evaluate_model(
        faster_model,
        "Faster R-CNN"
    )

    del faster_model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ========================================================
    # SSD300
    # ========================================================

    ssd_model = load_ssd()

    ssd_results = evaluate_model(
        ssd_model,
        "SSD300"
    )

    del ssd_model

    if DEVICE.type == "cuda":
        torch.cuda.empty_cache()

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    print()
    print()
    print("=" * 80)
    print("FINAL TORCHVISION MODEL COMPARISON")
    print("=" * 80)

    print(
        f"{'Model':18s}"
        f"{'mAP50':>10s}"
        f"{'mAP50:95':>12s}"
        f"{'Recall':>10s}"
        f"{'ms/img':>12s}"
        f"{'FPS':>10s}"
    )

    print("-" * 80)

    for result in [
        faster_results,
        ssd_results
    ]:

        print(
            f"{result['model']:18s}"
            f"{result['map50']:10.4f}"
            f"{result['map5095']:12.4f}"
            f"{result['recall']:10.4f}"
            f"{result['inference_ms']:12.2f}"
            f"{result['fps']:10.2f}"
        )

    print("=" * 80)