import os
import time
import torch
from torch.utils.data import DataLoader

from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from faster_rcnn_dataset import (
    ConstructionPPEDataset,
    collate_fn
)


DATASET_ROOT = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\datasets\construction-ppe"
)

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

MODEL_PATH = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\faster_rcnn_construction_ppe.pth"
)

NUM_CLASSES = 12

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("\nLoading validation dataset...")

val_dataset = ConstructionPPEDataset(
    VAL_IMAGES,
    VAL_LABELS
)

val_loader = DataLoader(
    val_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print(
    "Validation images:",
    len(val_dataset)
)


print("\nLoading trained Faster R-CNN...")

model = fasterrcnn_resnet50_fpn(
    weights=None,
    weights_backbone=None
)

in_features = (
    model.roi_heads
    .box_predictor
    .cls_score
    .in_features
)

model.roi_heads.box_predictor = (
    FastRCNNPredictor(
        in_features,
        NUM_CLASSES
    )
)

model.load_state_dict(
    torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True
    )
)

model.to(DEVICE)
model.eval()


print("Device:", DEVICE)
print("\nRunning inference...\n")


total_time = 0.0
total_images = 0

with torch.no_grad():

    for images, targets in val_loader:

        images = [
            image.to(DEVICE)
            for image in images
        ]

        start = time.perf_counter()

        predictions = model(images)

        end = time.perf_counter()

        total_time += end - start
        total_images += len(images)


avg_time = total_time / total_images
fps = 1 / avg_time


print("\n======================================")
print("Faster R-CNN Evaluation")
print("======================================")

print(
    f"Images evaluated : {total_images}"
)

print(
    f"Total inference  : {total_time:.2f} sec"
)

print(
    f"Avg inference    : {avg_time * 1000:.2f} ms/image"
)

print(
    f"Approx FPS       : {fps:.2f}"
)

print("======================================")
print("Evaluation complete.")