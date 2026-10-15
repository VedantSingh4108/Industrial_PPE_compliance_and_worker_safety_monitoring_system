import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.vgg import VGG16_Weights

from faster_rcnn_dataset import (
    ConstructionPPEDataset,
    collate_fn
)

DATASET_ROOT = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\datasets\construction-ppe"
)

TRAIN_IMAGES = DATASET_ROOT + r"\images\train"
TRAIN_LABELS = DATASET_ROOT + r"\labels\train"

NUM_CLASSES = 12

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Loading dataset...")

dataset = ConstructionPPEDataset(
    TRAIN_IMAGES,
    TRAIN_LABELS
)

loader = DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)

print("Loading SSD...")

model = ssd300_vgg16(
    weights=None,
    weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)
model.train()

print("Testing first batches...\n")

for batch_idx, (images, targets) in enumerate(loader):

    images = [
        image.to(DEVICE)
        for image in images
    ]

    targets = [
        {
            k: v.to(DEVICE)
            for k, v in target.items()
        }
        for target in targets
    ]

    with torch.no_grad():

        loss_dict = model(
            images,
            targets
        )

    print(
        f"Batch {batch_idx}:",
        {
            k: float(v)
            for k, v in loss_dict.items()
        }
    )

    total_loss = sum(loss_dict.values())

    if not torch.isfinite(total_loss):
        print("\n!!! NAN DETECTED !!!")
        print("Image index:", batch_idx)
        print("Boxes:")
        print(targets[0]["boxes"])
        print("Labels:")
        print(targets[0]["labels"])
        break

    if batch_idx >= 19:
        print("\nFirst 20 batches are OK.")
        break