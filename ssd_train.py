import os
import time
import torch
from torch.utils.data import DataLoader
from torchvision.models.detection import ssd300_vgg16
from torchvision.models.vgg import VGG16_Weights

from faster_rcnn_dataset import (
    ConstructionPPEDataset,
    collate_fn
)


# ============================================================
# CONFIG
# ============================================================

DATASET_ROOT = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\datasets\construction-ppe"
)

TRAIN_IMAGES = os.path.join(
    DATASET_ROOT, "images", "train"
)

TRAIN_LABELS = os.path.join(
    DATASET_ROOT, "labels", "train"
)

NUM_CLASSES = 12
NUM_EPOCHS = 20
BATCH_SIZE = 1

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# DATASET
# ============================================================

print("\nLoading dataset...")

train_dataset = ConstructionPPEDataset(
    TRAIN_IMAGES,
    TRAIN_LABELS
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

print("Training images:", len(train_dataset))


# ============================================================
# MODEL
# ============================================================

print("Loading SSD300 with pretrained VGG16 backbone...")

model = ssd300_vgg16(
    weights=None,
    weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

print("SSD model loaded successfully.")


# ============================================================
# FREEZE BACKBONE
# ============================================================

for param in model.backbone.parameters():
    param.requires_grad = False

print("VGG16 backbone frozen.")


# ============================================================
# OPTIMIZER
# ============================================================

params = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.SGD(
    params,
    lr=0.0005,
    momentum=0.9,
    weight_decay=0.0005
)


# ============================================================
# TRAIN
# ============================================================

print("\nDevice:", DEVICE)
print("Starting SSD training...\n")

model.train()

for epoch in range(NUM_EPOCHS):

    epoch_start = time.time()
    total_loss = 0.0

    for batch_idx, (images, targets) in enumerate(train_loader):

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

        optimizer.zero_grad()

        # Forward
        loss_dict = model(
            images,
            targets
        )

        losses = sum(
            loss for loss in loss_dict.values()
        )

        # NaN / Inf protection
        if not torch.isfinite(losses):

            print(
                f"\nWARNING: Non-finite loss at "
                f"Epoch {epoch + 1}, "
                f"Batch {batch_idx + 1}"
            )

            print({
                k: v.item()
                for k, v in loss_dict.items()
            })

            optimizer.zero_grad()
            continue

        # Backward
        losses.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            params,
            max_norm=5.0
        )

        # Update
        optimizer.step()

        total_loss += losses.item()

        if (batch_idx + 1) % 100 == 0:

            print(
                f"Epoch [{epoch + 1}/{NUM_EPOCHS}] "
                f"Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {losses.item():.4f}"
            )

    epoch_time = time.time() - epoch_start

    avg_loss = total_loss / len(train_loader)

    print(
        f"\nEpoch {epoch + 1}/{NUM_EPOCHS} "
        f"completed | "
        f"Average Loss: {avg_loss:.4f} | "
        f"Time: {epoch_time:.1f}s\n"
    )


# ============================================================
# SAVE
# ============================================================

output_path = "ssd_construction_ppe.pth"

torch.save(
    model.state_dict(),
    output_path
)

print("\nTraining complete!")

print(
    "Model saved to:",
    os.path.abspath(output_path)
)