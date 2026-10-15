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


# ============================================================
# DATASET
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

print("Loading SSD...")

model = ssd300_vgg16(
    weights=None,
    weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)
model.train()


# Freeze backbone
for param in model.backbone.parameters():
    param.requires_grad = False


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
# TRAINING STABILITY TEST
# ============================================================

print("\nStarting stability test...\n")

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

    optimizer.zero_grad()

    # Forward
    loss_dict = model(
        images,
        targets
    )

    losses = sum(loss_dict.values())

    # Check forward loss
    if not torch.isfinite(losses):

        print("\n!!! FORWARD LOSS BECAME NaN !!!")
        print("Batch:", batch_idx)

        print(
            "Loss dict:",
            {
                k: v.item()
                for k, v in loss_dict.items()
            }
        )

        break

    # Backward
    losses.backward()

    # Check gradients
    bad_gradient = False

    for name, param in model.named_parameters():

        if param.grad is not None:

            if not torch.isfinite(param.grad).all():

                print("\n!!! GRADIENT BECAME NaN/INF !!!")
                print("Batch:", batch_idx)
                print("Parameter:", name)

                bad_gradient = True
                break

    if bad_gradient:
        break

    # Clip gradients
    torch.nn.utils.clip_grad_norm_(
        params,
        max_norm=5.0
    )

    # Optimizer
    optimizer.step()

    # Check weights
    bad_weight = False

    for name, param in model.named_parameters():

        if not torch.isfinite(param).all():

            print("\n!!! WEIGHT BECAME NaN/INF !!!")
            print("Batch:", batch_idx)
            print("Parameter:", name)

            bad_weight = True
            break

    if bad_weight:
        break

    # Progress
    if (batch_idx + 1) % 10 == 0:

        print(
            f"Batch {batch_idx + 1}: "
            f"Loss = {losses.item():.4f}"
        )

else:

    print("\n================================")
    print("100% OF TRAINING SET COMPLETED")
    print("No NaN detected.")
    print("================================")