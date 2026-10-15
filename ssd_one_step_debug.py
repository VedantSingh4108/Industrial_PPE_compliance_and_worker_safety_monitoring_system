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

model = ssd300_vgg16(
    weights=None,
    weights_backbone=VGG16_Weights.IMAGENET1K_FEATURES,
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)
model.train()

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

print("Testing ONE optimization step...\n")

images, targets = next(iter(loader))

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

# ------------------------------------------------------------
# BEFORE
# ------------------------------------------------------------

print("Checking weights BEFORE training step...")

for name, param in model.named_parameters():

    if not torch.isfinite(param).all():

        print("BAD WEIGHT BEFORE:", name)
        break

else:
    print("All weights finite.")

# ------------------------------------------------------------
# FORWARD
# ------------------------------------------------------------

loss_dict = model(
    images,
    targets
)

print("\nLoss before backward:")

for name, loss in loss_dict.items():
    print(f"{name}: {loss.item()}")

losses = sum(loss_dict.values())

print("Total:", losses.item())

# ------------------------------------------------------------
# BACKWARD
# ------------------------------------------------------------

optimizer.zero_grad()

losses.backward()

print("\nChecking gradients...")

bad_grad = False

for name, param in model.named_parameters():

    if param.grad is not None:

        if not torch.isfinite(param.grad).all():

            print("BAD GRADIENT:", name)

            print(
                "Gradient min:",
                param.grad.nan_to_num().min().item()
            )

            print(
                "Gradient max:",
                param.grad.nan_to_num().max().item()
            )

            bad_grad = True
            break

if not bad_grad:
    print("All gradients finite.")

# ------------------------------------------------------------
# OPTIMIZER
# ------------------------------------------------------------

print("\nPerforming ONE optimizer step...")

torch.nn.utils.clip_grad_norm_(
    params,
    max_norm=5.0
)

optimizer.step()

# ------------------------------------------------------------
# AFTER
# ------------------------------------------------------------

print("\nChecking weights AFTER optimizer step...")

bad_weight = False

for name, param in model.named_parameters():

    if not torch.isfinite(param).all():

        print("BAD WEIGHT AFTER:", name)

        bad_weight = True
        break

if not bad_weight:
    print("All weights finite after optimizer step.")

print("\nDONE.")