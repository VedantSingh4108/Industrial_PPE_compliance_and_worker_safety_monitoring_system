import os
import time
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as F
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn,
    FasterRCNN_ResNet50_FPN_Weights
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


# ============================================================
# CONFIG
# ============================================================
DATASET_ROOT = r"C:\Users\vedan\Desktop\Python\VisionShield\datasets\construction-ppe"

TRAIN_IMAGES = os.path.join(DATASET_ROOT, "images", "train")
TRAIN_LABELS = os.path.join(DATASET_ROOT, "labels", "train")

VAL_IMAGES = os.path.join(DATASET_ROOT, "images", "val")
VAL_LABELS = os.path.join(DATASET_ROOT, "labels", "val")

NUM_CLASSES = 12  # 11 PPE classes + background
NUM_EPOCHS = 20
BATCH_SIZE = 1
IMAGE_SIZE = 640

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


CLASS_NAMES = [
    "helmet",
    "gloves",
    "vest",
    "boots",
    "goggles",
    "none",
    "Person",
    "no_helmet",
    "no_goggle",
    "no_gloves",
    "no_boots"
]


# ============================================================
# DATASET
# ============================================================

class ConstructionPPEDataset(Dataset):

    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        self.images.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(self.image_dir, image_name)

        label_name = os.path.splitext(image_name)[0] + ".txt"
        label_path = os.path.join(self.label_dir, label_name)

        image = Image.open(image_path).convert("RGB")

        width, height = image.size

        boxes = []
        labels = []

        if os.path.exists(label_path):

            with open(label_path, "r") as f:

                for line in f:

                    parts = line.strip().split()

                    if len(parts) != 5:
                        continue

                    class_id = int(parts[0])

                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    box_width = float(parts[3])
                    box_height = float(parts[4])

                    # YOLO normalized coordinates
                    x_center *= width
                    y_center *= height
                    box_width *= width
                    box_height *= height

                    xmin = x_center - box_width / 2
                    ymin = y_center - box_height / 2
                    xmax = x_center + box_width / 2
                    ymax = y_center + box_height / 2

                    # Keep boxes inside image
                    xmin = max(0, xmin)
                    ymin = max(0, ymin)
                    xmax = min(width, xmax)
                    ymax = min(height, ymax)

                    if xmax > xmin and ymax > ymin:

                        boxes.append([xmin, ymin, xmax, ymax])

                        # Faster R-CNN labels start at 1.
                        labels.append(class_id + 1)

        boxes = torch.as_tensor(boxes, dtype=torch.float32)

        labels = torch.as_tensor(labels, dtype=torch.int64)

        if len(boxes) == 0:

            boxes = torch.zeros((0, 4), dtype=torch.float32)

            labels = torch.zeros((0,), dtype=torch.int64)

        image_tensor = F.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
        }

        return image_tensor, target


# ============================================================
# COLLATE FUNCTION
# ============================================================

def collate_fn(batch):

    return tuple(zip(*batch))


# ============================================================
# CREATE DATASETS
# ============================================================

print("\nLoading dataset...")

train_dataset = ConstructionPPEDataset(
    TRAIN_IMAGES,
    TRAIN_LABELS
)

val_dataset = ConstructionPPEDataset(
    VAL_IMAGES,
    VAL_LABELS
)

print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0,
    collate_fn=collate_fn
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
    collate_fn=collate_fn
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading pretrained Faster R-CNN...")

weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT

model = fasterrcnn_resnet50_fpn(
    weights=weights
)

# Replace COCO prediction head
in_features = model.roi_heads.box_predictor.cls_score.in_features

model.roi_heads.box_predictor = FastRCNNPredictor(
    in_features,
    NUM_CLASSES
)

model.to(DEVICE)


# ============================================================
# OPTIMIZER
# ============================================================

params = [
    p for p in model.parameters()
    if p.requires_grad
]

optimizer = torch.optim.SGD(
    params,
    lr=0.005,
    momentum=0.9,
    weight_decay=0.0005
)


# ============================================================
# TRAIN
# ============================================================

print("\nDevice:", DEVICE)
print("Starting training...\n")

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

        loss_dict = model(
            images,
            targets
        )

        losses = sum(
            loss for loss in loss_dict.values()
        )

        losses.backward()

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
# SAVE MODEL
# ============================================================

output_path = "faster_rcnn_construction_ppe.pth"

torch.save(
    model.state_dict(),
    output_path
)

print("\nTraining complete!")

print(
    "Model saved to:",
    os.path.abspath(output_path)
)