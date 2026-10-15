import torch

from faster_rcnn_dataset import ConstructionPPEDataset

DATASET_ROOT = (
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\datasets\construction-ppe"
)

TRAIN_IMAGES = DATASET_ROOT + r"\images\train"
TRAIN_LABELS = DATASET_ROOT + r"\labels\train"

dataset = ConstructionPPEDataset(
    TRAIN_IMAGES,
    TRAIN_LABELS
)

print("Images:", len(dataset))
print("\nChecking targets...\n")

empty = 0
bad = 0
total_boxes = 0
min_width = float("inf")
min_height = float("inf")

for i in range(len(dataset)):

    image, target = dataset[i]

    boxes = target["boxes"]
    labels = target["labels"]

    # Empty target
    if boxes.shape[0] == 0:
        print("EMPTY TARGET:", i)
        empty += 1
        continue

    total_boxes += boxes.shape[0]

    # Check finite
    if not torch.isfinite(boxes).all():
        print("BAD: non-finite boxes:", i)
        print(boxes)
        bad += 1
        continue

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    widths = x2 - x1
    heights = y2 - y1

    min_width = min(min_width, widths.min().item())
    min_height = min(min_height, heights.min().item())

    if (widths <= 0).any():
        print("BAD: zero/negative width:", i)
        print(boxes)
        bad += 1

    if (heights <= 0).any():
        print("BAD: zero/negative height:", i)
        print(boxes)
        bad += 1

    if not torch.isfinite(labels.float()).all():
        print("BAD: non-finite labels:", i)
        print(labels)
        bad += 1

    if labels.numel() > 0:
        if (labels < 1).any() or (labels > 11).any():
            print("BAD: invalid labels:", i)
            print(labels)
            bad += 1


print("\n================================")
print("Total images:", len(dataset))
print("Empty targets:", empty)
print("Bad samples:", bad)
print("Total boxes:", total_boxes)
print("Smallest box width:", min_width)
print("Smallest box height:", min_height)
print("================================")