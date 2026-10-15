import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F


class ConstructionPPEDataset(Dataset):

    def __init__(self, image_dir, label_dir):
        self.image_dir = image_dir
        self.label_dir = label_dir

        self.images = sorted([
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image_name = self.images[idx]

        image_path = os.path.join(
            self.image_dir,
            image_name
        )

        label_path = os.path.join(
            self.label_dir,
            os.path.splitext(image_name)[0] + ".txt"
        )

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

                    x_center = float(parts[1]) * width
                    y_center = float(parts[2]) * height
                    box_width = float(parts[3]) * width
                    box_height = float(parts[4]) * height

                    xmin = max(
                        0,
                        x_center - box_width / 2
                    )

                    ymin = max(
                        0,
                        y_center - box_height / 2
                    )

                    xmax = min(
                        width,
                        x_center + box_width / 2
                    )

                    ymax = min(
                        height,
                        y_center + box_height / 2
                    )

                    if xmax > xmin and ymax > ymin:

                        boxes.append(
                            [xmin, ymin, xmax, ymax]
                        )

                        # Faster R-CNN reserves label 0
                        # for background.
                        labels.append(class_id + 1)

        boxes = torch.as_tensor(
            boxes,
            dtype=torch.float32
        )

        labels = torch.as_tensor(
            labels,
            dtype=torch.int64
        )

        if len(boxes) == 0:

            boxes = torch.zeros(
                (0, 4),
                dtype=torch.float32
            )

            labels = torch.zeros(
                (0,),
                dtype=torch.int64
            )

        image = F.to_tensor(image)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx])
        }

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))