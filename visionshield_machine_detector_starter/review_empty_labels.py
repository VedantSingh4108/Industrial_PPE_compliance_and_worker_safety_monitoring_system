
from pathlib import Path
from PIL import Image, ImageDraw
import math

root = Path("dataset/raw/prod_machines")
items = []

for split in ("train", "valid", "test"):
    images = root / split / "images"
    labels = root / split / "labels"

    if not labels.exists() or not images.exists():
        continue

    for label in labels.rglob("*.txt"):
        if label.read_text(encoding="utf-8", errors="ignore").strip():
            continue

        relative = label.relative_to(labels)
        matches = [
            images / relative.with_suffix(ext)
            for ext in (".jpg", ".jpeg", ".png", ".webp", ".bmp")
        ]
        image_path = next((p for p in matches if p.exists()), None)

        if image_path:
            items.append((split, image_path))

if not items:
    print("No matching images found.")
    raise SystemExit

cols = 3
tw, th, label_h = 320, 240, 48
rows = math.ceil(len(items) / cols)
sheet = Image.new("RGB", (cols * tw, rows * (th + label_h)), "white")
draw = ImageDraw.Draw(sheet)

for i, (split, path) in enumerate(items):
    x = (i % cols) * tw
    y = (i // cols) * (th + label_h)

    with Image.open(path) as im:
        im = im.convert("RGB")
        im.thumbnail((tw - 10, th - 10))
        sheet.paste(im, (x + (tw - im.width) // 2,
                         y + (th - im.height) // 2))

    draw.text((x + 5, y + th), f"{split}: {path.name[:38]}", fill="black")

out = Path("prod_machines_empty_labels_review.jpg")
sheet.save(out, quality=92)
print("Images:", len(items))
print("Saved:", out.resolve())
