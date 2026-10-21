
from pathlib import Path
from PIL import Image, ImageDraw
import math

root = Path(".")
list_file = root / "conveyor_negative_candidates.txt"
output_file = root / "conveyor_candidates_contact_sheet.jpg"

paths = [
    Path(line.strip())
    for line in list_file.read_text(encoding="utf-8").splitlines()
    if line.strip()
]

# Show a spread of 48 images from the candidate list.
if len(paths) > 48:
    indices = list(range(min(48, len(paths))))
    paths = [paths[i] for i in indices]

thumb_w, thumb_h = 220, 165
label_h = 42
columns = 4
rows = math.ceil(len(paths) / columns)

sheet = Image.new(
    "RGB",
    (columns * thumb_w, rows * (thumb_h + label_h)),
    "white"
)
draw = ImageDraw.Draw(sheet)

for i, path in enumerate(paths):
    x = (i % columns) * thumb_w
    y = (i // columns) * (thumb_h + label_h)

    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            im.thumbnail((thumb_w - 8, thumb_h - 8))
            px = x + (thumb_w - im.width) // 2
            py = y + (thumb_h - im.height) // 2
            sheet.paste(im, (px, py))

        # Display image number and filename.
        draw.text((x + 4, y + thumb_h), f"{i+1}. {path.name[:30]}", fill="black")
        draw.text((x + 4, y + thumb_h + 17), str(path.parent.parent.name), fill="gray")

    except Exception as e:
        draw.text((x + 4, y + 10), f"Could not open: {path.name[:20]}", fill="red")

sheet.save(output_file, quality=90)
print(f"Contact sheet saved: {output_file.resolve()}")
print(f"Images included: {len(paths)}")
