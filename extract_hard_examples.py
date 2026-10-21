
import cv2
from pathlib import Path

video = Path(
    r"C:\Users\vedan\Desktop\Python\VisionShield"
    r"\CINEMATIC CORPORATE VIDEO  Factory B ROLL.mp4"
)

output = Path("hard_examples") / "frames"
output.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(str(video))

if not cap.isOpened():
    raise FileNotFoundError(f"Cannot open video: {video}")

fps = cap.get(cv2.CAP_PROP_FPS)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

if fps <= 0:
    cap.release()
    raise RuntimeError("Could not determine video frame rate.")

duration = total / fps
saved = 0

for second in range(int(duration)):
    cap.set(cv2.CAP_PROP_POS_MSEC, second * 1000)
    ok, frame = cap.read()

    if ok:
        path = output / f"frame_{second:05d}.jpg"
        if cv2.imwrite(str(path), frame):
            saved += 1

cap.release()
print(f"Extracted {saved} frames to: {output.resolve()}")
