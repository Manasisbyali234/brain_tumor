"""Generate synthetic brain scan images for demo purposes."""
import os
import numpy as np
import cv2

np.random.seed(42)

def make_brain(size=224, has_tumor=False):
    img = np.zeros((size, size), dtype=np.uint8)
    cx, cy = size // 2, size // 2
    cv2.ellipse(img, (cx, cy), (80, 90), 0, 0, 360, 180, -1)
    cv2.ellipse(img, (cx, cy), (60, 70), 0, 0, 360, 140, -1)
    img = cv2.GaussianBlur(img, (15, 15), 0)
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    if has_tumor:
        tx = cx + np.random.randint(-30, 30)
        ty = cy + np.random.randint(-30, 30)
        tr = np.random.randint(10, 25)
        cv2.circle(img, (tx, ty), tr, 230, -1)
        img = cv2.GaussianBlur(img, (5, 5), 0)
    return img

for split, cls, has_tumor, n in [
    ("ct", "tumor", True, 60), ("ct", "no_tumor", False, 60),
    ("mri", "tumor", True, 60), ("mri", "no_tumor", False, 60),
]:
    folder = os.path.join("data", split, cls)
    os.makedirs(folder, exist_ok=True)
    for i in range(n):
        img = make_brain(has_tumor=has_tumor)
        cv2.imwrite(os.path.join(folder, f"{i:03d}.png"), img)

print("Done — 240 synthetic images generated.")
