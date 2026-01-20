# MapVerse/perturb_utils.py
"""
Image utility helpers used across model scripts:
- add_random_black_box
- jpeg_compress
- add_random_noise
- reduce_pixels_70 / 50 / 25
- blackout_random_pixels_pil
"""

from PIL import Image, ImageDraw
import io
import numpy as np
import random
from typing import Optional

__all__ = [
    "add_random_black_box",
    "jpeg_compress",
    "add_random_noise",
    "reduce_pixels_70",
    "reduce_pixels_50",
    "reduce_pixels_25",
    "blackout_random_pixels_pil",
]

def add_random_black_box(pil_img: Image.Image, area_ratio: float = 0.1) -> Image.Image:
    """Draw a black rectangle covering ~area_ratio of the image area."""
    img_w, img_h = pil_img.size
    img_area = img_w * img_h
    target_area = img_area * area_ratio

    aspect_ratio = random.uniform(0.5, 2.0)
    box_w = int((target_area * aspect_ratio) ** 0.5)
    box_h = int((target_area / aspect_ratio) ** 0.5)

    box_w = min(box_w, img_w)
    box_h = min(box_h, img_h)

    x1 = random.randint(0, img_w - box_w)
    y1 = random.randint(0, img_h - box_h)
    x2 = x1 + box_w
    y2 = y1 + box_h

    img_copy = pil_img.copy()
    draw = ImageDraw.Draw(img_copy)
    draw.rectangle([x1, y1, x2, y2], fill="black")
    return img_copy

def jpeg_compress(pil_img: Image.Image, quality: int = 50) -> Image.Image:
    """Apply JPEG compression and return a PIL image (RGB)."""
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=int(quality))
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")

def add_random_noise(pil_img: Image.Image, noise_level: int = 25) -> Image.Image:
    """Add uniform random noise in [-noise_level, noise_level] to each channel."""
    arr = np.array(pil_img).astype(np.int16)
    noise = np.random.randint(-noise_level, noise_level + 1, arr.shape, dtype=np.int16)
    noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(noisy)

def reduce_pixels_70(pil_img: Image.Image) -> Image.Image:
    """Reduce resolution to ~70% (via sqrt(0.5) factor) to emulate quality drop."""
    w, h = pil_img.size
    new_w, new_h = max(1, int(w / (2 ** 0.5))), max(1, int(h / (2 ** 0.5)))
    return pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)

def reduce_pixels_50(pil_img: Image.Image) -> Image.Image:
    """Reduce resolution to 50%."""
    w, h = pil_img.size
    return pil_img.resize((max(1, w // 2), max(1, h // 2)), Image.Resampling.LANCZOS)

def reduce_pixels_25(pil_img: Image.Image) -> Image.Image:
    """Reduce resolution to 25%."""
    w, h = pil_img.size
    return pil_img.resize((max(1, w // 4), max(1, h // 4)), Image.Resampling.LANCZOS)

def blackout_random_pixels_pil(input_image: Image.Image, blackout_percentage: float = 0.1) -> Image.Image:
    """
    Black out a random fraction of pixels (more memory-efficient implementation).
    """
    img = input_image.convert("RGB")
    arr = np.array(img)
    h, w = arr.shape[:2]
    total = h * w
    k = int(total * float(blackout_percentage))
    if k <= 0:
        return img
    flat = arr.reshape(-1, 3)
    idx = np.random.choice(flat.shape[0], k, replace=False)
    flat[idx] = 0
    return Image.fromarray(flat.reshape((h, w, 3)).astype("uint8"))