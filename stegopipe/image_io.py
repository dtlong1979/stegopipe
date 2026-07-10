from __future__ import annotations
from pathlib import Path
import numpy as np
from PIL import Image
LOSSLESS_SUFFIXES = {'.png', '.bmp', '.tif', '.tiff', '.ppm', '.pgm'}

def load_image(path: str | Path) -> np.ndarray:
    img = Image.open(path)
    if img.mode not in ('L', 'RGB', 'RGBA'):
        img = img.convert('RGB')
    return np.array(img, dtype=np.uint8)

def save_image(array: np.ndarray, path: str | Path) -> None:
    path = Path(path)
    if path.suffix.lower() not in LOSSLESS_SUFFIXES:
        raise ValueError(f'{path.suffix!r} is a lossy/unknown format; use one of {sorted(LOSSLESS_SUFFIXES)} to preserve embedded bits')
    array = np.asarray(array, dtype=np.uint8)
    mode = 'L' if array.ndim == 2 else {3: 'RGB', 4: 'RGBA'}[array.shape[2]]
    Image.fromarray(array, mode=mode).save(path)

def jpeg_roundtrip(array: np.ndarray, quality: int=90) -> np.ndarray:
    import io
    array = np.asarray(array, dtype=np.uint8)
    mode = 'L' if array.ndim == 2 else {3: 'RGB', 4: 'RGB'}[array.shape[2]]
    if array.ndim == 3 and array.shape[2] == 4:
        array = array[..., :3]
    buf = io.BytesIO()
    Image.fromarray(array, mode=mode).save(buf, format='JPEG', quality=quality, subsampling=0)
    buf.seek(0)
    return np.array(Image.open(buf), dtype=np.uint8)

def make_noise_image(height: int, width: int, channels: int=3, seed: int=0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    shape = (height, width) if channels == 1 else (height, width, channels)
    return rng.integers(0, 256, size=shape, dtype=np.uint8)

def make_gradient_image(height: int, width: int, channels: int=3) -> np.ndarray:
    xs = np.linspace(0, 255, width, dtype=np.float64)
    ys = np.linspace(0, 255, height, dtype=np.float64)
    base = (xs[None, :] + ys[:, None]) / 2.0
    base = base.astype(np.uint8)
    if channels == 1:
        return base
    return np.stack([base, np.roll(base, 5, axis=1), np.roll(base, 9, axis=0)], axis=-1)[..., :channels]
