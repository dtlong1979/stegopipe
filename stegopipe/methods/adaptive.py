from __future__ import annotations
import numpy as np
from .base import Carrier, LENGTH_BITS
from ..bitstream import bits_to_bytes, bits_to_int

class AdaptiveLSBM(Carrier):
    name = 'adaptive'

    def __init__(self, key: int | None=None):
        self.key = key

    @staticmethod
    def _texture(plane: np.ndarray) -> np.ndarray:
        p = np.pad(plane, 1, mode='reflect')
        up, down = (p[:-2, 1:-1], p[2:, 1:-1])
        left, right = (p[1:-1, :-2], p[1:-1, 2:])
        neigh = np.stack([up, down, left, right], axis=0).astype(np.float64)
        return neigh.std(axis=0)

    def _order(self, image: np.ndarray) -> np.ndarray:
        arr = image if image.ndim == 3 else image[..., None]
        h, w, c = arr.shape
        yy, xx = np.mgrid[0:h, 0:w]
        a_mask = (yy + xx) % 2 == 0
        orders = []
        rng = np.random.default_rng(self.key) if self.key is not None else None
        for ch in range(c):
            texture = self._texture(arr[..., ch].astype(np.float64))
            ys, xs = np.nonzero(a_mask)
            score = texture[ys, xs]
            if rng is not None:
                score = score + rng.random(score.size) * 1e-06
            local_order = np.argsort(-score, kind='stable')
            flat = (ys[local_order] * w + xs[local_order]) * c + ch
            orders.append(flat)
        return np.concatenate(orders)

    def capacity_bits(self, image: np.ndarray) -> int:
        arr = image if image.ndim == 3 else image[..., None]
        h, w, c = arr.shape
        n_a = int((np.mgrid[0:h, 0:w].sum(0) % 2 == 0).sum()) * c
        return n_a - LENGTH_BITS

    def embed(self, image: np.ndarray, blob: bytes) -> np.ndarray:
        bits = self._blob_to_bits(blob, self.capacity_bits(image) + LENGTH_BITS)
        order = self._order(image)[:bits.size]
        flat = image.reshape(-1).astype(np.int16).copy()
        vals = flat[order]
        need = vals & 1 != bits
        rng = np.random.default_rng(0 if self.key is None else self.key + 7)
        direction = rng.integers(0, 2, size=vals.size) * 2 - 1
        direction = np.where(vals <= 0, 1, direction)
        direction = np.where(vals >= 255, -1, direction)
        flat[order] = np.where(need, vals + direction, vals)
        return np.clip(flat, 0, 255).astype(np.uint8).reshape(image.shape)

    def extract(self, image: np.ndarray) -> bytes:
        order = self._order(image)
        low = image.reshape(-1).astype(np.uint8)[order] & 1
        length = bits_to_int(low[:LENGTH_BITS])
        total = LENGTH_BITS + length * 8
        return bits_to_bytes(low[LENGTH_BITS:total])
