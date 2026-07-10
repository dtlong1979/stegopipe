from __future__ import annotations
import numpy as np
from .base import Carrier, LENGTH_BITS
from ..bitstream import bits_to_bytes, bits_to_int

class LSBMatching(Carrier):
    name = 'lsbm'

    def __init__(self, key: int | None=None):
        self.key = key

    def _order(self, n: int) -> np.ndarray:
        idx = np.arange(n)
        if self.key is not None:
            rng = np.random.default_rng(self.key)
            rng.shuffle(idx)
        return idx

    def _pm_rng(self):
        return np.random.default_rng(0 if self.key is None else self.key + 1)

    def capacity_bits(self, image: np.ndarray) -> int:
        return image.size - LENGTH_BITS

    def embed(self, image: np.ndarray, blob: bytes) -> np.ndarray:
        bits = self._blob_to_bits(blob, self.capacity_bits(image) + LENGTH_BITS)
        flat = image.reshape(-1).astype(np.int16).copy()
        order = self._order(flat.size)[:bits.size]
        rng = self._pm_rng()
        vals = flat[order]
        need_flip = vals & 1 != bits
        directions = rng.integers(0, 2, size=vals.size) * 2 - 1
        directions = np.where(vals <= 0, 1, directions)
        directions = np.where(vals >= 255, -1, directions)
        vals = np.where(need_flip, vals + directions, vals)
        flat[order] = vals
        return np.clip(flat, 0, 255).astype(np.uint8).reshape(image.shape)

    def extract(self, image: np.ndarray) -> bytes:
        flat = image.reshape(-1).astype(np.uint8)
        order = self._order(flat.size)
        low = flat[order] & 1
        length = bits_to_int(low[:LENGTH_BITS])
        total_bits = LENGTH_BITS + length * 8
        payload_bits = low[LENGTH_BITS:total_bits]
        return bits_to_bytes(payload_bits)
