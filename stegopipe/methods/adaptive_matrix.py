from __future__ import annotations
import numpy as np
from .adaptive import AdaptiveLSBM
from .base import LENGTH_BITS, CarrierError
from ..bitstream import bytes_to_bits, bits_to_bytes, int_to_bits, bits_to_int

def _parity_matrix(k: int) -> np.ndarray:
    n = 2 ** k - 1
    return np.array([[j >> k - 1 - i & 1 for j in range(1, n + 1)] for i in range(k)], dtype=np.uint8)

class MatrixAdaptive(AdaptiveLSBM):
    name = 'amx'

    def __init__(self, k: int=3, key: int | None=None):
        super().__init__(key)
        if not 1 <= k <= 8:
            raise ValueError('k must be in 1..8')
        self.k = k
        self.n = 2 ** k - 1
        self.H = _parity_matrix(k)

    def capacity_bits(self, image: np.ndarray) -> int:
        n_a = super().capacity_bits(image) + LENGTH_BITS
        return (n_a - LENGTH_BITS) // self.n * self.k

    def _flip_dir(self, val, rng):
        return 1 if val <= 0 else -1 if val >= 255 else int(rng.integers(0, 2)) * 2 - 1

    def embed(self, image: np.ndarray, blob: bytes) -> np.ndarray:
        bits = bytes_to_bits(blob)
        if bits.size > self.capacity_bits(image):
            raise CarrierError(f'blob needs {bits.size} bits > capacity {self.capacity_bits(image)} (amx k={self.k})')
        pad = -bits.size % self.k
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, np.uint8)])
        order = self._order(image)
        flat = image.reshape(-1).astype(np.int16).copy()
        rng = np.random.default_rng(7 if self.key is None else self.key + 7)
        for i, b in enumerate(int_to_bits(len(blob), LENGTH_BITS)):
            p = order[i]
            if flat[p] & 1 != b:
                flat[p] += self._flip_dir(flat[p], rng)
        pos = LENGTH_BITS
        for gi, g in enumerate(range(0, bits.size, self.k)):
            m = bits[g:g + self.k].astype(np.uint8)
            grp = order[pos + gi * self.n:pos + (gi + 1) * self.n]
            syndrome = self.H @ (flat[grp] & 1) % 2
            diff = syndrome ^ m
            col = int(''.join(map(str, diff)), 2)
            if col:
                p = grp[col - 1]
                flat[p] += self._flip_dir(flat[p], rng)
        return np.clip(flat, 0, 255).astype(np.uint8).reshape(image.shape)

    def extract(self, image: np.ndarray) -> bytes:
        order = self._order(image)
        flat = image.reshape(-1).astype(np.uint8)
        length = bits_to_int(flat[order[:LENGTH_BITS]] & 1)
        nbits = length * 8
        ngroups = (nbits + self.k - 1) // self.k
        pos = LENGTH_BITS
        out = []
        for gi in range(ngroups):
            grp = order[pos + gi * self.n:pos + (gi + 1) * self.n]
            out.extend((self.H @ (flat[grp] & 1) % 2).tolist())
        bits = np.array(out[:nbits], dtype=np.uint8)
        return bits_to_bytes(bits[:bits.size // 8 * 8])
