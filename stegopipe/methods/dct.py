from __future__ import annotations
import numpy as np
from .base import Carrier, LENGTH_BITS
from ..bitstream import bits_to_bytes, bits_to_int
BLOCK = 8
COEF = (4, 1)

def _dct_matrix(n: int=BLOCK) -> np.ndarray:
    k = np.arange(n).reshape(-1, 1)
    x = np.arange(n).reshape(1, -1)
    d = np.cos(np.pi * (2 * x + 1) * k / (2 * n))
    d *= np.sqrt(2.0 / n)
    d[0, :] *= np.sqrt(1.0 / 2.0)
    return d
_D = _dct_matrix()

def _dct2(block: np.ndarray) -> np.ndarray:
    return _D @ block @ _D.T

def _idct2(coef: np.ndarray) -> np.ndarray:
    return _D.T @ coef @ _D

class DCT(Carrier):
    name = 'dct'

    def __init__(self, step: float=16.0, key: int | None=None):
        if step < 8:
            raise ValueError('step < 8 is not robust to 8-bit round-trip; use >= 8')
        self.step = float(step)
        self.key = key

    def _channels(self, image: np.ndarray) -> int:
        return 1 if image.ndim == 2 else image.shape[2]

    def _block_order(self, image: np.ndarray) -> list[tuple[int, int, int]]:
        h, w = image.shape[:2]
        blocks = []
        for c in range(self._channels(image)):
            for by in range(h // BLOCK):
                for bx in range(w // BLOCK):
                    blocks.append((c, by, bx))
        if self.key is not None:
            rng = np.random.default_rng(self.key)
            rng.shuffle(blocks)
        return blocks

    def capacity_bits(self, image: np.ndarray) -> int:
        h, w = image.shape[:2]
        n_blocks = h // BLOCK * (w // BLOCK) * self._channels(image)
        return n_blocks - LENGTH_BITS

    def _get_block(self, work: np.ndarray, c: int, by: int, bx: int) -> np.ndarray:
        ys, xs = (by * BLOCK, bx * BLOCK)
        if work.ndim == 2:
            return work[ys:ys + BLOCK, xs:xs + BLOCK]
        return work[ys:ys + BLOCK, xs:xs + BLOCK, c]

    def _set_block(self, work: np.ndarray, c: int, by: int, bx: int, block: np.ndarray) -> None:
        ys, xs = (by * BLOCK, bx * BLOCK)
        if work.ndim == 2:
            work[ys:ys + BLOCK, xs:xs + BLOCK] = block
        else:
            work[ys:ys + BLOCK, xs:xs + BLOCK, c] = block

    def embed(self, image: np.ndarray, blob: bytes) -> np.ndarray:
        bits = self._blob_to_bits(blob, self.capacity_bits(image) + LENGTH_BITS)
        work = image.astype(np.float64).copy()
        order = self._block_order(image)
        for i, bit in enumerate(bits):
            c, by, bx = order[i]
            block = self._get_block(work, c, by, bx)
            coef = _dct2(block)
            q = np.round(coef[COEF] / self.step)
            if int(q) % 2 != int(bit):
                q += 1
            coef[COEF] = q * self.step
            self._set_block(work, c, by, bx, _idct2(coef))
        return np.clip(np.round(work), 0, 255).astype(np.uint8)

    def extract(self, image: np.ndarray) -> bytes:
        work = image.astype(np.float64)
        order = self._block_order(image)

        def read_bit(i: int) -> int:
            c, by, bx = order[i]
            coef = _dct2(self._get_block(work, c, by, bx))
            return int(np.round(coef[COEF] / self.step)) % 2
        length = bits_to_int(np.array([read_bit(i) for i in range(LENGTH_BITS)], dtype=np.uint8))
        total_bits = LENGTH_BITS + length * 8
        payload_bits = np.array([read_bit(i) for i in range(LENGTH_BITS, total_bits)], dtype=np.uint8)
        return bits_to_bytes(payload_bits)
