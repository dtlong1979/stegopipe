from __future__ import annotations
import numpy as np
from .base import Carrier, LENGTH_BITS
from ..bitstream import bits_to_bytes, int_to_bits, bits_to_int

class LSB(Carrier):
    name = 'lsb'

    def __init__(self, bits_per_channel: int=1, key: int | None=None):
        if not 1 <= bits_per_channel <= 4:
            raise ValueError('bits_per_channel must be in 1..4')
        self.bits_per_channel = bits_per_channel
        self.key = key

    def _order(self, n: int) -> np.ndarray:
        idx = np.arange(n)
        if self.key is not None:
            rng = np.random.default_rng(self.key)
            rng.shuffle(idx)
        return idx

    def capacity_bits(self, image: np.ndarray) -> int:
        return image.size * self.bits_per_channel - LENGTH_BITS

    def embed(self, image: np.ndarray, blob: bytes) -> np.ndarray:
        bits = self._blob_to_bits(blob, self.capacity_bits(image) + LENGTH_BITS)
        flat = image.reshape(-1).astype(np.uint8).copy()
        order = self._order(flat.size)
        bpc = self.bits_per_channel
        n_samples = (bits.size + bpc - 1) // bpc
        padded = np.zeros(n_samples * bpc, dtype=np.uint8)
        padded[:bits.size] = bits
        chunks = padded.reshape(n_samples, bpc)
        values = np.zeros(n_samples, dtype=np.uint8)
        for i in range(bpc):
            values |= (chunks[:, i] << bpc - 1 - i).astype(np.uint8)
        targets = order[:n_samples]
        mask = np.uint8(255 ^ (1 << bpc) - 1)
        flat[targets] = flat[targets] & mask | values
        return flat.reshape(image.shape)

    def extract(self, image: np.ndarray) -> bytes:
        flat = image.reshape(-1).astype(np.uint8)
        order = self._order(flat.size)
        bpc = self.bits_per_channel
        low_mask = (1 << bpc) - 1

        def sample_bits(sample_index: int) -> np.ndarray:
            v = flat[order[sample_index]] & low_mask
            return np.array([v >> bpc - 1 - i & 1 for i in range(bpc)], dtype=np.uint8)
        header_bits = []
        s = 0
        while len(header_bits) < LENGTH_BITS:
            header_bits.extend(sample_bits(s).tolist())
            s += 1
        length = bits_to_int(np.array(header_bits[:LENGTH_BITS], dtype=np.uint8))
        total_bits = LENGTH_BITS + length * 8
        n_samples = (total_bits + bpc - 1) // bpc
        collected = []
        for i in range(n_samples):
            collected.extend(sample_bits(i).tolist())
        payload_bits = np.array(collected[LENGTH_BITS:total_bits], dtype=np.uint8)
        return bits_to_bytes(payload_bits)
