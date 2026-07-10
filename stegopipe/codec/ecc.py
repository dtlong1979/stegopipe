from __future__ import annotations
import numpy as np
from ..bitstream import bits_to_bytes, bytes_to_bits

def _pad_to_byte(bits: np.ndarray) -> np.ndarray:
    remainder = -bits.size % 8
    if remainder:
        bits = np.concatenate([bits, np.zeros(remainder, dtype=np.uint8)])
    return bits

class RepetitionECC:

    def __init__(self, r: int=3):
        if r < 1 or r % 2 == 0:
            raise ValueError('r must be a positive odd integer (majority vote)')
        self.r = r

    def forward(self, data: bytes) -> bytes:
        bits = bytes_to_bits(data)
        coded = np.repeat(bits, self.r)
        return bits_to_bytes(_pad_to_byte(coded))

    def inverse(self, data: bytes) -> bytes:
        coded = bytes_to_bits(data)
        n_groups = coded.size // self.r
        coded = coded[:n_groups * self.r].reshape(n_groups, self.r)
        decoded = (coded.sum(axis=1) * 2 > self.r).astype(np.uint8)
        decoded = decoded[:decoded.size // 8 * 8]
        return bits_to_bytes(decoded)
_G = np.array([[1, 0, 0, 0, 1, 1, 0], [0, 1, 0, 0, 1, 0, 1], [0, 0, 1, 0, 0, 1, 1], [0, 0, 0, 1, 1, 1, 1]], dtype=np.uint8)
_H = np.array([[1, 1, 0, 1, 1, 0, 0], [1, 0, 1, 1, 0, 1, 0], [0, 1, 1, 1, 0, 0, 1]], dtype=np.uint8)
_SYNDROME_TO_BIT = {}
for _col in range(7):
    _s = int(_H[0, _col]) << 2 | int(_H[1, _col]) << 1 | int(_H[2, _col])
    _SYNDROME_TO_BIT[_s] = _col

class Hamming74:

    def forward(self, data: bytes) -> bytes:
        bits = bytes_to_bits(data)
        pad = -bits.size % 4
        if pad:
            bits = np.concatenate([bits, np.zeros(pad, dtype=np.uint8)])
        blocks = bits.reshape(-1, 4)
        coded = blocks @ _G % 2
        return bits_to_bytes(_pad_to_byte(coded.ravel().astype(np.uint8)))

    def inverse(self, data: bytes) -> bytes:
        coded = bytes_to_bits(data)
        n_blocks = coded.size // 7
        blocks = coded[:n_blocks * 7].reshape(n_blocks, 7).astype(np.uint8)
        synd = blocks @ _H.T % 2
        synd_int = synd[:, 0] * 4 + synd[:, 1] * 2 + synd[:, 2]
        for i, s in enumerate(synd_int):
            if s:
                bit = _SYNDROME_TO_BIT.get(int(s))
                if bit is not None:
                    blocks[i, bit] ^= 1
        data_bits = blocks[:, :4].ravel()
        data_bits = data_bits[:data_bits.size // 8 * 8]
        return bits_to_bytes(data_bits)
