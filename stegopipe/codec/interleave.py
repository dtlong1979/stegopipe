from __future__ import annotations
import numpy as np
from ..bitstream import bits_to_bytes, bytes_to_bits

class ByteInterleaver:

    def __init__(self, depth: int=32):
        if depth < 2:
            raise ValueError('depth must be >= 2')
        self.depth = depth

    def forward(self, data: bytes) -> bytes:
        pad = -len(data) % self.depth
        buf = data + bytes(pad)
        arr = np.frombuffer(buf, dtype=np.uint8).reshape(-1, self.depth)
        header = len(data).to_bytes(4, 'big')
        return header + arr.T.tobytes()

    def inverse(self, data: bytes) -> bytes:
        orig_len = int.from_bytes(data[:4], 'big')
        body = np.frombuffer(data[4:], dtype=np.uint8).reshape(self.depth, -1)
        return body.T.tobytes()[:orig_len]

class SpreadRepetition:

    def __init__(self, r: int=5):
        if r < 1 or r % 2 == 0:
            raise ValueError('r must be a positive odd integer')
        self.r = r
    _HREP = 9

    def forward(self, data: bytes) -> bytes:
        bits = bytes_to_bits(data)
        n = bits.size
        coded = np.tile(bits, self.r)
        pad = -coded.size % 8
        if pad:
            coded = np.concatenate([coded, np.zeros(pad, np.uint8)])
        header_bits = np.tile(np.unpackbits(np.frombuffer(n.to_bytes(4, 'big'), np.uint8)), self._HREP)
        return bits_to_bytes(header_bits) + bits_to_bytes(coded)

    def inverse(self, data: bytes) -> bytes:
        hbytes = 4 * self._HREP
        hbits = bytes_to_bits(data[:hbytes]).reshape(self._HREP, 32)
        n = int.from_bytes(np.packbits((hbits.sum(0) * 2 > self._HREP).astype(np.uint8)).tobytes(), 'big')
        bits = bytes_to_bits(data[hbytes:])
        if self.r * n > bits.size:
            return b''
        copies = bits[:self.r * n].reshape(self.r, n)
        decoded = (copies.sum(0) * 2 > self.r).astype(np.uint8)
        decoded = decoded[:decoded.size // 8 * 8]
        return bits_to_bytes(decoded)
