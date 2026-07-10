from __future__ import annotations
import numpy as np

def bytes_to_bits(data: bytes) -> np.ndarray:
    if len(data) == 0:
        return np.zeros(0, dtype=np.uint8)
    arr = np.frombuffer(data, dtype=np.uint8)
    return np.unpackbits(arr)

def bits_to_bytes(bits: np.ndarray) -> bytes:
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    if bits.size % 8 != 0:
        raise ValueError(f'bit length {bits.size} is not a multiple of 8')
    if bits.size == 0:
        return b''
    return np.packbits(bits).tobytes()

def int_to_bits(value: int, width: int) -> np.ndarray:
    if value < 0:
        raise ValueError('value must be non-negative')
    if value >= 1 << width:
        raise ValueError(f'value {value} does not fit in {width} bits')
    return np.array([value >> width - 1 - i & 1 for i in range(width)], dtype=np.uint8)

def bits_to_int(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits, dtype=np.uint8).ravel():
        value = value << 1 | int(bit)
    return value
