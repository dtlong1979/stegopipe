from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..bitstream import bits_to_bytes, bytes_to_bits, int_to_bits, bits_to_int
LENGTH_BITS = 32

class CarrierError(ValueError):
    pass

class Carrier(ABC):
    name: str = 'base'

    @abstractmethod
    def capacity_bits(self, image: np.ndarray) -> int:
        pass

    @abstractmethod
    def embed(self, image: np.ndarray, blob: bytes) -> np.ndarray:
        pass

    @abstractmethod
    def extract(self, image: np.ndarray) -> bytes:
        pass

    def _blob_to_bits(self, blob: bytes, capacity: int) -> np.ndarray:
        payload_bits = bytes_to_bits(blob)
        total = LENGTH_BITS + payload_bits.size
        if total > capacity:
            raise CarrierError(f'blob needs {total} bits but carrier holds {capacity} ({self.name}); use a larger image or a higher-capacity method')
        header = int_to_bits(len(blob), LENGTH_BITS)
        return np.concatenate([header, payload_bits])

    def _read_length(self, bit_reader) -> int:
        return bits_to_int(np.array([bit_reader(i) for i in range(LENGTH_BITS)], dtype=np.uint8))
