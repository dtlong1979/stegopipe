from __future__ import annotations
from dataclasses import dataclass, field
from typing import Protocol
import numpy as np
from .crypto import decrypt, derive_key, encrypt
from .framing import FrameStage
from .methods import Carrier, get_method

class Stage(Protocol):

    def forward(self, data: bytes) -> bytes:
        ...

    def inverse(self, data: bytes) -> bytes:
        ...

@dataclass
class EncryptStage:
    key: str | bytes
    nonce: bytes = b''

    def _material(self) -> bytes:
        if isinstance(self.key, bytes) and len(self.key) == 32:
            return self.key
        return derive_key(self.key)

    def forward(self, data: bytes) -> bytes:
        return encrypt(data, self._material(), self.nonce)

    def inverse(self, data: bytes) -> bytes:
        return decrypt(data, self._material(), self.nonce)

@dataclass
class Pipeline:
    carrier: Carrier | str
    passphrase: str | None = None
    frame: bool = True
    stages: list[Stage] = field(default_factory=list)
    aead: str | bytes | None = None
    aead_algorithm: str = 'aes-gcm'

    def __post_init__(self) -> None:
        if isinstance(self.carrier, str):
            self.carrier = get_method(self.carrier)
        if self.aead is not None and self.passphrase is not None:
            raise ValueError('set either `aead` (recommended) or `passphrase`, not both')

    def _build_stages(self) -> list[Stage]:
        chain: list[Stage] = []
        if self.frame:
            chain.append(FrameStage(encrypted=self.aead is not None or self.passphrase is not None))
        if self.aead is not None:
            from .aead import AEADStage
            chain.append(AEADStage(self.aead, algorithm=self.aead_algorithm))
        elif self.passphrase is not None:
            chain.append(EncryptStage(self.passphrase))
        chain.extend(self.stages)
        return chain

    def capacity_bytes(self, image: np.ndarray) -> int:
        overhead = 14 if self.frame else 0
        if self.aead is not None:
            overhead += 12 + 16
        return max(0, self.carrier.capacity_bits(image) // 8 - overhead)

    def hide(self, image: np.ndarray, data: bytes | str) -> np.ndarray:
        if isinstance(data, str):
            data = data.encode('utf-8')
        blob = data
        for stage in self._build_stages():
            blob = stage.forward(blob)
        return self.carrier.embed(image, blob)

    def reveal(self, image: np.ndarray) -> bytes:
        blob = self.carrier.extract(image)
        for stage in reversed(self._build_stages()):
            blob = stage.inverse(blob)
        return blob

    def reveal_text(self, image: np.ndarray, encoding: str='utf-8') -> str:
        return self.reveal(image).decode(encoding)
