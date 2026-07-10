from __future__ import annotations
import hashlib
import os
from dataclasses import dataclass

class AEADError(ValueError):
    pass

def _load_aead(algorithm: str):
    from cryptography.hazmat.primitives.ciphers.aead import AESGCM, ChaCha20Poly1305
    return {'aes-gcm': AESGCM, 'chacha20-poly1305': ChaCha20Poly1305}[algorithm]

def derive_key(passphrase: str | bytes, *, salt: bytes=b'stegopipe-aead-v2', length: int=32) -> bytes:
    if isinstance(passphrase, str):
        passphrase = passphrase.encode('utf-8')
    return hashlib.pbkdf2_hmac('sha256', passphrase, salt, 200000, dklen=length)

@dataclass
class AEADStage:
    key: str | bytes
    algorithm: str = 'aes-gcm'
    aad: bytes = b'stegopipe'
    _NONCE = 12

    def _material(self) -> bytes:
        klen = 32
        if isinstance(self.key, bytes) and len(self.key) == klen:
            return self.key
        return derive_key(self.key, length=klen)

    def forward(self, data: bytes) -> bytes:
        aead = _load_aead(self.algorithm)(self._material())
        nonce = os.urandom(self._NONCE)
        return nonce + aead.encrypt(nonce, data, self.aad)

    def inverse(self, blob: bytes) -> bytes:
        if len(blob) < self._NONCE + 16:
            raise AEADError('ciphertext too short for nonce + tag')
        aead = _load_aead(self.algorithm)(self._material())
        nonce, ct = (blob[:self._NONCE], blob[self._NONCE:])
        try:
            return aead.decrypt(nonce, ct, self.aad)
        except Exception as exc:
            raise AEADError('AEAD verification failed (wrong key or tampering)') from exc

def aead_available() -> bool:
    try:
        _load_aead('aes-gcm')
        return True
    except Exception:
        return False
