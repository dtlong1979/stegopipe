from __future__ import annotations
import hashlib
import struct

def _keystream(key: bytes, nbytes: int, nonce: bytes=b'') -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < nbytes:
        block = hashlib.sha256(key + nonce + struct.pack('>Q', counter)).digest()
        out.extend(block)
        counter += 1
    return bytes(out[:nbytes])

def derive_key(passphrase: str | bytes, *, salt: bytes=b'stegopipe-v2') -> bytes:
    if isinstance(passphrase, str):
        passphrase = passphrase.encode('utf-8')
    return hashlib.pbkdf2_hmac('sha256', passphrase, salt, 100000, dklen=32)

def encrypt(data: bytes, key: bytes, nonce: bytes=b'') -> bytes:
    ks = _keystream(key, len(data), nonce)
    return bytes((b ^ k for b, k in zip(data, ks)))
decrypt = encrypt
