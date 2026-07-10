import numpy as np
import pytest
from stegopipe.bitstream import bits_to_bytes, bytes_to_bits
from stegopipe.codec import Hamming74, RepetitionECC
from stegopipe.image_io import jpeg_roundtrip, make_gradient_image
from stegopipe.methods import DCT, LSB

def _ber(a: bytes, b: bytes) -> float:
    n = min(len(a), len(b))
    if n == 0:
        return 0.5
    return float(np.mean(bytes_to_bits(a[:n]) != bytes_to_bits(b[:n])))

def _bsc(data: bytes, eps: float, seed: int=0) -> bytes:
    bits = bytes_to_bits(data).copy()
    rng = np.random.default_rng(seed)
    bits[rng.random(bits.size) < eps] ^= 1
    return bits_to_bytes(bits)

def test_jpeg_destroys_lsb():
    cover = make_gradient_image(256, 256, 1)
    payload = b'this will not survive jpeg'
    stego = LSB().embed(cover, payload)
    degraded = jpeg_roundtrip(stego, quality=90)
    assert not np.array_equal(stego, degraded)
    try:
        recovered = LSB().extract(degraded)
    except Exception:
        recovered = b''
    assert recovered != payload

def test_dct_survives_jpeg():
    cover = make_gradient_image(256, 256, 1)
    payload = b'robust payload'
    dct = DCT(step=24.0)
    stego = dct.embed(cover, payload)
    for q in (95, 85, 70):
        degraded = jpeg_roundtrip(stego, quality=q)
        assert dct.extract(degraded) == payload

def test_repetition_beats_uncoded_on_bsc():
    payload = bytes(range(200))
    eps = 0.05
    uncoded = _ber(payload, _bsc(payload, eps))
    rep = RepetitionECC(3)
    coded_ber = _ber(payload, rep.inverse(_bsc(rep.forward(payload), eps))[:len(payload)])
    assert coded_ber < uncoded

def test_hamming_corrects_light_noise():
    payload = bytes(range(120))
    eps = 0.01
    ham = Hamming74()
    out = ham.inverse(_bsc(ham.forward(payload), eps))[:len(payload)]
    assert _ber(payload, out) < eps

def test_stronger_repetition_corrects_more():
    payload = bytes(range(150))
    eps = 0.12
    b3 = _ber(payload, RepetitionECC(3).inverse(_bsc(RepetitionECC(3).forward(payload), eps))[:len(payload)])
    b5 = _ber(payload, RepetitionECC(5).inverse(_bsc(RepetitionECC(5).forward(payload), eps))[:len(payload)])
    assert b5 <= b3
