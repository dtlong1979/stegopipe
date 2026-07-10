import numpy as np
import pytest
from stegopipe.framing import FrameStage, FrameError
from stegopipe.pipeline import EncryptStage, Pipeline
from stegopipe.codec import ReedSolomon
from stegopipe.image_io import make_noise_image
from stegopipe.methods import get_method, list_methods

def _entropy(b: bytes) -> float:
    hist = np.bincount(np.frombuffer(b, np.uint8), minlength=256).astype(float)
    p = hist[hist > 0] / len(b)
    return float(-(p * np.log2(p)).sum())

def test_frame_inner_detects_wrong_key():
    img = make_noise_image(64, 64, 3, seed=0)
    stego = Pipeline(carrier='lsb', passphrase='right').hide(img, b'secret')
    with pytest.raises(FrameError):
        Pipeline(carrier='lsb', passphrase='wrong').reveal(stego)

def test_frame_outer_ordering_is_silent():
    data = b'attack at dawn'
    k1 = bytes(range(32))
    k2 = bytes(range(1, 33))
    enc1, enc2, frame = (EncryptStage(k1), EncryptStage(k2), FrameStage())
    blob = frame.forward(enc1.forward(data))
    out = enc2.inverse(frame.inverse(blob))
    assert out != data

def test_no_silent_wrong_data_under_tamper():
    rng = np.random.default_rng(1)
    frame = FrameStage()
    silent_wrong = 0
    for _ in range(400):
        data = rng.integers(0, 256, size=int(rng.integers(4, 40)), dtype=np.uint8).tobytes()
        blob = bytearray(frame.forward(data))
        pos = rng.choice(len(blob) * 8, size=int(rng.integers(1, 9)), replace=False)
        for p in pos:
            blob[p // 8] ^= 1 << p % 8
        try:
            if frame.inverse(bytes(blob)) not in (data,):
                silent_wrong += 1
        except FrameError:
            pass
    assert silent_wrong == 0

def test_encryption_flattens_structured_payload():
    img = make_noise_image(96, 96, 1, seed=2)
    structured = b'AAAA' * 150
    plain = Pipeline(carrier='lsb', frame=False)
    enc = Pipeline(carrier='lsb', passphrase='k', frame=False)
    raw_plain = plain.carrier.extract(plain.hide(img, structured))[:len(structured)]
    raw_enc = enc.carrier.extract(enc.hide(img, structured))[:len(structured)]
    assert _entropy(raw_plain) < 3.0
    assert _entropy(raw_enc) > 7.0

@pytest.mark.parametrize('method', list_methods())
def test_composability_every_carrier(method):
    img = make_noise_image(256, 256, 3, seed=3)
    cap = get_method(method).capacity_bits(img) // 8
    payload = bytes(range(min(cap - 300, 20)))
    for pipe in (Pipeline(carrier=get_method(method), frame=False), Pipeline(carrier=get_method(method), frame=True), Pipeline(carrier=get_method(method), passphrase='pw', frame=True), Pipeline(carrier=get_method(method), passphrase='pw', frame=True, stages=[ReedSolomon(16)])):
        assert pipe.reveal(pipe.hide(img, payload)) == payload

def test_crc_frame_is_forgeable_but_aead_is_not():
    import zlib
    from stegopipe.framing import MAGIC, VERSION, HEADER
    forged = b'attacker-controlled payload'
    crc = zlib.crc32(forged) & 4294967295
    forged_frame = HEADER.pack(MAGIC, VERSION, 0, len(forged), crc) + forged
    assert FrameStage().inverse(forged_frame) == forged
    from stegopipe.aead import AEADStage, AEADError, aead_available
    if aead_available():
        s = AEADStage('victim-key')
        blob = bytearray(s.forward(b'genuine'))
        blob[-2] ^= 1
        try:
            s.inverse(bytes(blob))
            assert False, 'AEAD must reject tampering'
        except AEADError:
            pass
