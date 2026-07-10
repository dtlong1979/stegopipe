import numpy as np
import pytest
from stegopipe import Pipeline
from stegopipe.aead import AEADStage, AEADError, aead_available
from stegopipe.image_io import make_noise_image
pytestmark = pytest.mark.skipif(not aead_available(), reason='cryptography AEAD unavailable')

def test_stage_roundtrip():
    s = AEADStage('passphrase')
    for msg in (b'', b'hi', bytes(range(256))):
        assert s.inverse(s.forward(msg)) == msg

def test_wrong_key_rejected():
    blob = AEADStage('right').forward(b'secret')
    with pytest.raises(AEADError):
        AEADStage('wrong').inverse(blob)

def test_tamper_rejected():
    s = AEADStage('k')
    blob = bytearray(s.forward(b'important message'))
    blob[-1] ^= 1
    with pytest.raises(AEADError):
        s.inverse(bytes(blob))

def test_pipeline_aead_roundtrip():
    img = make_noise_image(96, 96, 3, seed=1)
    pipe = Pipeline(carrier='lsb', aead='s3cret')
    stego = pipe.hide(img, 'classified via AEAD')
    assert Pipeline(carrier='lsb', aead='s3cret').reveal_text(stego) == 'classified via AEAD'

def test_pipeline_aead_wrong_key_raises():
    img = make_noise_image(96, 96, 3, seed=2)
    stego = Pipeline(carrier='lsb', aead='right').hide(img, b'msg')
    with pytest.raises(AEADError):
        Pipeline(carrier='lsb', aead='wrong').reveal(stego)

def test_aead_and_passphrase_mutually_exclusive():
    with pytest.raises(ValueError):
        Pipeline(carrier='lsb', aead='a', passphrase='b')

def test_chacha20_backend():
    s = AEADStage('k', algorithm='chacha20-poly1305')
    assert s.inverse(s.forward(b'payload')) == b'payload'
