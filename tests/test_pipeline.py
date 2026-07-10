import numpy as np
import pytest
from stegopipe import Pipeline
from stegopipe.framing import FrameError
from stegopipe.image_io import make_noise_image

@pytest.fixture
def img():
    return make_noise_image(96, 96, 3, seed=5)

def test_plain_roundtrip(img):
    pipe = Pipeline(carrier='lsb')
    stego = pipe.hide(img, 'hello pipeline v2')
    assert pipe.reveal_text(stego) == 'hello pipeline v2'

def test_encrypted_roundtrip(img):
    pipe = Pipeline(carrier='lsb', passphrase='s3cret')
    stego = pipe.hide(img, 'classified')
    assert Pipeline(carrier='lsb', passphrase='s3cret').reveal_text(stego) == 'classified'

def test_wrong_passphrase_raises(img):
    stego = Pipeline(carrier='lsb', passphrase='right').hide(img, 'msg')
    with pytest.raises(FrameError):
        Pipeline(carrier='lsb', passphrase='wrong').reveal(stego)

def test_bytes_payload(img):
    pipe = Pipeline(carrier='lsbm')
    payload = bytes(range(256))
    stego = pipe.hide(img, payload)
    assert pipe.reveal(stego) == payload

def test_capacity_bytes_positive(img):
    assert Pipeline(carrier='lsb').capacity_bytes(img) > 0
    assert Pipeline(carrier='dct').capacity_bytes(img) >= 0

def test_dct_pipeline_roundtrip(img):
    pipe = Pipeline(carrier='dct', passphrase='key')
    stego = pipe.hide(img, 'robust')
    assert Pipeline(carrier='dct', passphrase='key').reveal_text(stego) == 'robust'

def test_encryption_flattens_payload(img):
    plain = Pipeline(carrier='lsb', frame=False)
    enc = Pipeline(carrier='lsb', passphrase='k', frame=False)
    msg = 'A' * 200
    s_plain = plain.hide(img, msg)
    s_enc = enc.hide(img, msg)
    assert not np.array_equal(s_plain, s_enc)
