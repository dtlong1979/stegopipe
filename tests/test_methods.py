import numpy as np
import pytest
from stegopipe.image_io import make_noise_image, make_gradient_image
from stegopipe.methods import LSB, LSBMatching, DCT, get_method, list_methods, CarrierError

@pytest.fixture
def rgb():
    return make_noise_image(128, 128, 3, seed=1)

@pytest.fixture
def gray():
    return make_noise_image(128, 128, 1, seed=2)
ALL = [LSB(), LSB(bits_per_channel=2), LSB(key=1234), LSBMatching(), LSBMatching(key=99), DCT(step=16.0), DCT(step=24.0, key=7)]

@pytest.mark.parametrize('carrier', ALL)
def test_roundtrip_rgb(carrier, rgb):
    blob = b'pipeline v2 steganography research \x00\x01\x02'
    stego = carrier.embed(rgb, blob)
    assert stego.shape == rgb.shape
    assert stego.dtype == np.uint8
    assert carrier.extract(stego) == blob

@pytest.mark.parametrize('carrier', ALL)
def test_roundtrip_gray(carrier, gray):
    blob = b'grayscale carrier'
    stego = carrier.embed(gray, blob)
    assert carrier.extract(stego) == blob

def test_registry():
    assert set(list_methods()) == {'lsb', 'lsbm', 'dct', 'adaptive', 'amx', 'rdct'}
    assert isinstance(get_method('dct', step=20.0), DCT)
    with pytest.raises(ValueError):
        get_method('nope')

def test_capacity_overflow(gray):
    tiny = make_noise_image(8, 8, 1)
    with pytest.raises(CarrierError):
        LSB().embed(tiny, b'x' * 100)

def test_lsb_changes_only_low_bits(rgb):
    stego = LSB().embed(rgb, b'hello world')
    diff = np.abs(rgb.astype(int) - stego.astype(int))
    assert diff.max() <= 1

def test_dct_capacity_is_low():
    small = make_noise_image(64, 64, 3)
    assert DCT().capacity_bits(small) == 8 * 8 * 3 - 32

def test_lsbm_no_pov_signature():
    img = make_gradient_image(64, 64, 1)
    stego = LSBMatching().embed(img, b'A' * 50)
    assert np.abs(img.astype(int) - stego.astype(int)).max() <= 1
