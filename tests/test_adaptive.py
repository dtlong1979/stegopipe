import numpy as np
import pytest
from stegopipe.methods import AdaptiveLSBM
from stegopipe.image_io import make_noise_image, make_gradient_image

@pytest.mark.parametrize('carrier', [AdaptiveLSBM(), AdaptiveLSBM(key=42)])
def test_roundtrip_rgb(carrier):
    img = make_noise_image(64, 64, 3, seed=3)
    blob = b'adaptive content-aware embedding'
    stego = carrier.embed(img, blob)
    assert carrier.extract(stego) == blob

def test_roundtrip_gray():
    img = make_noise_image(64, 64, 1, seed=4)
    blob = b'gray adaptive'
    stego = AdaptiveLSBM().embed(img, blob)
    assert AdaptiveLSBM().extract(stego) == blob

def test_only_set_A_modified():
    img = make_noise_image(48, 48, 1, seed=5)
    stego = AdaptiveLSBM().embed(img, b'x' * 20)
    diff = img.astype(int) != stego.astype(int)
    ys, xs = np.nonzero(diff)
    assert np.all((ys + xs) % 2 == 0)

def test_capacity_is_half_minus_header():
    img = make_noise_image(64, 64, 1)
    assert AdaptiveLSBM().capacity_bits(img) == 2048 - 32

def test_prefers_textured_regions():
    rng = np.random.default_rng(0)
    img = np.zeros((64, 64), dtype=np.uint8)
    img[:, 32:] = rng.integers(0, 256, size=(64, 32), dtype=np.uint8)
    stego = AdaptiveLSBM().embed(img, b'Q' * 60)
    diff = img.astype(int) != stego.astype(int)
    left_changes = diff[:, :32].sum()
    right_changes = diff[:, 32:].sum()
    assert right_changes > left_changes

def test_adaptive_concentrates_more_than_random():
    from stegopipe.methods import LSBMatching
    rng = np.random.default_rng(0)
    img = np.zeros((96, 96), dtype=np.uint8)
    img[:, 48:] = rng.integers(0, 256, size=(96, 48), dtype=np.uint8)
    payload = b'Z' * 30

    def frac_in_busy(carrier):
        stego = carrier.embed(img, payload)
        assert carrier.extract(stego) == payload
        p = img.astype(float)
        tex = np.abs(np.diff(p, axis=1, prepend=p[:, :1])) + np.abs(np.diff(p, axis=0, prepend=p[:1, :]))
        changed = img != stego
        return (tex[changed] >= np.median(tex)).mean()
    assert frac_in_busy(AdaptiveLSBM()) > frac_in_busy(LSBMatching(key=1))
