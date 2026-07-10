import numpy as np
import pytest
from stegopipe.methods import MatrixAdaptive, AdaptiveLSBM, get_method
from stegopipe.image_io import make_noise_image

@pytest.mark.parametrize('k', [1, 2, 3, 4, 5])
def test_roundtrip(k):
    img = make_noise_image(96, 96, 1, seed=k)
    carrier = MatrixAdaptive(k=k)
    blob = b'matrix embedding on adaptive selection!!'
    assert carrier.extract(carrier.embed(img, blob)) == blob

def test_registered():
    assert isinstance(get_method('amx', k=3), MatrixAdaptive)

def test_more_efficient_than_plain_adaptive():
    img = make_noise_image(128, 128, 1, seed=7)
    payload = bytes(range(40))
    plain = AdaptiveLSBM()
    mx = MatrixAdaptive(k=3)
    ch_plain = int(np.sum(img != plain.embed(img, payload)))
    ch_mx = int(np.sum(img != mx.embed(img, payload)))
    assert ch_mx < ch_plain

def test_higher_k_is_more_efficient():
    img = make_noise_image(160, 160, 1, seed=1)
    payload = bytes(range(30))
    ch3 = int(np.sum(img != MatrixAdaptive(k=3).embed(img, payload)))
    ch4 = int(np.sum(img != MatrixAdaptive(k=4).embed(img, payload)))
    assert ch4 <= ch3

def test_capacity_decreases_with_k():
    img = make_noise_image(128, 128, 1)
    caps = [MatrixAdaptive(k=k).capacity_bits(img) for k in (1, 2, 3, 4)]
    assert caps[0] > caps[1] > caps[2] > caps[3]

def test_only_set_A_modified():
    img = make_noise_image(64, 64, 1, seed=2)
    stego = MatrixAdaptive(k=3).embed(img, b'abc')
    ys, xs = np.nonzero(img != stego)
    assert np.all((ys + xs) % 2 == 0)
