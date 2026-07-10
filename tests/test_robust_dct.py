import numpy as np
import pytest
from stegopipe.methods import RobustDCT, get_method
from stegopipe.image_io import jpeg_roundtrip, make_gradient_image

def _textured(size=256, seed=1):
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size]
    img = 128.0
    for _ in range(4):
        fx, fy = rng.uniform(0.05, 0.5, 2)
        img += 30 * np.sin(fx * xx + rng.uniform(0, 6)) * np.cos(fy * yy + rng.uniform(0, 6))
    return np.clip(img + rng.normal(0, 8, img.shape), 0, 255).astype(np.uint8)

def test_clean_roundtrip():
    img = _textured()
    r = RobustDCT(step=40, body_rep=1, header_rep=9)
    blob = b'robust payload'
    assert r.extract(r.embed(img, blob)) == blob

def test_registered():
    assert isinstance(get_method('rdct'), RobustDCT)

def test_survives_low_jpeg_quality():
    img = _textured()
    r = RobustDCT(step=56, body_rep=1, header_rep=9)
    payload = b'survive q20'
    stego = r.embed(img, payload)
    for q in (40, 30, 20):
        assert r.extract(jpeg_roundtrip(stego, quality=q)) == payload

def test_larger_step_more_robust_than_baseline():
    img = _textured()
    payload = b'cmp'
    weak = RobustDCT(step=16, body_rep=1, header_rep=9)
    strong = RobustDCT(step=56, body_rep=1, header_rep=9)
    q = 22
    weak_ok = weak.extract(jpeg_roundtrip(weak.embed(img, payload), quality=q)) == payload
    strong_ok = strong.extract(jpeg_roundtrip(strong.embed(img, payload), quality=q)) == payload
    assert strong_ok and (not weak_ok)

def test_capacity_matches_blocks():
    img = _textured(128)
    r = RobustDCT(step=40, body_rep=3, header_rep=9)
    assert r.capacity_bits(img) == (256 - 32 * 9) // 3

def test_quant_robust_survives_lower_quality_than_midfreq():
    img = _textured()
    payload = b'deep jpeg'
    mid = RobustDCT(step=40, body_rep=1, header_rep=9)
    qr = RobustDCT(step=40, body_rep=1, header_rep=9, quant_robust=True)
    q = 14
    mid_ok = mid.extract(jpeg_roundtrip(mid.embed(img, payload), quality=q)) == payload
    qr_ok = qr.extract(jpeg_roundtrip(qr.embed(img, payload), quality=q)) == payload
    assert qr_ok and (not mid_ok)

def test_explicit_coefs():
    img = _textured()
    r = RobustDCT(step=40, coefs=[(0, 2)])
    assert r.coefs == [(0, 2)]
    assert r.extract(r.embed(img, b'x')) == b'x'

def test_robustness_presets():
    from stegopipe.methods.robust_dct import ROBUSTNESS_PRESETS
    img = _textured()
    for name in ROBUSTNESS_PRESETS:
        r = RobustDCT(step=40, preset=name)
        assert r.coefs == ROBUSTNESS_PRESETS[name]
        assert r.extract(r.embed(img, b'hi')) == b'hi'
    payload = b'preset test'
    mx = RobustDCT(step=40, preset='max')
    ql = RobustDCT(step=40, preset='quality')
    q = 14
    assert mx.extract(jpeg_roundtrip(mx.embed(img, payload), quality=q)) == payload
    assert ql.extract(jpeg_roundtrip(ql.embed(img, payload), quality=q)) != payload

def test_invalid_preset():
    import pytest
    with pytest.raises(ValueError):
        RobustDCT(preset='nonsense')
