import numpy as np
from stegopipe import metrics
from stegopipe.image_io import make_noise_image

def test_psnr_identical_is_inf():
    img = make_noise_image(32, 32, 3)
    assert metrics.psnr(img, img) == float('inf')
    assert metrics.mse(img, img) == 0.0

def test_ssim_identical_is_one():
    img = make_noise_image(32, 32, 3)
    assert abs(metrics.ssim(img, img) - 1.0) < 1e-09

def test_psnr_decreases_with_noise():
    img = make_noise_image(32, 32, 3).astype(np.int16)
    small = np.clip(img + 1, 0, 255).astype(np.uint8)
    big = np.clip(img + 30, 0, 255).astype(np.uint8)
    assert metrics.psnr(img.astype(np.uint8), small) > metrics.psnr(img.astype(np.uint8), big)

def test_ber():
    a = np.array([0, 1, 0, 1])
    b = np.array([0, 1, 1, 1])
    assert metrics.ber(a, b) == 0.25
    assert metrics.ber(a, a) == 0.0

def test_capacity_report():
    img = make_noise_image(32, 32, 3)
    rep = metrics.capacity_report(img, payload_bytes=10, carrier_capacity_bits=1000)
    assert rep['payload_bits'] == 80
    assert rep['utilisation'] == 0.08
    assert rep['carrier_capacity_bytes'] == 125

def test_chi_square_range():
    img = make_noise_image(64, 64, 1)
    score = metrics.chi_square_lsb(img)
    assert 0.0 <= score <= 1.0
