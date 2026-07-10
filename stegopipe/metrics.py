from __future__ import annotations
import numpy as np

def mse(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    return float(np.mean((a - b) ** 2))

def psnr(a: np.ndarray, b: np.ndarray, peak: float=255.0) -> float:
    error = mse(a, b)
    if error == 0:
        return float('inf')
    return float(10.0 * np.log10(peak ** 2 / error))

def _ssim_gray(a: np.ndarray, b: np.ndarray, peak: float=255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    c1 = (0.01 * peak) ** 2
    c2 = (0.03 * peak) ** 2
    mu_a, mu_b = (a.mean(), b.mean())
    va, vb = (a.var(), b.var())
    cov = np.mean((a - mu_a) * (b - mu_b))
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(num / den)

def ssim(a: np.ndarray, b: np.ndarray, peak: float=255.0) -> float:
    if a.ndim == 2:
        return _ssim_gray(a, b, peak)
    return float(np.mean([_ssim_gray(a[..., c], b[..., c], peak) for c in range(a.shape[-1])]))

def ber(bits_a: np.ndarray, bits_b: np.ndarray) -> float:
    bits_a = np.asarray(bits_a).ravel()
    bits_b = np.asarray(bits_b).ravel()
    if bits_a.size != bits_b.size:
        raise ValueError('bit arrays must be the same length')
    if bits_a.size == 0:
        return 0.0
    return float(np.mean(bits_a != bits_b))

def chi_square_lsb(image: np.ndarray) -> float:
    flat = image.reshape(-1).astype(np.int64)
    hist = np.bincount(flat, minlength=256)[:256]
    even = hist[0::2].astype(np.float64)
    odd = hist[1::2].astype(np.float64)
    pair_total = even + odd
    mask = pair_total > 0
    imbalance = np.zeros_like(pair_total)
    imbalance[mask] = np.abs(even[mask] - odd[mask]) / pair_total[mask]
    return float(imbalance[mask].mean()) if mask.any() else 0.0

def capacity_report(image: np.ndarray, payload_bytes: int, carrier_capacity_bits: int) -> dict:
    used_bits = payload_bytes * 8
    return {'carrier_capacity_bits': carrier_capacity_bits, 'carrier_capacity_bytes': carrier_capacity_bits // 8, 'payload_bytes': payload_bytes, 'payload_bits': used_bits, 'utilisation': used_bits / carrier_capacity_bits if carrier_capacity_bits else 0.0, 'bits_per_pixel': used_bits / (image.shape[0] * image.shape[1])}
