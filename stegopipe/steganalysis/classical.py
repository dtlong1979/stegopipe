from __future__ import annotations
import numpy as np

def to_plane(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return image.astype(np.int64)
    return image[..., min(1, image.shape[2] - 1)].astype(np.int64)

def chi_square_pvalue(image: np.ndarray) -> float:
    plane = to_plane(image)
    hist = np.bincount(plane.ravel(), minlength=256)[:256].astype(np.float64)
    even = hist[0::2]
    odd = hist[1::2]
    expected = (even + odd) / 2.0
    mask = expected > 0
    if mask.sum() < 2:
        return 0.0
    chi = np.sum((even[mask] - expected[mask]) ** 2 / expected[mask])
    k = int(mask.sum()) - 1
    return float(1.0 - _chi2_cdf(chi, k))

def _chi2_cdf(x: float, k: int) -> float:
    if x <= 0:
        return 0.0
    a = k / 2.0
    xx = x / 2.0
    term = 1.0 / a
    total = term
    for n in range(1, 500):
        term *= xx / (a + n)
        total += term
        if term < total * 1e-12:
            break
    from math import exp, lgamma, log
    return float(total * exp(-xx + a * log(xx) - lgamma(a)))

def _f1(x: np.ndarray) -> np.ndarray:
    return x ^ 1

def _f_neg1(x: np.ndarray) -> np.ndarray:
    return (x + 1 ^ 1) - 1

def _discrimination(groups: np.ndarray) -> np.ndarray:
    return np.abs(np.diff(groups, axis=1)).sum(axis=1)

def _rs_counts(plane: np.ndarray, mask: np.ndarray, flip):
    n = mask.size
    flat = plane.ravel()
    usable = flat.size // n * n
    groups = flat[:usable].reshape(-1, n)
    flipped = groups.copy()
    cols = np.nonzero(mask)[0]
    flipped[:, cols] = flip(groups[:, cols])
    f_orig = _discrimination(groups)
    f_flip = _discrimination(flipped)
    regular = int(np.sum(f_flip > f_orig))
    singular = int(np.sum(f_flip < f_orig))
    return (regular, singular)

def rs_analysis(image: np.ndarray, mask=(1, 0, 0, 1)) -> float:
    plane = to_plane(image)
    mask = np.asarray(mask, dtype=np.int64)
    rm, sm = _rs_counts(plane, mask, _f1)
    rmn, smn = _rs_counts(plane, mask, _f_neg1)
    flipped_plane = _f1(plane)
    rm1, sm1 = _rs_counts(flipped_plane, mask, _f1)
    rmn1, smn1 = _rs_counts(flipped_plane, mask, _f_neg1)
    d0, dm0 = (rm - sm, rmn - smn)
    d1, dm1 = (rm1 - sm1, rmn1 - smn1)
    a = 2 * (d1 + d0)
    b = dm0 - dm1 - d1 - 3 * d0
    c = d0 - dm0
    if a == 0:
        x = c / b if b != 0 else 0.0
    else:
        disc = b * b - 4 * a * c
        if disc < 0:
            disc = 0.0
        r = np.sqrt(disc)
        x1, x2 = ((-b + r) / (2 * a), (-b - r) / (2 * a))
        x = x1 if abs(x1) <= abs(x2) else x2
    denom = x - 0.5
    p = 0.0 if denom == 0 else x / denom
    return float(np.clip(p, 0.0, 1.0))

def sample_pair_analysis(image: np.ndarray) -> float:
    plane = to_plane(image)
    u = plane[:, :-1].ravel()
    v = plane[:, 1:].ravel()
    even_v = v % 2 == 0
    x = int(np.sum(even_v & (u < v) | ~even_v & (u > v)))
    y = int(np.sum(even_v & (u > v) | ~even_v & (u < v)))
    same_msb = u >> 1 == v >> 1
    w = int(np.sum(same_msb & (u != v)))
    z = int(np.sum(u == v))
    a = 0.5 * (w + z)
    b = 2 * x - w - y - z
    c = float(y - x)
    if a == 0:
        return float(np.clip(0.0 if b == 0 else -c / b, 0.0, 1.0))
    disc = b * b - 4 * a * c
    if disc < 0:
        disc = 0.0
    root = np.sqrt(disc)
    p1, p2 = ((-b + root) / (2 * a), (-b - root) / (2 * a))
    p = min(abs(p1), abs(p2))
    return float(np.clip(p, 0.0, 1.0))
