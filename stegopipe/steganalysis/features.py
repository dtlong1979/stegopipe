from __future__ import annotations
import numpy as np
from .classical import rs_analysis, sample_pair_analysis, to_plane

def _diff_hist(diff: np.ndarray, t: int) -> np.ndarray:
    clipped = np.clip(diff, -t, t)
    hist = np.bincount((clipped + t).ravel(), minlength=2 * t + 1)[:2 * t + 1]
    return hist.astype(np.float64) / max(1, hist.sum())

def _markov(diff: np.ndarray, t: int=1) -> np.ndarray:
    c = np.clip(diff, -t, t) + t
    a, b = (c[:, :-1].ravel(), c[:, 1:].ravel())
    m = 2 * t + 1
    trans = np.zeros((m, m), dtype=np.float64)
    np.add.at(trans, (a, b), 1.0)
    row = trans.sum(axis=1, keepdims=True)
    row[row == 0] = 1.0
    return (trans / row).ravel()

def extract_features(image: np.ndarray, t: int=4) -> np.ndarray:
    plane = to_plane(image)
    dh = np.diff(plane, axis=1)
    dv = np.diff(plane, axis=0)
    feats = [_diff_hist(dh, t), _diff_hist(dv, t), _markov(dh, 1), _markov(dv, 1), np.array([rs_analysis(image), sample_pair_analysis(image)])]
    return np.concatenate(feats)

def _hcf_com(hist: np.ndarray) -> float:
    H = np.abs(np.fft.fft(hist.astype(np.float64)))
    half = len(H) // 2
    k = np.arange(half)
    return float(np.sum(k * H[:half]) / (np.sum(H[:half]) + 1e-09))

def feat_hcf(image: np.ndarray) -> np.ndarray:
    plane = to_plane(image)
    hist = np.bincount(plane.ravel(), minlength=256)[:256]
    com = _hcf_com(hist)
    ds = plane[::2, ::2]
    com_ds = _hcf_com(np.bincount(ds.ravel(), minlength=256)[:256])
    adj = np.bincount((np.clip(np.diff(plane, axis=1), -128, 127) + 128).ravel(), minlength=256)[:256]
    return np.array([com, com_ds, com / (com_ds + 1e-09), _hcf_com(adj)])

def _srm_cooc(residual: np.ndarray, t: int=2) -> np.ndarray:
    rc = np.clip(residual, -t, t) + t
    m = 2 * t + 1
    a, b = (rc[:, :-1].ravel(), rc[:, 1:].ravel())
    mat = np.zeros((m, m))
    np.add.at(mat, (a, b), 1.0)
    return (mat / max(mat.sum(), 1)).ravel()

def feat_srm(image: np.ndarray) -> np.ndarray:
    p = to_plane(image).astype(np.int64)
    res = [p[:, 1:] - p[:, :-1], p[:, :-2] - 2 * p[:, 1:-1] + p[:, 2:], p[:, :-3] - 3 * p[:, 1:-2] + 3 * p[:, 2:-1] - p[:, 3:]]
    return np.concatenate([_srm_cooc(r) for r in res])

def extract_features_rich(image: np.ndarray) -> np.ndarray:
    return np.concatenate([extract_features(image), feat_hcf(image), feat_srm(image)])
FEATURE_DIM = None

def feature_dim(t: int=4) -> int:
    return (2 * t + 1) * 2 + (2 * 1 + 1) ** 2 * 2 + 2
