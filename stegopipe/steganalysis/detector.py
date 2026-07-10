from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .features import extract_features, extract_features_rich
from ..image_io import make_gradient_image, make_noise_image
from ..methods import get_method

@dataclass
class LogisticRegression:
    lr: float = 0.1
    epochs: int = 400
    l2: float = 0.001
    w: np.ndarray = field(default=None, repr=False)
    b: float = 0.0
    mean_: np.ndarray = field(default=None, repr=False)
    std_: np.ndarray = field(default=None, repr=False)

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LogisticRegression':
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0) + 1e-08
        Xs = (X - self.mean_) / self.std_
        n, d = Xs.shape
        self.w = np.zeros(d)
        self.b = 0.0
        for _ in range(self.epochs):
            p = self._sigmoid(Xs @ self.w + self.b)
            grad_w = Xs.T @ (p - y) / n + self.l2 * self.w
            grad_b = float(np.mean(p - y))
            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        Xs = (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_
        return self._sigmoid(Xs @ self.w + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)

@dataclass
class MLP:
    hidden: int = 24
    lr: float = 0.05
    epochs: int = 600
    l2: float = 0.0001
    seed: int = 0

    def fit(self, X, y):
        rng = np.random.default_rng(self.seed)
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        self.mean_, self.std_ = (X.mean(0), X.std(0) + 1e-08)
        Xs = (X - self.mean_) / self.std_
        n, d = Xs.shape
        self.W1 = rng.normal(0, 0.3, (d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.W2 = rng.normal(0, 0.3, self.hidden)
        self.b2 = 0.0
        for _ in range(self.epochs):
            Z = np.tanh(Xs @ self.W1 + self.b1)
            p = 1.0 / (1.0 + np.exp(-np.clip(Z @ self.W2 + self.b2, -30, 30)))
            g = (p - y) / n
            gW2 = Z.T @ g + self.l2 * self.W2
            gb2 = g.sum()
            gZ = np.outer(g, self.W2) * (1 - Z ** 2)
            self.W1 -= self.lr * (Xs.T @ gZ + self.l2 * self.W1)
            self.b1 -= self.lr * gZ.sum(0)
            self.W2 -= self.lr * gW2
            self.b2 -= self.lr * gb2
        return self

    def predict_proba(self, X):
        Xs = (np.asarray(X, dtype=np.float64) - self.mean_) / self.std_
        Z = np.tanh(Xs @ self.W1 + self.b1)
        return 1.0 / (1.0 + np.exp(-np.clip(Z @ self.W2 + self.b2, -30, 30)))

    def predict(self, X):
        return (self.predict_proba(X) >= 0.5).astype(int)

def _random_cover(rng: np.random.Generator, size: int=128) -> np.ndarray:
    kind = rng.integers(0, 3)
    if kind == 0:
        base = make_gradient_image(size, size, 1).astype(np.float64)
    elif kind == 1:
        yy, xx = np.mgrid[0:size, 0:size]
        freq = rng.uniform(0.05, 0.3)
        base = 128 + 100 * np.sin(freq * xx + rng.uniform(0, 6)) * np.cos(freq * yy + rng.uniform(0, 6))
    else:
        base = make_noise_image(size, size, 1, seed=int(rng.integers(0, 1000000.0))).astype(np.float64)
    base = base + rng.normal(0, 3, base.shape)
    return np.clip(base, 0, 255).astype(np.uint8)

def make_dataset(n: int=120, method: str='lsb', size: int=128, seed: int=0, min_rate: float=0.3, max_rate: float=1.0, feature_fn=extract_features):
    rng = np.random.default_rng(seed)
    carrier = get_method(method)
    X, y = ([], [])
    for _ in range(n):
        cover = _random_cover(rng, size)
        X.append(feature_fn(cover))
        y.append(0)
        stego_cover = _random_cover(rng, size)
        cap = carrier.capacity_bits(stego_cover) // 8
        rate = rng.uniform(min_rate, max_rate)
        nbytes = max(1, int(cap * rate))
        payload = rng.integers(0, 256, size=nbytes, dtype=np.uint8).tobytes()
        stego = carrier.embed(stego_cover, payload)
        X.append(feature_fn(stego))
        y.append(1)
    return (np.array(X), np.array(y))

@dataclass
class SteganalysisDetector:
    model: LogisticRegression = field(default_factory=LogisticRegression)

    def train(self, X, y) -> 'SteganalysisDetector':
        self.model.fit(X, y)
        return self

    def score_image(self, image: np.ndarray) -> float:
        return float(self.model.predict_proba(extract_features(image)[None, :])[0])

    def is_stego(self, image: np.ndarray) -> bool:
        return self.score_image(image) >= 0.5

def evaluate(method: str='lsb', n_train: int=120, n_test: int=60, seed: int=0, feature_fn=extract_features, model=None) -> dict:
    Xtr, ytr = make_dataset(n_train, method=method, seed=seed, feature_fn=feature_fn)
    Xte, yte = make_dataset(n_test, method=method, seed=seed + 999, feature_fn=feature_fn)
    clf = (model if model is not None else LogisticRegression()).fit(Xtr, ytr)
    pred = clf.predict(Xte)
    acc = float(np.mean(pred == yte))
    stego = yte == 1
    recall = float(np.mean(pred[stego] == 1))
    fa = float(np.mean(pred[~stego] == 1))
    return {'method': method, 'accuracy': acc, 'recall': recall, 'false_alarm': fa}
