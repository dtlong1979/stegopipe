import numpy as np
import pytest
from stegopipe.steganalysis import chi_square_pvalue, rs_analysis, sample_pair_analysis, extract_features, feature_dim
from stegopipe.steganalysis.detector import make_dataset, SteganalysisDetector
from stegopipe.methods import LSB
from stegopipe.image_io import make_gradient_image

@pytest.fixture
def cover():
    return make_gradient_image(256, 256, 1)

@pytest.fixture
def full_stego(cover):
    rng = np.random.default_rng(0)
    payload = rng.integers(0, 256, size=6000, dtype=np.uint8).tobytes()
    return LSB().embed(cover, payload)

def test_rs_separates_clean_from_stego(cover, full_stego):
    assert rs_analysis(cover) < 0.1
    assert rs_analysis(full_stego) > 0.25

def test_spa_separates_clean_from_stego(cover, full_stego):
    assert sample_pair_analysis(cover) < 0.1
    assert sample_pair_analysis(full_stego) > 0.2

def test_estimates_are_monotonic(cover):
    rng = np.random.default_rng(1)
    payload = rng.integers(0, 256, size=6000, dtype=np.uint8).tobytes()
    half = LSB().embed(cover, payload[:3000])
    full = LSB().embed(cover, payload)
    assert rs_analysis(cover) < rs_analysis(half) < rs_analysis(full)

def test_feature_vector_shape(cover):
    feats = extract_features(cover)
    assert feats.ndim == 1
    assert feats.size == feature_dim()
    assert np.all(np.isfinite(feats))

def test_learned_detector_beats_chance_on_lsb():
    Xtr, ytr = make_dataset(120, method='lsb', seed=2, min_rate=0.8, max_rate=1.0)
    Xte, yte = make_dataset(60, method='lsb', seed=555, min_rate=0.8, max_rate=1.0)
    det = SteganalysisDetector().train(Xtr, ytr)
    acc = float(np.mean(det.model.predict(Xte) == yte))
    assert acc > 0.7

def test_chi_square_pvalue_range(cover):
    assert 0.0 <= chi_square_pvalue(cover) <= 1.0
