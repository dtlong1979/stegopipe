import numpy as np
from stegopipe.steganalysis import extract_features_rich, feat_hcf, feat_srm, evaluate, MLP, extract_features
from stegopipe.image_io import make_gradient_image

def test_rich_feature_shapes():
    img = make_gradient_image(128, 128, 1)
    assert feat_hcf(img).shape == (4,)
    assert np.all(np.isfinite(feat_hcf(img)))
    srm = feat_srm(img)
    assert srm.shape == (3 * 25,)
    rich = extract_features_rich(img)
    assert rich.size == extract_features(img).size + 4 + 75
    assert np.all(np.isfinite(rich))

def test_rich_features_improve_lsbm_detection():
    base = evaluate(method='lsbm', n_train=90, n_test=50, seed=3)
    rich = evaluate(method='lsbm', n_train=90, n_test=50, seed=3, feature_fn=extract_features_rich, model=MLP(hidden=24, seed=0))
    assert rich['accuracy'] >= base['accuracy']
    assert rich['accuracy'] > 0.55

def test_mlp_trains_and_predicts():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 8))
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    clf = MLP(hidden=8, epochs=300).fit(X, y)
    assert float(np.mean(clf.predict(X) == y)) > 0.8
