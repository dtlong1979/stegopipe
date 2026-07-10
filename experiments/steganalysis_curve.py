from __future__ import annotations
import argparse
import glob
import os
import numpy as np
from PIL import Image
from stegopipe.methods import get_method, list_methods
from stegopipe.steganalysis import extract_features_rich, MLP
EXT = ('*.pgm', '*.png', '*.bmp', '*.tif', '*.tiff', '*.ppm')
RATES = (0.1, 0.25, 0.5, 0.9)

def load_paths(d):
    p = []
    for e in EXT:
        p += glob.glob(os.path.join(d, '**', e), recursive=True)
    return sorted(p)

def load_gray(path, size):
    a = np.array(Image.open(path).convert('L'), dtype=np.uint8)
    if size and a.shape[0] >= size and (a.shape[1] >= size):
        y0, x0 = ((a.shape[0] - size) // 2, (a.shape[1] - size) // 2)
        a = a[y0:y0 + size, x0:x0 + size]
    return a

def auc(scores, labels):
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels == 1
    n_pos, n_neg = (int(pos.sum()), int((~pos).sum()))
    if n_pos == 0 or n_neg == 0:
        return float('nan')
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)

def ci95(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else 0.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./data/bossbase')
    ap.add_argument('-n', type=int, default=400)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    paths = load_paths(args.data)
    if not paths:
        print(f'No images in {args.data!r}.')
        return
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(paths), size=min(args.n, len(paths)), replace=False)
    covers = [load_gray(paths[i], args.size or None) for i in sel]
    covers = [c for c in covers if c.ndim == 2 and min(c.shape) >= 64]
    n = len(covers)
    px = covers[0].size
    print(f'E8 multi-rate steganalysis on {n} BOSSBase images (size {args.size}x{args.size}); rich features + MLP; {args.folds} random 50/50 splits; mean ± 95% CI.\n')
    Xcov = np.array([extract_features_rich(c) for c in covers])
    payload_rng = np.random.default_rng(12345)
    print(f"{'carrier':9} | {'rate%cap':>7} | {'bpp':>7} | {'accuracy':>15} | {'AUC':>15}")
    print('-' * 66)
    for name in list_methods():
        carrier = get_method(name)
        for r in RATES:
            Xst, keep = ([], [])
            for i, cov in enumerate(covers):
                cap = carrier.capacity_bits(cov) // 8
                nb = max(1, int(r * cap))
                if nb < 1 or nb > cap:
                    continue
                blob = payload_rng.integers(0, 256, size=nb, dtype=np.uint8).tobytes()
                try:
                    st = carrier.embed(cov, blob)
                except Exception:
                    continue
                Xst.append(extract_features_rich(st))
                keep.append(i)
            if len(Xst) < 20:
                print(f"{name:9} | {int(r * 100):6d}% | {'--':>7} | (capacity too low)")
                continue
            Xst = np.array(Xst)
            keep = np.array(keep)
            bpp = r * carrier.capacity_bits(covers[0]) / px
            X = np.vstack([Xcov[keep], Xst])
            y = np.concatenate([np.zeros(len(keep)), np.ones(len(Xst))]).astype(int)
            accs, aucs = ([], [])
            for f in range(args.folds):
                sp = np.random.default_rng(1000 + f)
                idx = sp.permutation(len(keep))
                tr_c, te_c = (idx[:len(idx) // 2], idx[len(idx) // 2:])
                tr = np.concatenate([tr_c, tr_c + len(keep)])
                te = np.concatenate([te_c, te_c + len(keep)])
                clf = MLP(hidden=32).fit(X[tr], y[tr])
                p = clf.predict_proba(X[te])
                p1 = p[:, 1] if p.ndim == 2 else p
                accs.append(float(np.mean((p1 >= 0.5).astype(int) == y[te])))
                aucs.append(auc(p1, y[te]))
            print(f'{name:9} | {int(r * 100):6d}% | {bpp:7.4f} | {np.mean(accs):6.3f}±{ci95(accs):5.3f} | {np.nanmean(aucs):6.3f}±{ci95(aucs):5.3f}')
        print('-' * 66)
if __name__ == '__main__':
    main()
