from __future__ import annotations
import argparse
import glob
import os
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from stegopipe import Pipeline
from stegopipe.methods import get_method, list_methods
EXT = ('*.pgm', '*.png', '*.bmp', '*.tif', '*.tiff', '*.ppm')

def load_paths(data_dir):
    paths = []
    for e in EXT:
        paths += glob.glob(os.path.join(data_dir, '**', e), recursive=True)
    return sorted(paths)

def load_gray(path, size):
    img = Image.open(path).convert('L')
    a = np.array(img, dtype=np.uint8)
    if size and (a.shape[0] >= size and a.shape[1] >= size):
        y0, x0 = ((a.shape[0] - size) // 2, (a.shape[1] - size) // 2)
        a = a[y0:y0 + size, x0:x0 + size]
    return a

def ci95(x):
    x = np.asarray(x, dtype=float)
    if x.size < 2:
        return 0.0
    return 1.96 * x.std(ddof=1) / np.sqrt(x.size)

def quality_capacity(covers, payload, size):
    print(f'\nQuality & capacity on {len(covers)} real images (payload {len(payload)} B; mean ± 95% CI)')
    print(f"{'carrier':9} | {'PSNR(dB)':>16} | {'SSIM':>18} | {'cap(bpp)':>8} | {'recover':>8}")
    print('-' * 74)
    for name in list_methods():
        carrier = get_method(name)
        P, S, C = ([], [], [])
        attempted = ok = 0
        for cov in covers:
            if len(payload) > carrier.capacity_bits(cov) // 8:
                continue
            attempted += 1
            pipe = Pipeline(carrier=get_method(name), frame=True)
            stego = pipe.hide(cov, payload)
            try:
                recovered = pipe.reveal(stego) == payload
            except Exception:
                recovered = False
            if not recovered:
                continue
            ok += 1
            P.append(sk_psnr(cov, stego, data_range=255))
            S.append(sk_ssim(cov, stego, data_range=255))
            C.append(carrier.capacity_bits(cov) / cov.size)
        if not P:
            print(f'{name:9} | (payload exceeds capacity or 0% recovery, n={attempted})')
            continue
        Pf = [p for p in P if np.isfinite(p)]
        rate = ok / attempted if attempted else 0.0
        print(f'{name:9} | {np.mean(Pf):8.2f}±{ci95(Pf):5.2f} | {np.mean(S):9.5f}±{ci95(S):7.5f} | {np.mean(C):8.4f} | {ok}/{attempted} ({rate:5.1%})')

def steganalysis(covers, payload, seed=0):
    from stegopipe.steganalysis import extract_features_rich, MLP
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(covers))
    tr, te = (idx[:len(idx) // 2], idx[len(idx) // 2:])
    print(f'\nSteganalysis detection accuracy (rich features + MLP), {len(tr)} train / {len(te)} test covers, 0.5 = undetectable')
    print(f"{'carrier':9} | {'accuracy':>8}")
    print('-' * 22)
    for name in list_methods():
        carrier = get_method(name)

        def feats(ids, label):
            X, y = ([], [])
            for i in ids:
                cov = covers[i]
                X.append(extract_features_rich(cov))
                y.append(0)
                if len(payload) <= carrier.capacity_bits(cov) // 8:
                    X.append(extract_features_rich(carrier.embed(cov, payload)))
                    y.append(1)
            return (np.array(X), np.array(y))
        Xtr, ytr = feats(tr, name)
        Xte, yte = feats(te, name)
        if len(set(ytr)) < 2 or len(Xte) == 0:
            print(f'{name:9} | n/a')
            continue
        clf = MLP(hidden=24).fit(Xtr, ytr)
        acc = float(np.mean(clf.predict(Xte) == yte))
        print(f'{name:9} | {acc:>8.3f}')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./data/bossbase')
    ap.add_argument('-n', type=int, default=500)
    ap.add_argument('--size', type=int, default=256, help='centre-crop size (0 = native)')
    ap.add_argument('--payload', type=int, default=40, help='payload bytes')
    ap.add_argument('--steganalysis', action='store_true')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    paths = load_paths(args.data)
    if not paths:
        print(f'No images found in {args.data!r}.\nPut a subset of BOSSBase (PGM/PNG) there, e.g. data/bossbase/, then rerun.\nThe sandbox cannot read your local D: drive — push a subset to the repo\nor upload it via a GitHub Release (GitHub is reachable here).')
        return
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(paths), size=min(args.n, len(paths)), replace=False)
    size = args.size or None
    covers = [load_gray(paths[i], size) for i in sel]
    covers = [c for c in covers if c.ndim == 2 and min(c.shape) >= 64]
    payload = np.random.default_rng(1).integers(0, 256, size=args.payload, dtype=np.uint8).tobytes()
    print(f"Loaded {len(covers)} images from {args.data} (size {('native' if size is None else f'{size}x{size}')}).")
    quality_capacity(covers, payload, size)
    if args.steganalysis:
        steganalysis(covers, payload, args.seed)
if __name__ == '__main__':
    main()
