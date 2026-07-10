from __future__ import annotations
import argparse
import glob
import os
import numpy as np
from PIL import Image
from stegopipe import Pipeline
from stegopipe.methods import get_method
from stegopipe.codec import ReedSolomon
EXT = ('*.png', '*.pgm', '*.bmp', '*.tif', '*.tiff', '*.ppm')

def load_paths(d):
    p = []
    for e in EXT:
        p += glob.glob(os.path.join(d, '**', e), recursive=True)
    return sorted(p)

def montages(imgs, n, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        i = rng.integers(0, len(imgs), 4)
        top = np.hstack([imgs[i[0]], imgs[i[1]]])
        bot = np.hstack([imgs[i[2]], imgs[i[3]]])
        out.append(np.vstack([top, bot]).astype(np.uint8))
    return out

def rule_of_three(fail, n):
    return 3.0 / n if fail == 0 else None

def recovery(covers, carrier_name, nsym, payload):
    ok = att = 0
    for cov in covers:
        stages = [ReedSolomon(nsym)] if nsym else []
        pipe = Pipeline(carrier=get_method(carrier_name), frame=True, stages=stages)
        if len(payload) > pipe.capacity_bytes(cov):
            continue
        att += 1
        st = pipe.hide(cov, payload)
        try:
            ok += pipe.reveal(st) == payload
        except Exception:
            pass
    return (ok, att)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./data/bossbase')
    ap.add_argument('-n', type=int, default=250)
    ap.add_argument('--payload', type=int, default=40)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    paths = load_paths(args.data)
    if not paths:
        print(f'No images in {args.data!r}.')
        return
    imgs = [np.array(Image.open(p).convert('L'), np.uint8) for p in paths]
    imgs = [a for a in imgs if a.shape == (256, 256)]
    covers = montages(imgs, args.n, args.seed)
    payload = np.random.default_rng(1).integers(0, 256, size=args.payload, dtype=np.uint8).tobytes()
    print(f'E11 ECC recovery on {len(covers)} 512x512 montage covers (payload {args.payload} B).\n')
    print(f"{'carrier':6} | {'configuration':16} | {'exact recovery':>16}")
    print('-' * 46)
    for carrier in ('dct', 'rdct'):
        for nsym in (0, 16):
            ok, att = recovery(covers, carrier, nsym, payload)
            cfg = 'frame only' if nsym == 0 else f'frame + RS({nsym})'
            rate = ok / att if att else 0.0
            note = ''
            rt = rule_of_three(att - ok, att)
            if rt is not None:
                note = f'  (95% UCB {rt:.1e})'
            print(f'{carrier:6} | {cfg:16} | {ok}/{att} = {rate * 100:5.1f}%{note}')
        print('-' * 46)
if __name__ == '__main__':
    main()
