from __future__ import annotations
import argparse
import glob
import os
import time
import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from stegopipe import Pipeline
from stegopipe.methods import get_method
EXT = ('*.pgm', '*.png', '*.bmp', '*.tif', '*.tiff', '*.ppm')

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

def mean_ci(x):
    x = np.asarray(x, float)
    m = x.mean()
    ci = 1.96 * x.std(ddof=1) / np.sqrt(x.size) if x.size > 1 else 0.0
    return (m, ci)

def bench_pipe(covers, payload, name, frame, aead):
    key = b'benchmark-key-0123456789abcdef01'
    P, S, Th, Te, ok, tamper_caught = ([], [], [], [], 0, 0)
    embedded = None
    for cov in covers:
        kw = dict(carrier=get_method('lsb'), frame=frame)
        if aead:
            kw.update(aead=key, aead_algorithm='aes-gcm')
        pipe = Pipeline(**kw)
        t = time.perf_counter()
        st = pipe.hide(cov, payload)
        Th.append(time.perf_counter() - t)
        if embedded is None:
            blob = payload
            for stage in pipe._build_stages():
                blob = stage.forward(blob)
            embedded = len(blob)
        t = time.perf_counter()
        try:
            rec = pipe.reveal(st)
        except Exception:
            rec = None
        Te.append(time.perf_counter() - t)
        if rec == payload:
            ok += 1
            P.append(sk_psnr(cov, st, data_range=255))
            S.append(sk_ssim(cov, st, data_range=255))
        st2 = st.copy().reshape(-1)
        st2[0] ^= 1
        st2 = st2.reshape(st.shape)
        try:
            r2 = pipe.reveal(st2)
            if r2 != payload:
                tamper_caught += 1
        except Exception:
            tamper_caught += 1
    return dict(name=name, ok=ok, n=len(covers), psnr=mean_ci(P), ssim=mean_ci(S), th=mean_ci(Th), te=mean_ci(Te), embedded=embedded, tamper=tamper_caught / len(covers), authed=aead)

def bench_stegano(covers, payload_text, size):
    from stegano import lsb as slsb
    P, S, Th, Te, ok, tamper_wrong = ([], [], [], [], 0, 0)
    tmp_c, tmp_s = ('/tmp/_bc.png', '/tmp/_bs.png')
    for cov in covers:
        rgb = np.stack([cov, cov, cov], axis=-1)
        Image.fromarray(rgb).save(tmp_c)
        t = time.perf_counter()
        st_img = slsb.hide(tmp_c, payload_text)
        Th.append(time.perf_counter() - t)
        st_img.save(tmp_s)
        t = time.perf_counter()
        try:
            rec = slsb.reveal(tmp_s)
        except Exception:
            rec = None
        Te.append(time.perf_counter() - t)
        st = np.array(Image.open(tmp_s))
        if rec == payload_text:
            ok += 1
            P.append(sk_psnr(rgb, st, data_range=255))
            S.append(sk_ssim(rgb, st, data_range=255, channel_axis=-1))
        st2 = st.copy()
        st2[0, 0, 0] ^= 1
        Image.fromarray(st2).save(tmp_s)
        try:
            r2 = slsb.reveal(tmp_s)
            if r2 != payload_text:
                tamper_wrong += 1
        except Exception:
            tamper_wrong += 1
    cap_bpp = covers[0].size * 3 / covers[0].size
    return dict(name='stegano (lsb)', ok=ok, n=len(covers), psnr=mean_ci(P), ssim=mean_ci(S), th=mean_ci(Th), te=mean_ci(Te), cap=cap_bpp, tamper=None, tamper_wrong=tamper_wrong / len(covers))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='./data/bossbase')
    ap.add_argument('-n', type=int, default=100)
    ap.add_argument('--size', type=int, default=256)
    ap.add_argument('--payload', type=int, default=100)
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
    payload = np.random.default_rng(1).integers(0, 256, size=args.payload, dtype=np.uint8).tobytes()
    payload_text = ''.join((chr(65 + b % 26) for b in payload))
    print(f'E9 ablation benchmark on {len(covers)} BOSSBase images (size {args.size}x{args.size}, payload {args.payload} B).\n')
    rows = [bench_pipe(covers, payload, 'LSB only', frame=False, aead=False), bench_pipe(covers, payload, 'LSB + frame', frame=True, aead=False), bench_pipe(covers, payload, 'LSB + frame + AEAD', frame=True, aead=True)]
    steg = bench_stegano(covers, payload_text, args.size)
    print(f"{'configuration':22} | {'recover':>8} | {'embed B':>7} | {'PSNR(dB)':>13} | {'SSIM':>10} | {'embed ms':>11} | {'extract ms':>11} | {'auth?':>6}")
    print('-' * 106)
    for r in rows:
        pm, pc = r['psnr']
        sm, sc = r['ssim']
        hm, hc = r['th']
        em, ec = r['te']
        auth = 'yes' if r['authed'] else 'no'
        print(f"{r['name']:22} | {r['ok']}/{r['n']:>3} | {r['embedded']:>7} | {pm:6.2f}±{pc:4.2f} | {sm:7.5f} | {hm * 1000:7.2f}±{hc * 1000:4.2f} | {em * 1000:7.2f}±{ec * 1000:4.2f} | {auth:>6}")
    pm, pc = steg['psnr']
    sm, sc = steg['ssim']
    hm, hc = steg['th']
    em, ec = steg['te']
    print(f"{'stegano (LSB, RGB)':22} | {steg['ok']}/{steg['n']:>3} | {args.payload:>7} | {pm:6.2f}±{pc:4.2f} | {sm:7.5f} | {hm * 1000:7.2f}±{hc * 1000:4.2f} | {em * 1000:7.2f}±{ec * 1000:4.2f} | {'no':>6}")
    print()
    print('Authentication under a one-bit tamper of the stego image:')
    print(f"  LSB + frame + AEAD : rejected in {rows[2]['tamper']:.1%} of trials (keyed tag).")
    print(f'  LSB + frame (CRC)  : the unkeyed CRC is recomputable; it is not authentication.')
    print(f"  stegano (LSB)      : no authentication — wrong/garbled output in {steg['tamper_wrong']:.1%} of trials.")
if __name__ == '__main__':
    main()
