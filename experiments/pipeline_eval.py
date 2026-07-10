from __future__ import annotations
import math
import platform
import sys
import zlib
import numpy as np
from stegopipe import metrics
from stegopipe.framing import FrameStage, FrameError, MAGIC, VERSION, HEADER
from stegopipe.pipeline import EncryptStage, Pipeline
from stegopipe.aead import AEADStage, AEADError, aead_available
from stegopipe.codec import ReedSolomon
from stegopipe.image_io import make_noise_image, make_gradient_image
from stegopipe.methods import get_method, list_methods

def e1_wrong_key_detection(trials=2000, seed=0):
    rng = np.random.default_rng(seed)
    detected = {'frame-inner (ours)': 0, 'frame-outer': 0, 'no-frame': 0}
    for _ in range(trials):
        data = rng.integers(0, 256, size=int(rng.integers(1, 40)), dtype=np.uint8).tobytes()
        k1 = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
        k2 = rng.integers(0, 256, size=32, dtype=np.uint8).tobytes()
        if k1 == k2:
            continue
        enc1, enc2 = (EncryptStage(k1), EncryptStage(k2))
        frame = FrameStage(encrypted=True)
        blob = enc1.forward(frame.forward(data))
        try:
            frame.inverse(enc2.inverse(blob))
        except FrameError:
            detected['frame-inner (ours)'] += 1
        blob = frame.forward(enc1.forward(data))
        try:
            out = enc2.inverse(frame.inverse(blob))
            if out == data:
                pass
        except FrameError:
            detected['frame-outer'] += 1
        blob = enc1.forward(data)
        try:
            _ = enc2.inverse(blob)
        except Exception:
            detected['no-frame'] += 1
    return {k: v / trials for k, v in detected.items()}

def e2_tamper_detection(trials=3000, seed=1):
    rng = np.random.default_rng(seed)
    frame = FrameStage()
    caught = correct = silent_wrong = 0
    for _ in range(trials):
        data = rng.integers(0, 256, size=int(rng.integers(4, 60)), dtype=np.uint8).tobytes()
        blob = bytearray(frame.forward(data))
        original = bytes(blob)
        nflips = int(rng.integers(1, 9))
        positions = rng.choice(len(blob) * 8, size=nflips, replace=False)
        for pos in positions:
            blob[pos // 8] ^= 1 << pos % 8
        if bytes(blob) == original:
            continue
        try:
            out = frame.inverse(bytes(blob))
            if out == data:
                correct += 1
            else:
                silent_wrong += 1
        except FrameError:
            caught += 1
    n = caught + correct + silent_wrong
    return (caught / n, correct / n, silent_wrong / n)

def _entropy_bits_per_byte(b: bytes) -> float:
    if not b:
        return 0.0
    hist = np.bincount(np.frombuffer(b, np.uint8), minlength=256).astype(float)
    p = hist[hist > 0] / len(b)
    return float(-(p * np.log2(p)).sum())

def e3_flatten(seed=2):
    img = make_noise_image(128, 128, 1, seed=seed)
    structured = b'AAAA' * 200 + bytes(range(64)) * 3
    lsb = get_method('lsb')
    plain = Pipeline(carrier=lsb, frame=False)
    enc = Pipeline(carrier=lsb, passphrase='k', frame=False)
    raw_plain = plain.carrier.extract(plain.hide(img, structured))[:len(structured)]
    raw_enc = enc.carrier.extract(enc.hide(img, structured))[:len(structured)]
    return {'payload_entropy': _entropy_bits_per_byte(structured), 'embedded_plain_entropy': _entropy_bits_per_byte(raw_plain), 'embedded_encrypted_entropy': _entropy_bits_per_byte(raw_enc)}

def e4_composability(seed=3):
    rng = np.random.default_rng(seed)
    results = {}
    for name in list_methods():
        img = make_noise_image(512, 512, 3, seed=seed)
        carrier = get_method(name)
        cap = carrier.capacity_bits(img) // 8
        payload = rng.integers(0, 256, size=min(cap - 300, 24), dtype=np.uint8).tobytes()
        row = {}
        crypto = {'aead': 'pw'} if aead_available() else {'passphrase': 'pw'}
        configs = {'plain': Pipeline(carrier=get_method(name), frame=False), '+frame': Pipeline(carrier=get_method(name), frame=True), '+AEAD+frame': Pipeline(carrier=get_method(name), frame=True, **crypto), '+RS+AEAD+frame': Pipeline(carrier=get_method(name), frame=True, stages=[ReedSolomon(16)], **crypto)}
        for cfg, pipe in configs.items():
            try:
                row[cfg] = pipe.reveal(pipe.hide(img, payload)) == payload
            except Exception as exc:
                row[cfg] = f'ERR:{type(exc).__name__}'
        results[name] = row
    return results

def _covers(n, size=256, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        kind = i % 3
        if kind == 0:
            out.append(make_gradient_image(size, size, 1))
        elif kind == 1:
            out.append(make_noise_image(size, size, 1, seed=seed + i))
        else:
            yy, xx = np.mgrid[0:size, 0:size]
            f = rng.uniform(0.05, 0.3)
            img = 128 + 90 * np.sin(f * xx) * np.cos(f * yy) + rng.normal(0, 4, (size, size))
            out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out

def e6_quality_capacity(n_covers=12, size=256, payload_bytes=40, seed=7):
    covers = _covers(n_covers, size, seed)
    payload = np.random.default_rng(0).integers(0, 256, size=payload_bytes, dtype=np.uint8).tobytes()
    rows = {}
    for name in list_methods():
        carrier = get_method(name)
        psnrs, ssims, mses, caps = ([], [], [], [])
        for cov in covers:
            if payload_bytes > carrier.capacity_bits(cov) // 8:
                continue
            pipe = Pipeline(carrier=get_method(name), frame=True)
            stego = pipe.hide(cov, payload)
            assert pipe.reveal(stego) == payload
            psnrs.append(metrics.psnr(cov, stego))
            ssims.append(metrics.ssim(cov, stego))
            mses.append(metrics.mse(cov, stego))
            caps.append(carrier.capacity_bits(cov) / (size * size))
        if psnrs:
            finite = [p for p in psnrs if np.isfinite(p)]
            rows[name] = dict(psnr=(np.mean(finite) if finite else float('inf'), np.std(finite) if finite else 0.0), ssim=(np.mean(ssims), np.std(ssims)), mse=(np.mean(mses), np.std(mses)), cap_bpp=np.mean(caps))
    bpp = payload_bytes * 8 / (size * size)
    return (rows, bpp, n_covers, size, payload_bytes)

def e7_forgery(trials=1000, seed=11):
    rng = np.random.default_rng(seed)
    frame = FrameStage()
    crc_forgery_accepted = 0
    aead_forgery_rejected = 0
    have_aead = aead_available()
    for _ in range(trials):
        original = rng.integers(0, 256, size=int(rng.integers(4, 40)), dtype=np.uint8).tobytes()
        forged = rng.integers(0, 256, size=int(rng.integers(4, 40)), dtype=np.uint8).tobytes()
        crc = zlib.crc32(forged) & 4294967295
        forged_frame = HEADER.pack(MAGIC, VERSION, 0, len(forged), crc) + forged
        try:
            if frame.inverse(forged_frame) == forged:
                crc_forgery_accepted += 1
        except FrameError:
            pass
        if have_aead:
            victim = AEADStage('victim-key')
            blob = bytearray(victim.forward(original))
            for i in rng.choice(len(blob), size=min(3, len(blob)), replace=False):
                blob[i] ^= 1
            try:
                victim.inverse(bytes(blob))
            except AEADError:
                aead_forgery_rejected += 1
    return (crc_forgery_accepted / trials, aead_forgery_rejected / trials if have_aead else None)

def _env():
    import numpy, PIL
    return f"Python {platform.python_version()} on {platform.system()} {platform.machine()}; numpy {numpy.__version__}; Pillow {PIL.__version__}; cryptography AEAD {('available' if aead_available() else 'ABSENT')}"

def main():
    print('Environment:', _env())
    print('=' * 68)
    print('E1  Wrong-key detection rate (2000 trials)')
    for k, v in e1_wrong_key_detection().items():
        print(f'     {k:22} : {v:6.1%} detected')
    print('     -> frame-inner (ours) turns a wrong key into a hard error;')
    print('        frame-outer / no-frame return silent garbage.')
    print('=' * 68)
    caught, correct, silent = e2_tamper_detection()
    print('E2  Tamper (1-8 distinct bit flips), outcome rates')
    print(f'     caught by CRC/magic (FrameError)  : {caught:.2%}')
    print(f'     harmless (flip hit flags only)    : {correct:.2%}')
    print(f'     SILENT WRONG DATA accepted        : {silent:.4%}  <- must be ~0')
    print(f'     theoretical CRC32 false-negative  : 2^-32 = {2 ** (-32):.2e}')
    print('=' * 68)
    print('E3  Entropy (bits/byte, 8.0 = uniform) of the embedded stream')
    for k, v in e3_flatten().items():
        print(f'     {k:28} : {v:.3f}')
    print('     -> a structured payload lands flat once encrypted.')
    print('=' * 68)
    print('E4  Composability: every carrier x every stage combo recovers')
    res = e4_composability()
    cfgs = list(next(iter(res.values())).keys())
    print(f"     {'carrier':9} | " + ' | '.join((c[:14] for c in cfgs)))
    for name, row in res.items():
        marks = ' | '.join((('  ok ' if row[c] is True else str(row[c]))[:14].center(14) for c in cfgs))
        print(f'     {name:9} | {marks}')
    print('=' * 68)
    print('E5  Overhead: frame header = 14 bytes/payload (magic4+ver1+flags1+len4+crc4); AEAD adds 12-byte nonce + 16-byte tag.')
    print('=' * 68)
    rows, bpp, ncov, size, plen = e6_quality_capacity()
    print(f'E6  Image quality & capacity through the pipeline')
    print(f'     {ncov} covers {size}x{size} gray, payload {plen} B ({bpp:.4f} bpp), mean±std')
    print(f"     {'carrier':9} | {'PSNR(dB)':>14} | {'SSIM':>15} | {'MSE':>13} | {'cap(bpp)':>8}")
    print('     ' + '-' * 70)
    for name, r in rows.items():
        pm, ps = r['psnr']
        sm, ss = r['ssim']
        mm, ms = r['mse']
        print(f"     {name:9} | {pm:8.2f}±{ps:4.2f} | {sm:8.5f}±{ss:6.5f} | {mm:7.4f}±{ms:5.4f} | {r['cap_bpp']:8.4f}")
    print('=' * 68)
    crc_acc, aead_rej = e7_forgery()
    print('E7  CRC32 is error-detection, NOT authentication (1000 forgery trials)')
    print(f"     CRC frame accepts an attacker's forged payload : {crc_acc:.1%}")
    if aead_rej is not None:
        print(f'     AEAD rejects forged/tampered ciphertext        : {aead_rej:.1%}')
    print('     -> an unkeyed CRC is trivially recomputed by an adversary; only a')
    print('        keyed AEAD tag provides integrity against active tampering.')
if __name__ == '__main__':
    main()
