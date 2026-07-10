from __future__ import annotations
import numpy as np
from skimage import data
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.metrics import peak_signal_noise_ratio as sk_psnr
from skimage.metrics import structural_similarity as sk_ssim
from stegopipe import Pipeline
from stegopipe.methods import get_method, list_methods
REAL = ['camera', 'coins', 'moon', 'page', 'text', 'clock', 'gravel', 'brick', 'cell', 'astronaut', 'cat', 'coffee', 'rocket']

def _prep(name, size=256):
    img = getattr(data, name)()
    if img.ndim == 3:
        img = rgb2gray(img[..., :3]) * 255
    img = np.asarray(img, dtype=np.float64)
    h, w = img.shape
    if h < size or w < size:
        img = resize(img, (max(size, h), max(size, w)), preserve_range=True, anti_aliasing=True)
        h, w = img.shape
    y0, x0 = ((h - size) // 2, (w - size) // 2)
    return np.clip(img[y0:y0 + size, x0:x0 + size], 0, 255).astype(np.uint8)

def main():
    covers = [(n, _prep(n)) for n in REAL]
    payload = np.random.default_rng(0).integers(0, 256, size=40, dtype=np.uint8).tobytes()
    size = 256
    print(f'E6b · Quality & capacity on {len(covers)} REAL 256x256 grayscale photos')
    print(f'     (scikit-image standard test images; payload 40 B; mean ± std)')
    print(f"     {'carrier':9} | {'PSNR(dB)':>15} | {'SSIM':>16} | {'MSE':>13} | {'cap(bpp)':>8}")
    print('     ' + '-' * 74)
    for name in list_methods():
        carrier = get_method(name)
        P, S, M, C = ([], [], [], [])
        for _, cov in covers:
            if 40 > carrier.capacity_bits(cov) // 8:
                continue
            pipe = Pipeline(carrier=get_method(name), frame=True)
            stego = pipe.hide(cov, payload)
            assert pipe.reveal(stego) == payload
            P.append(sk_psnr(cov, stego, data_range=255))
            S.append(sk_ssim(cov, stego, data_range=255))
            M.append(float(np.mean((cov.astype(float) - stego) ** 2)))
            C.append(carrier.capacity_bits(cov) / (size * size))
        finite = [p for p in P if np.isfinite(p)]
        pm = np.mean(finite) if finite else float('inf')
        print(f'     {name:9} | {pm:8.2f}±{(np.std(finite) if finite else 0):5.2f} | {np.mean(S):9.5f}±{np.std(S):6.5f} | {np.mean(M):7.4f}±{np.std(M):5.4f} | {np.mean(C):8.4f}')
    from stegopipe import metrics as our
    cov = covers[0][1]
    st = Pipeline(carrier='lsb', frame=True).hide(cov, payload)
    print(f"\n     metric cross-check on '{covers[0][0]}' (lsb):")
    print(f'       PSNR: ours {our.psnr(cov, st):.2f} dB vs skimage {sk_psnr(cov, st, data_range=255):.2f} dB')
    print(f'       SSIM: ours {our.ssim(cov, st):.5f} vs skimage {sk_ssim(cov, st, data_range=255):.5f}')
if __name__ == '__main__':
    main()
