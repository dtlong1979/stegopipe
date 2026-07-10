# stegopipe

A modular, reproducible framework for image-steganography experiments.

## Files

- `stegopipe/` — framework package
  - `pipeline.py` — stage pipeline (`hide` / `reveal`)
  - `framing.py` — self-describing frame (CRC32)
  - `aead.py` — authenticated encryption (AES-GCM / ChaCha20-Poly1305)
  - `crypto.py` — illustrative stream cipher
  - `bitstream.py` — byte/bit conversion
  - `image_io.py` — image load/save, JPEG round-trip, cover generators
  - `metrics.py` — PSNR, SSIM, MSE, BER, capacity
  - `methods/` — carriers: `lsb`, `lsbm`, `dct`, `adaptive`, `amx`, `rdct`
  - `codec/` — repetition, Hamming(7,4), Reed–Solomon, interleavers
  - `steganalysis/` — RS/SPA/chi-square detectors + rich-feature classifier
  - `cli.py`, `__main__.py` — command-line interface
- `experiments/`
  - `pipeline_eval.py` — E1–E7
  - `real_images_eval.py` — E6b (standard test images)
  - `bossbase_eval.py` — E6c corpus evaluation (quality/capacity/recovery)
  - `steganalysis_curve.py` — E8 rate-normalised steganalysis (accuracy/AUC/CI)
  - `baseline_bench.py` — E9 layer ablation + `stegano` baseline
  - `ecc_recovery_eval.py` — E11 error-correcting recovery (transform carriers)
  - `colab_xunet.ipynb` — E10 CNN steganalysis (Xu-Net), GPU notebook
  - `bossbase_subset1.zip`, `bossbase_subset2.zip` — 1000-image BOSSBase subset (256×256)
- `tests/` — test suite
- `results/` — captured experiment outputs
- `requirements.txt` — pinned dependency versions
- `LICENSE` — MIT

## Run

```
pip install -r requirements.txt
mkdir -p data/bossbase/s1 data/bossbase/s2
unzip -j experiments/bossbase_subset1.zip -d data/bossbase/s1
unzip -j experiments/bossbase_subset2.zip -d data/bossbase/s2
PYTHONPATH=. python experiments/pipeline_eval.py
PYTHONPATH=. python experiments/real_images_eval.py
PYTHONPATH=. python experiments/bossbase_eval.py --data data/bossbase -n 1000 --size 256
PYTHONPATH=. python experiments/steganalysis_curve.py --data data/bossbase -n 400
PYTHONPATH=. python experiments/baseline_bench.py --data data/bossbase -n 150
PYTHONPATH=. python experiments/ecc_recovery_eval.py --data data/bossbase -n 250
python -m pytest -q
```

E10 (Xu-Net CNN) runs on a GPU via `experiments/colab_xunet.ipynb`.

Python 3.11.
