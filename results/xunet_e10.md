E10 — Xu-Net CNN steganalysis on BOSSBase (Colab T4 GPU, PyTorch)
Content-disjoint 60/20/20 split, flip/rot augmentation, 30 epochs.

| carrier | bpp | Xu-Net acc | Xu-Net AUC | MLP AUC (E8, nearest rate) |
|---------|-----|-----------|-----------|----------------------------|
| lsbm     | 0.40 | 1.000 | 1.000 | 0.992 (0.50 bpp) |
| adaptive | 0.20 | 0.988 | 1.000 | ~0.90 (0.12-0.25 bpp) |
| amx      | 0.10 | 0.890 | 0.974 | 0.791 (0.107 bpp) |

Learned residual features detect at lower embedding rates than the E8 rich-feature
MLP, confirming the E8 curve is a lower bound on detectability.
