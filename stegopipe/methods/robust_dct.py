from __future__ import annotations
import numpy as np
from .base import Carrier, LENGTH_BITS
from .dct import _dct2, _idct2, BLOCK
from ..bitstream import bytes_to_bits, bits_to_bytes, int_to_bits, bits_to_int
_MIDFREQ = [(4, 1), (3, 2), (2, 3), (1, 4), (4, 2), (2, 4)]
QUANT_ROBUST = [(0, 2)]
ROBUSTNESS_PRESETS = {'max': [(0, 2)], 'balanced': [(2, 1)], 'quality': [(3, 2)]}

class RobustDCT(Carrier):
    name = 'rdct'

    def __init__(self, step: float=40.0, body_rep: int=1, header_rep: int=9, m_coefs: int=1, key: int | None=None, coefs: list | None=None, quant_robust: bool=False, preset: str | None=None):
        if step < 8:
            raise ValueError('step must be >= 8 for 8-bit round-trip stability')
        if body_rep < 1 or header_rep < 1:
            raise ValueError('repetition counts must be >= 1')
        if not 1 <= m_coefs <= len(_MIDFREQ):
            raise ValueError(f'm_coefs must be in 1..{len(_MIDFREQ)}')
        self.step = float(step)
        self.body_rep = body_rep
        self.header_rep = header_rep
        if coefs is not None:
            self.coefs = list(coefs)
        elif preset is not None:
            if preset not in ROBUSTNESS_PRESETS:
                raise ValueError(f'preset must be one of {sorted(ROBUSTNESS_PRESETS)}')
            self.coefs = list(ROBUSTNESS_PRESETS[preset])
        elif quant_robust:
            self.coefs = list(QUANT_ROBUST)
        else:
            self.coefs = _MIDFREQ[:m_coefs]
        self.key = key

    def _order(self, image):
        h, w = image.shape[:2]
        ch = 1 if image.ndim == 2 else image.shape[2]
        blocks = [(c, by, bx) for c in range(ch) for by in range(h // BLOCK) for bx in range(w // BLOCK)]
        if self.key is not None:
            np.random.default_rng(self.key).shuffle(blocks)
        return blocks

    def capacity_bits(self, image):
        nblk = len(self._order(image))
        return (nblk - LENGTH_BITS * self.header_rep) // self.body_rep

    def _get(self, work, c, by, bx):
        ys, xs = (by * BLOCK, bx * BLOCK)
        return work[ys:ys + BLOCK, xs:xs + BLOCK] if work.ndim == 2 else work[ys:ys + BLOCK, xs:xs + BLOCK, c]

    def _set(self, work, c, by, bx, blk):
        ys, xs = (by * BLOCK, bx * BLOCK)
        if work.ndim == 2:
            work[ys:ys + BLOCK, xs:xs + BLOCK] = blk
        else:
            work[ys:ys + BLOCK, xs:xs + BLOCK, c] = blk

    def _embed_bit(self, work, slot, bit):
        c, by, bx = slot
        coef = _dct2(self._get(work, c, by, bx))
        for u, v in self.coefs:
            q = np.round(coef[u, v] / self.step)
            if int(q) % 2 != int(bit):
                q += 1
            coef[u, v] = q * self.step
        self._set(work, c, by, bx, _idct2(coef))

    def _read_bit(self, work, slot):
        c, by, bx = slot
        coef = _dct2(self._get(work, c, by, bx))
        votes = [int(np.round(coef[u, v] / self.step)) % 2 for u, v in self.coefs]
        return 1 if sum(votes) * 2 > len(votes) else 0

    def embed(self, image, blob):
        bits = bytes_to_bits(blob)
        cap = self.capacity_bits(image)
        if bits.size > cap:
            from .base import CarrierError
            raise CarrierError(f'blob needs {bits.size} bits > capacity {cap} (rdct)')
        order = self._order(image)
        work = image.astype(np.float64).copy()
        hdr = np.tile(int_to_bits(len(blob), LENGTH_BITS), self.header_rep)
        seq = np.concatenate([hdr, np.repeat(bits, self.body_rep)])
        for i, bit in enumerate(seq):
            self._embed_bit(work, order[i], bit)
        return np.clip(np.round(work), 0, 255).astype(np.uint8)

    def extract(self, image):
        work = image.astype(np.float64)
        order = self._order(image)
        hr, br = (self.header_rep, self.body_rep)
        raw_hdr = np.array([self._read_bit(work, order[i]) for i in range(LENGTH_BITS * hr)])
        length = bits_to_int((raw_hdr.reshape(hr, LENGTH_BITS).sum(0) * 2 > hr).astype(np.uint8))
        start = LENGTH_BITS * hr
        nbits = length * 8
        n_read = min(nbits * br, len(order) - start)
        nbits = n_read // br
        raw = np.array([self._read_bit(work, order[start + i]) for i in range(nbits * br)])
        if br > 1:
            bits = (raw.reshape(nbits, br).sum(1) * 2 > br).astype(np.uint8)
        else:
            bits = raw.astype(np.uint8)
        return bits_to_bytes(bits[:bits.size // 8 * 8])
