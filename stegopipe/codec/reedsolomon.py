from __future__ import annotations
PRIM = 285
_EXP = [0] * 512
_LOG = [0] * 256
_x = 1
for _i in range(255):
    _EXP[_i] = _x
    _LOG[_x] = _i
    _nx = _x << 1
    _x = _nx ^ PRIM if _nx & 256 else _nx
for _i in range(255, 512):
    _EXP[_i] = _EXP[_i - 255]

def _mul(a, b):
    return 0 if a == 0 or b == 0 else _EXP[_LOG[a] + _LOG[b]]

def _div(a, b):
    if b == 0:
        raise ZeroDivisionError
    return 0 if a == 0 else _EXP[(_LOG[a] + 255 - _LOG[b]) % 255]

def _pow(a, p):
    return _EXP[_LOG[a] * p % 255]

def _inv(a):
    return _EXP[(255 - _LOG[a]) % 255]

def _poly_scale(p, s):
    return [_mul(c, s) for c in p]

def _poly_add(p, q):
    r = [0] * max(len(p), len(q))
    for i in range(len(p)):
        r[i + len(r) - len(p)] = p[i]
    for i in range(len(q)):
        r[i + len(r) - len(q)] ^= q[i]
    return r

def _poly_mul(p, q):
    r = [0] * (len(p) + len(q) - 1)
    for j in range(len(q)):
        for i in range(len(p)):
            r[i + j] ^= _mul(p[i], q[j])
    return r

def _poly_eval(p, x):
    y = p[0]
    for c in p[1:]:
        y = _mul(y, x) ^ c
    return y

def _generator(nsym):
    g = [1]
    for i in range(nsym):
        g = _poly_mul(g, [1, _pow(2, i)])
    return g

def rs_encode_block(msg, nsym):
    gen = _generator(nsym)
    out = [0] * (len(msg) + len(gen) - 1)
    out[:len(msg)] = msg
    for i in range(len(msg)):
        coef = out[i]
        if coef != 0:
            for j in range(1, len(gen)):
                out[i + j] ^= _mul(gen[j], coef)
    out[:len(msg)] = msg
    return out

def _syndromes(msg, nsym):
    return [0] + [_poly_eval(msg, _pow(2, i)) for i in range(nsym)]

def _error_locator(synd, nsym):
    err_loc, old_loc = ([1], [1])
    for i in range(nsym):
        delta = synd[i + 1]
        for j in range(1, len(err_loc)):
            delta ^= _mul(err_loc[-(j + 1)], synd[i + 1 - j])
        old_loc = old_loc + [0]
        if delta != 0:
            if len(old_loc) > len(err_loc):
                new_loc = _poly_scale(old_loc, delta)
                old_loc = _poly_scale(err_loc, _inv(delta))
                err_loc = new_loc
            err_loc = _poly_add(err_loc, _poly_scale(old_loc, delta))
    while err_loc and err_loc[0] == 0:
        err_loc.pop(0)
    return err_loc

def _find_errors(err_loc, nmess):
    errs = len(err_loc) - 1
    pos = []
    for i in range(255):
        if _poly_eval(err_loc, _pow(2, i)) == 0:
            deg = (255 - i) % 255
            p = nmess - 1 - deg
            if 0 <= p < nmess:
                pos.append(p)
    return pos if len(pos) == errs else None

def _errata_locator(e_pos):
    e_loc = [1]
    for i in e_pos:
        e_loc = _poly_mul(e_loc, _poly_add([1], [_pow(2, i), 0]))
    return e_loc

def _error_evaluator(synd, err_loc, nsym):
    rem = _poly_mul(synd, err_loc)
    return rem[len(rem) - (nsym + 1):]

def rs_correct_block(msg_in, nsym):
    msg = list(msg_in)
    synd = _syndromes(msg, nsym)
    if max(synd) == 0:
        return msg
    err_loc = _error_locator(synd, nsym)
    err_pos = _find_errors(err_loc, len(msg))
    if err_pos is None:
        raise ValueError('RS: cannot locate errors (too many)')
    coef_pos = [len(msg) - 1 - p for p in err_pos]
    errata = _errata_locator(coef_pos)
    err_eval = _error_evaluator(synd[::-1], errata, len(errata) - 1)[::-1]
    X = [_pow(2, p) for p in coef_pos]
    for i, Xi in enumerate(X):
        Xi_inv = _inv(Xi)
        denom = 1
        for j in range(len(X)):
            if j != i:
                denom = _mul(denom, 1 ^ _mul(Xi_inv, X[j]))
        y = _mul(Xi, _poly_eval(err_eval[::-1], Xi_inv))
        msg[err_pos[i]] ^= _div(y, denom)
    return msg

class ReedSolomon:

    def __init__(self, nsym: int=32):
        if not 2 <= nsym <= 254 or nsym % 2 != 0:
            raise ValueError('nsym must be an even integer in 2..254')
        self.nsym = nsym
        self.k = 255 - nsym

    @property
    def t(self) -> int:
        return self.nsym // 2

    def forward(self, data: bytes) -> bytes:
        payload = len(data).to_bytes(4, 'big') + data
        out = bytearray()
        for i in range(0, len(payload), self.k):
            block = list(payload[i:i + self.k])
            if len(block) < self.k:
                block += [0] * (self.k - len(block))
            out += bytes(rs_encode_block(block, self.nsym))
        return bytes(out)

    def inverse(self, data: bytes) -> bytes:
        out = bytearray()
        n = self.k + self.nsym
        for i in range(0, len(data), n):
            block = list(data[i:i + n])
            if len(block) < n:
                break
            try:
                fixed = rs_correct_block(block, self.nsym)
            except Exception:
                fixed = block
            out += bytes(fixed[:self.k])
        if len(out) < 4:
            return b''
        length = int.from_bytes(bytes(out[:4]), 'big')
        return bytes(out[4:4 + length])
