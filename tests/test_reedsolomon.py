import numpy as np
import pytest
from stegopipe.codec import ByteInterleaver, ReedSolomon, SpreadRepetition
from stegopipe.codec.reedsolomon import rs_correct_block, rs_encode_block
from stegopipe.bitstream import bits_to_bytes, bytes_to_bits

def test_rs_block_corrects_up_to_t():
    rng = np.random.default_rng(0)
    nsym = 20
    for _ in range(50):
        msg = list(rng.integers(0, 256, size=40))
        code = rs_encode_block(msg, nsym)
        ne = int(rng.integers(0, nsym // 2 + 1))
        idx = rng.choice(len(code), size=ne, replace=False)
        for i in idx:
            code[i] ^= int(rng.integers(1, 256))
        assert rs_correct_block(code, nsym)[:40] == msg

def test_rs_stage_roundtrip_clean():
    rs = ReedSolomon(32)
    for msg in (b'', b'hi', bytes(range(256)) + bytes(range(50))):
        assert rs.inverse(rs.forward(msg))[:len(msg)] == msg

def test_rs_corrects_burst():
    rs = ReedSolomon(32)
    payload = bytes(range(200))
    coded = bytearray(rs.forward(payload))
    for i in range(30, 42):
        coded[i] ^= 165
    assert rs.inverse(bytes(coded))[:len(payload)] == payload

def test_byte_interleaver_roundtrip():
    il = ByteInterleaver(16)
    for data in (b'', b'abc', bytes(range(100))):
        assert il.inverse(il.forward(data)) == data

def _burst(data: bytes, start_bit: int, length: int) -> bytes:
    bits = bytes_to_bits(data).copy()
    bits[start_bit:start_bit + length] ^= 1
    return bits_to_bytes(bits)

def test_spread_repetition_beats_adjacent_on_burst():
    from stegopipe.codec import RepetitionECC
    payload = bytes(range(120))
    adjacent = RepetitionECC(5)
    spread = SpreadRepetition(5)
    a_coded = adjacent.forward(payload)
    a_out = adjacent.inverse(_burst(a_coded, 200, 120))
    s_coded = spread.forward(payload)
    s_out = spread.inverse(_burst(s_coded, 200, 120))

    def ber(x):
        n = min(len(payload), len(x))
        return float(np.mean(bytes_to_bits(payload[:n]) != bytes_to_bits(x[:n])))
    assert ber(s_out) < ber(a_out)
    assert ber(s_out) == 0.0

def test_spread_repetition_header_survives_channel_noise():
    import numpy as np
    from stegopipe.codec import SpreadRepetition
    payload = bytes(range(100))
    code = SpreadRepetition(5)
    coded = bytes_to_bits(code.forward(payload)).copy()
    rng = np.random.default_rng(0)
    coded[rng.random(coded.size) < 0.05] ^= 1
    out = code.inverse(bits_to_bytes(coded))
    n = min(len(payload), len(out))
    assert float(np.mean(bytes_to_bits(payload[:n]) != bytes_to_bits(out[:n]))) < 0.02
