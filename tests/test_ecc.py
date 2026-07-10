import numpy as np
import pytest
from stegopipe.codec import Hamming74, RepetitionECC
from stegopipe.bitstream import bits_to_bytes, bytes_to_bits

@pytest.mark.parametrize('code', [RepetitionECC(3), RepetitionECC(5), Hamming74()])
def test_clean_roundtrip(code):
    for msg in (b'', b'A', b'hello world', bytes(range(64))):
        out = code.inverse(code.forward(msg))
        assert out[:len(msg)] == msg

def _flip_bits(data: bytes, positions) -> bytes:
    bits = bytes_to_bits(data).copy()
    for p in positions:
        if p < bits.size:
            bits[p] ^= 1
    return bits_to_bytes(bits)

def test_repetition_corrects_single_error():
    code = RepetitionECC(3)
    msg = b'correct me'
    coded = code.forward(msg)
    corrupted = _flip_bits(coded, range(0, bytes_to_bits(coded).size, 3))
    assert code.inverse(corrupted)[:len(msg)] == msg

def test_hamming_corrects_one_error_per_block():
    code = Hamming74()
    msg = b'hamming!!'
    coded = code.forward(msg)
    corrupted = _flip_bits(coded, range(2, bytes_to_bits(coded).size, 7))
    assert code.inverse(corrupted)[:len(msg)] == msg

def test_repetition_rejects_even_r():
    with pytest.raises(ValueError):
        RepetitionECC(4)

def test_hamming_rate():
    assert len(Hamming74().forward(b'abcd')) == 7
