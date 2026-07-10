import numpy as np
from stegopipe.bitstream import bits_to_bytes, bits_to_int, bytes_to_bits, int_to_bits

def test_bytes_bits_roundtrip():
    data = b'hello \x00\xff world'
    bits = bytes_to_bits(data)
    assert bits.size == len(data) * 8
    assert bits_to_bytes(bits) == data

def test_empty():
    assert bytes_to_bits(b'').size == 0
    assert bits_to_bytes(np.zeros(0, dtype=np.uint8)) == b''

def test_int_bits_roundtrip():
    for v in (0, 1, 255, 4096, 2 ** 31 - 1):
        bits = int_to_bits(v, 32)
        assert bits.size == 32
        assert bits_to_int(bits) == v

def test_int_overflow():
    import pytest
    with pytest.raises(ValueError):
        int_to_bits(256, 8)

def test_msb_first_ordering():
    assert list(bytes_to_bits(b'\x80')) == [1, 0, 0, 0, 0, 0, 0, 0]
