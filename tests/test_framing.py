import pytest
from stegopipe.framing import FrameStage, FrameError, HEADER_SIZE

def test_frame_roundtrip():
    stage = FrameStage()
    payload = b'the quick brown fox'
    blob = stage.forward(payload)
    assert len(blob) == HEADER_SIZE + len(payload)
    assert stage.inverse(blob) == payload

def test_empty_payload():
    stage = FrameStage()
    assert stage.inverse(stage.forward(b'')) == b''

def test_bad_magic():
    with pytest.raises(FrameError):
        FrameStage().inverse(b'XXXX' + b'\x00' * 20)

def test_crc_detects_corruption():
    stage = FrameStage()
    blob = bytearray(stage.forward(b'important'))
    blob[-1] ^= 255
    with pytest.raises(FrameError, match='CRC'):
        stage.inverse(bytes(blob))

def test_truncated():
    stage = FrameStage()
    blob = stage.forward(b'abcdef')
    with pytest.raises(FrameError):
        stage.inverse(blob[:-2])

def test_encrypted_flag():
    assert FrameStage(encrypted=True).forward(b'x')[5] & 1
