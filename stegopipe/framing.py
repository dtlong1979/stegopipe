from __future__ import annotations
import struct
import zlib
from dataclasses import dataclass
MAGIC = b'SPZ2'
VERSION = 2
HEADER = struct.Struct('>4sBBII')
HEADER_SIZE = HEADER.size
FLAG_ENCRYPTED = 1

class FrameError(ValueError):
    pass

@dataclass
class FrameStage:
    encrypted: bool = False

    def forward(self, data: bytes) -> bytes:
        flags = FLAG_ENCRYPTED if self.encrypted else 0
        crc = zlib.crc32(data) & 4294967295
        header = HEADER.pack(MAGIC, VERSION, flags, len(data), crc)
        return header + data

    def inverse(self, blob: bytes) -> bytes:
        if len(blob) < HEADER_SIZE:
            raise FrameError('blob shorter than frame header')
        magic, version, flags, length, crc = HEADER.unpack(blob[:HEADER_SIZE])
        if magic != MAGIC:
            raise FrameError(f'bad magic {magic!r} (not a stegopipe frame)')
        if version != VERSION:
            raise FrameError(f'unsupported frame version {version}')
        payload = blob[HEADER_SIZE:HEADER_SIZE + length]
        if len(payload) != length:
            raise FrameError('frame truncated: payload shorter than declared length')
        if zlib.crc32(payload) & 4294967295 != crc:
            raise FrameError('CRC mismatch: corrupted data or wrong key')
        return payload
