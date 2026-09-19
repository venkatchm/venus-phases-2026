"""Minimal PNG writer and reader, so headless renders need no image library."""
from __future__ import annotations

import struct
import zlib


def write_png(path: str, rgb, level: int = 6) -> None:
    """`rgb` is a (height, width, 3) uint8 numpy array.

    `level` is the zlib compression level. The default suits a final image; a
    render writing thousands of intermediate frames to disk should pass 1,
    where zlib costs a fraction as much and the files are only a little larger.

    Rows are assembled in one numpy operation rather than by appending each row
    to a bytearray in Python: at 1920x1080 that loop was a measurable share of a
    frame's total render time.
    """
    import numpy as np

    height, width = rgb.shape[0], rgb.shape[1]
    # PNG wants a filter-type byte in front of every scanline; filter 0 is None
    raw = np.zeros((height, 1 + width * 3), dtype=np.uint8)
    raw[:, 1:] = np.asarray(rgb, dtype=np.uint8).reshape(height, width * 3)
    raw = raw.tobytes()

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    # Written to a neighbouring temp file and renamed into place. os.replace is
    # atomic within a filesystem, so a process killed mid-write leaves either the
    # previous file or none -- never a half-written one that a later run would
    # mistake for finished work.
    import os

    tmp = f"{path}.part"
    with open(tmp, "wb") as fh:
        fh.write(b"\x89PNG\r\n\x1a\n")
        fh.write(chunk(b"IHDR", header))
        fh.write(chunk(b"IDAT", zlib.compress(raw, level)))
        fh.write(chunk(b"IEND", b""))
    os.replace(tmp, path)


IEND = b"\x00\x00\x00\x00IEND\xaeB`\x82"


def png_complete(path: str) -> bool:
    """True if `path` is a PNG that was finished being written.

    A process killed partway through leaves a file that exists but is truncated
    -- often zero bytes. Anything resuming from a directory of frames has to
    check for that, because `os.path.exists` will happily confirm the corpse.
    """
    import os

    try:
        if os.path.getsize(path) < len(IEND) + 8:
            return False
        with open(path, "rb") as fh:
            fh.seek(-len(IEND), os.SEEK_END)
            return fh.read(len(IEND)) == IEND
    except OSError:
        return False


def read_png(path: str):
    """Read back a PNG written by `write_png`. Returns (height, width, 3) uint8.

    Deliberately narrow: 8-bit truecolour, filter type 0, which is exactly what
    `write_png` produces. It exists because `imageio` ships no PNG *reader*
    backend by default -- it can encode video perfectly well but cannot open a
    PNG without Pillow, pyav or opencv -- and a frame cache that can only be
    written is not a cache. Hand-rolling the reader keeps the dependency list
    where it is, and the writer next door already sets the precedent.
    """
    import numpy as np

    with open(path, "rb") as fh:
        data = fh.read()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError(f"{path} is not a PNG")

    pos, width, height = 8, None, None
    idat = bytearray()
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        tag = data[pos + 4:pos + 8]
        payload = data[pos + 8:pos + 8 + length]
        if tag == b"IHDR":
            width, height, depth, colour = struct.unpack(">IIBB", payload[:10])
            if depth != 8 or colour != 2:
                raise ValueError(f"{path}: only 8-bit RGB is supported")
        elif tag == b"IDAT":
            idat += payload
        elif tag == b"IEND":
            break
        pos += 12 + length                  # length + tag + payload + CRC

    if width is None:
        raise ValueError(f"{path}: no IHDR chunk")
    raw = np.frombuffer(zlib.decompress(bytes(idat)), dtype=np.uint8)
    rows = raw.reshape(height, 1 + width * 3)
    if rows[:, 0].any():
        raise ValueError(f"{path}: only filter type 0 is supported")
    return rows[:, 1:].reshape(height, width, 3)
