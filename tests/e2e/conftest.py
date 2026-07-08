"""E2E fixtures: synthetic PDF + mocked LLM/OCR."""
from __future__ import annotations

import zlib
import struct
from pathlib import Path
from typing import Callable

import fitz
import pytest


def _tiny_png(width: int = 8, height: int = 8) -> bytes:
    def chunk(typ: bytes, data: bytes) -> bytes:
        c = typ + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    sig = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = b"".join(b"\x00\xff\x00\xff\x00\xff\x00\xff" for _ in range(height))
    idat = zlib.compress(raw)
    return sig + chunk(b"IHDR", ihdr) + chunk(b"IDAT", idat) + chunk(b"IEND", b"")


@pytest.fixture
def make_pdf(tmp_dir: Path) -> Callable[..., Path]:
    def _build(pages: list[dict], name: str = "paper.pdf") -> Path:
        doc = fitz.open()
        for spec in pages:
            p = doc.new_page()
            if spec.get("image"):
                p.insert_image(fitz.Rect(0, 0, 200, 200), stream=_tiny_png())
            if spec.get("text"):
                p.insert_text((72, 72), spec["text"])
        path = tmp_dir / name
        doc.save(str(path))
        doc.close()
        return path

    return _build
