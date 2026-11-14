"""MegaLoc, a place-recognition model (DINOv2 + SALAD; the authors' ONNX export): one descriptor
per frame, unit length, so a dot product is how alike two places look.

The export takes 322x322 images only, so frames are squashed to that, as MegaLoc's own resize
does. It runs on the CPU: DirectML fails on the export's Reshape node.
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

SIZE = 322
BATCH = 8
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


def identity(path: Path) -> dict:
    """What says two caches came from the same model: its file name and size."""
    return {"name": path.name, "bytes": path.stat().st_size}


class MegaLoc:
    def __init__(self, path: Path):
        import onnxruntime as ort

        ort.set_default_logger_severity(3)  # errors only
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name

    def _run(self, chunk: list[bytes]) -> np.ndarray:
        images = np.stack([np.frombuffer(b, np.uint8).reshape(SIZE, SIZE, 3) for b in chunk])
        x = ((images.astype(np.float32) / 255.0 - MEAN) / STD).transpose(0, 3, 1, 2)
        y = self.session.run(None, {self.input: np.ascontiguousarray(x)})[0].astype(np.float32)
        return (y / np.maximum(np.linalg.norm(y, axis=1, keepdims=True), 1e-12)).astype(np.float16)

    def describe(self, frames: Iterable[bytes]) -> Iterator[np.ndarray]:
        """Descriptors (float16 rows) of raw RGB SIZE x SIZE frames, a batch at a time."""
        chunk: list[bytes] = []
        for frame in frames:
            chunk.append(frame)
            if len(chunk) == BATCH:
                yield self._run(chunk)
                chunk = []
        if chunk:
            yield self._run(chunk)
