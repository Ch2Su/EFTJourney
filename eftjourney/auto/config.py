"""Where the analysis finds its tools, without importing them (doctor uses this too).

Folders and files given once on the command line are remembered in data/auto/settings.json.
"""
from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path

from ..store import _atomic_write_json, _read_json
from . import video

AUTO_DIR = "auto"
SETTINGS_FILE = "settings.json"
MODEL_FILE = "megaloc.onnx"
MODEL_URL = "https://huggingface.co/gberton/MegaLoc"  # the authors' ONNX export
NEEDS = ("numpy", "onnxruntime")
KEYS = ("videos", "ffmpeg", "model")


@dataclass
class Tools:
    videos: Path | None  # the recordings folder
    ffmpeg: Path | None
    ffprobe: Path | None
    model: Path  # where the model is looked for
    remembered: dict

    @property
    def model_found(self) -> bool:
        return self.model.is_file()


def missing_modules() -> list[str]:
    return [name for name in NEEDS if importlib.util.find_spec(name) is None]


def auto_dir(data_dir: Path) -> Path:
    return Path(data_dir) / AUTO_DIR


def load_settings(data_dir: Path) -> dict:
    doc = _read_json(auto_dir(data_dir) / SETTINGS_FILE)
    if not isinstance(doc, dict):
        return {}
    return {k: v for k, v in doc.items() if k in KEYS and isinstance(v, str) and v}


def remember(data_dir: Path, **values: str | os.PathLike | None) -> dict:
    """Save the folders and files given (absolute), keeping the ones not given."""
    settings = load_settings(data_dir)
    given = {k: os.path.abspath(v) for k, v in values.items() if k in KEYS and v}
    if given and any(settings.get(k) != v for k, v in given.items()):
        settings.update(given)
        _atomic_write_json(auto_dir(data_dir) / SETTINGS_FILE, settings)
    return settings


def resolve(data_dir: Path, videos: str | None = None, ffmpeg: str | None = None, model: str | None = None) -> Tools:
    """The tools to use: what was given now, else what was remembered, else the defaults."""
    saved = load_settings(data_dir)
    videos = videos or saved.get("videos")
    ffmpeg = ffmpeg or saved.get("ffmpeg")
    model = model or saved.get("model")
    return Tools(
        videos=video.recordings_dir(videos),
        ffmpeg=video.find_tool("ffmpeg", ffmpeg),
        ffprobe=video.find_tool("ffprobe", ffmpeg),
        model=Path(model) if model else auto_dir(data_dir) / MODEL_FILE,
        remembered=saved,
    )
