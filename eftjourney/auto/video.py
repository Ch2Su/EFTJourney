"""Recordings: where OBS saves them, when their first frame was, and their frames.

The first frame's wall-clock time is the recording file's creation time, to the millisecond
(OBS creates the file as it starts; copies converted by the lab's convert.ps1 keep it). A file
copied by hand gets a new creation time: then the container's creation_time tag (whole seconds)
is the better of the two.
"""
from __future__ import annotations

import configparser
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

VIDEO_EXTS = (".mp4", ".mkv", ".mov", ".flv", ".ts")
SETTLED = 60.0  # seconds since a recording's last write before it counts as finished
TAG_SLACK = 2.0  # seconds a creation time may differ from the tag before the tag wins (a copied file)
TIMEOUT = 60  # seconds ffprobe may take for one file
NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW: no console flashing up


def find_tool(name: str, folder: str | os.PathLike | None = None) -> Path | None:
    """ffmpeg or ffprobe: in `folder` (or its bin folder) when given, else on PATH."""
    exe = name + (".exe" if sys.platform == "win32" else "")
    if folder:
        for candidate in (Path(folder) / exe, Path(folder) / "bin" / exe):
            if candidate.is_file():
                return candidate
        return None
    found = shutil.which(name)
    return Path(found) if found else None


def obs_recordings_dir(appdata: str | None = None) -> Path | None:
    """The folder OBS records into, from its first profile that names one."""
    root = Path(appdata or os.environ.get("APPDATA") or Path.home() / ".config") / "obs-studio" / "basic" / "profiles"
    try:
        profiles = sorted(p for p in root.iterdir() if p.is_dir())
    except OSError:
        return None
    for profile in profiles:
        ini = configparser.ConfigParser(interpolation=None, strict=False)
        try:
            ini.read_string((profile / "basic.ini").read_text(encoding="utf-8-sig", errors="replace"))
        except (OSError, configparser.Error):
            continue
        advanced = ini.get("Output", "Mode", fallback="Simple").lower() == "advanced"
        path = ini.get("AdvOut", "RecFilePath", fallback=None) if advanced else ini.get("SimpleOutput", "FilePath", fallback=None)
        if path:
            return Path(path.replace("\\\\", "\\"))
    return None


def recordings_dir(override: str | os.PathLike | None = None) -> Path | None:
    if override:
        return Path(override)
    return obs_recordings_dir() or (Path.home() / "Videos")


def list_recordings(folder: Path | None, now: float | None = None) -> list[Path]:
    """Finished recordings in `folder`: video files nobody has written for SETTLED seconds."""
    if not folder:
        return []
    now = now if now is not None else datetime.now().timestamp()
    try:
        files = [p for p in folder.iterdir() if p.suffix.lower() in VIDEO_EXTS and p.is_file()]
    except OSError:
        return []
    out = []
    for p in files:
        try:
            if now - p.stat().st_mtime >= SETTLED:
                out.append(p)
        except OSError:
            continue
    return sorted(out)


def _created(st: os.stat_result) -> float:
    birth = getattr(st, "st_birthtime", None)  # Python 3.12+; before that Windows put it in st_ctime
    return birth if birth is not None else st.st_ctime


def _tag_time(text: str | None) -> datetime | None:
    if not text:
        return None
    try:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)  # the tag is UTC by convention
    return dt.astimezone().replace(tzinfo=None)


def first_frame(path: Path, tag: str | None = None) -> datetime:
    """The wall-clock time (naive local) of the recording's first frame."""
    created = datetime.fromtimestamp(_created(path.stat()))
    tagged = _tag_time(tag)
    if tagged is not None and abs((created - tagged).total_seconds()) > TAG_SLACK:
        return tagged
    return created


def probe(ffprobe: Path, path: Path) -> dict | None:
    """Duration (s), the video's pixel format and the creation_time tag; None when it can't be read."""
    try:
        run = subprocess.run([str(ffprobe), "-v", "error", "-select_streams", "v:0", "-show_entries",
                              "stream=pix_fmt:format=duration:format_tags=creation_time", "-of", "json", str(path)],
                             capture_output=True, text=True, timeout=TIMEOUT, creationflags=NO_WINDOW)
        info = json.loads(run.stdout or "{}")
        stream = (info.get("streams") or [{}])[0]
        duration = float(info["format"]["duration"])
    except (OSError, subprocess.SubprocessError, ValueError, KeyError, TypeError):
        return None
    return {"duration": duration, "pix_fmt": stream.get("pix_fmt") or "",
            "tag": (info["format"].get("tags") or {}).get("creation_time")}


def _pipe(ffmpeg: Path, args: list[str], size: int) -> Iterator[bytes]:
    proc = subprocess.Popen([str(ffmpeg), "-hide_banner", "-v", "error", *args, "-f", "rawvideo", "-"],
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, creationflags=NO_WINDOW)
    try:
        while len(buf := proc.stdout.read(size)) == size:
            yield buf
    finally:
        proc.stdout.close()
        proc.kill()
        proc.wait()


def frames(ffmpeg: Path, path: Path, pix_fmt: str, first: float, seconds: float, size: int,
           step: float = 1.0) -> Iterator[bytes]:
    """One RGB frame (size x size, squashed) every `step` s of the recording from `first`, as raw
    bytes: decoded on an NVIDIA GPU when there is one, else on the CPU."""
    scale = f"scale={size}:{size}:flags=area,format=rgb24"
    span = ["-ss", f"{first:.3f}", "-t", f"{seconds:.3f}", "-i", str(path)]
    gpu_format = "p010le" if "10" in pix_fmt else "nv12"
    nbytes = size * size * 3
    any_frame = False
    for buf in _pipe(ffmpeg, ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda", *span,
                              "-vf", f"fps=1/{step:g},hwdownload,format={gpu_format},{scale}"], nbytes):
        any_frame = True
        yield buf
    if not any_frame:  # no CUDA here, or this file's codec isn't decoded there
        yield from _pipe(ffmpeg, [*span, "-vf", f"fps=1/{step:g},{scale}"], nbytes)
