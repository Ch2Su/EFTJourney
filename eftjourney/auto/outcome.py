"""The raid's result from the screens after it in its recording.

Once a raid ends the game shows its summary ('레이드 종료' with the result: '작전 중 사망', '생존'
on a band) and then the experience screen, whose band says it again ('작전 중 사망 (KIA)',
'생존함'). Windows' own OCR (Windows.Media.Ocr, run through Windows PowerShell 5.1: no download)
reads a frame a second from the raid's end on, and the result most frames show wins. Only the
result is kept: the screens also show the player's nickname, which is never stored.
"""
from __future__ import annotations

import base64
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

from .video import NO_WINDOW

BEFORE = 2.0  # seconds before the raid's end the frames start
AFTER = 120.0  # seconds after it the result screens are looked for
TIMEOUT = 600  # seconds the OCR of one raid's frames may take
# A result in the words the Korean and English clients write it (spaces taken out: OCR drops and adds them)
RESULTS = (
    ("killed", ("작전중사망", "(KIA)", "KILLEDINACTION")),
    ("mia", ("작전중실종", "(MIA)", "MISSINGINACTION")),
    ("left", ("작전중이탈", "LEFTTHEACTION")),
    ("survived", ("생존함", "SURVIVED")),
    ("runthrough", ("RUNTHROUGH",)),
)
ALONE = {"생존": "survived", "도주": "runthrough", "도주.": "runthrough", "도주함": "runthrough"}  # band words on a line of their own

# Windows PowerShell 5.1 (WinRT types; PowerShell 7 lacks them): OCR every image listed in
# $env:EFTJ_OCR_LIST, one JSON line each into $env:EFTJ_OCR_OUT. Exit 3: no OCR language installed.
SCRIPT = r"""
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Runtime.WindowsRuntime
$asTask = ([System.WindowsRuntimeSystemExtensions].GetMethods() | Where-Object {
    $_.Name -eq 'AsTask' -and $_.GetParameters().Count -eq 1 -and $_.GetParameters()[0].ParameterType.Name -eq 'IAsyncOperation`1' })[0]
function Await($op, [type]$type) { $t = $asTask.MakeGenericMethod($type).Invoke($null, @($op)); $t.Wait(-1) | Out-Null; $t.Result }
[Windows.Storage.StorageFile, Windows.Storage, ContentType = WindowsRuntime] | Out-Null
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
[Windows.Graphics.Imaging.BitmapDecoder, Windows.Graphics, ContentType = WindowsRuntime] | Out-Null
[Windows.Storage.Streams.IRandomAccessStream, Windows.Storage.Streams, ContentType = WindowsRuntime] | Out-Null
[Windows.Globalization.Language, Windows.Globalization, ContentType = WindowsRuntime] | Out-Null
$engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromLanguage([Windows.Globalization.Language]::new('ko'))
if (-not $engine) { $engine = [Windows.Media.Ocr.OcrEngine]::TryCreateFromUserProfileLanguages() }
if (-not $engine) { exit 3 }
$out = New-Object System.IO.StreamWriter($env:EFTJ_OCR_OUT, $false, (New-Object System.Text.UTF8Encoding $false))
foreach ($path in [System.IO.File]::ReadAllLines($env:EFTJ_OCR_LIST)) {
    $file = Await ([Windows.Storage.StorageFile]::GetFileFromPathAsync($path)) ([Windows.Storage.StorageFile])
    $stream = Await ($file.OpenAsync([Windows.Storage.FileAccessMode]::Read)) ([Windows.Storage.Streams.IRandomAccessStream])
    $decoder = Await ([Windows.Graphics.Imaging.BitmapDecoder]::CreateAsync($stream)) ([Windows.Graphics.Imaging.BitmapDecoder])
    $bitmap = Await ($decoder.GetSoftwareBitmapAsync()) ([Windows.Graphics.Imaging.SoftwareBitmap])
    $result = Await ($engine.RecognizeAsync($bitmap)) ([Windows.Media.Ocr.OcrResult])
    $stream.Dispose()
    $out.WriteLine((ConvertTo-Json -Compress -InputObject @{ file = $path; lines = @($result.Lines | ForEach-Object { $_.Text }) }))
}
$out.Close()
"""
LANGUAGES = r"""
[Windows.Media.Ocr.OcrEngine, Windows.Foundation, ContentType = WindowsRuntime] | Out-Null
([Windows.Media.Ocr.OcrEngine]::AvailableRecognizerLanguages | ForEach-Object { $_.LanguageTag }) -join ','
"""


def _powershell(script: str, env: dict | None = None, timeout: float = 60) -> subprocess.CompletedProcess | None:
    """Run a script in Windows PowerShell 5.1, passed encoded: no file, so no execution policy applies."""
    if sys.platform != "win32":
        return None
    encoded = base64.b64encode(script.encode("utf-16-le")).decode("ascii")
    try:
        return subprocess.run(["powershell.exe", "-NoProfile", "-NonInteractive", "-EncodedCommand", encoded],
                              env=env, capture_output=True, text=True, timeout=timeout, creationflags=NO_WINDOW)
    except (OSError, subprocess.SubprocessError):
        return None


def ocr_languages() -> list[str] | None:
    """The OCR languages Windows has; None where there is no Windows OCR to ask."""
    run = _powershell(LANGUAGES)
    if run is None or run.returncode != 0:
        return None
    return [tag for tag in run.stdout.strip().split(",") if tag]


def ocr(images: list[Path]) -> list[list[str]] | None:
    """The lines of text Windows OCR reads in each image; None when it can't run here."""
    if not images:
        return []
    with tempfile.TemporaryDirectory(prefix="eftjourney-ocr-") as tmp:
        listed, out = Path(tmp) / "images.txt", Path(tmp) / "text.jsonl"
        listed.write_text("\n".join(str(p) for p in images), encoding="utf-8")
        run = _powershell(SCRIPT, dict(os.environ, EFTJ_OCR_LIST=str(listed), EFTJ_OCR_OUT=str(out)), TIMEOUT)
        if run is None or run.returncode != 0 or not out.exists():
            return None
        read: dict[str, list[str]] = {}
        for line in out.read_text(encoding="utf-8").splitlines():
            try:
                doc = json.loads(line)
            except ValueError:
                continue
            lines = doc.get("lines") or []
            read[doc.get("file")] = [str(x) for x in (lines if isinstance(lines, list) else [lines])]
    return [read.get(str(p), []) for p in images]


def result_of(lines: list[str]) -> str | None:
    """The result one frame's text shows, if any."""
    for line in lines:
        flat = re.sub(r"\s+", "", line).upper()
        for status, words in RESULTS:
            if any(word in flat for word in words):
                return status
        if flat in ALONE:
            return ALONE[flat]
    return None


def decide(frames: list[list[str]]) -> tuple[str | None, int]:
    """The result most frames show (on a tie the earliest), and how many show it."""
    seen = [s for s in (result_of(lines) for lines in frames) if s]
    if not seen:
        return None, 0
    counts = Counter(seen)
    best = max(counts.values())
    status = next(s for s in seen if counts[s] == best)
    return status, best


def read_result(ffmpeg: Path, recording: Path, end: float, duration: float) -> dict | None:
    """The result screens after `end` (seconds into the recording): {"status", "frames"}, status None
    when none shows (or the recording stops as the raid ends); None when they could not be read
    (no Windows OCR here, FFmpeg failed): worth trying again later."""
    lo = max(0.0, end - BEFORE)
    hi = min(duration, end + AFTER)
    if hi - end < 5:  # the recording stops as the raid ends: no result screen on it
        return {"status": None, "frames": 0}
    with tempfile.TemporaryDirectory(prefix="eftjourney-end-") as tmp:
        pattern = str(Path(tmp) / "%03d.jpg")
        try:
            subprocess.run([str(ffmpeg), "-hide_banner", "-v", "error", "-hwaccel", "cuda", "-ss", f"{lo:.3f}",
                            "-t", f"{hi - lo:.3f}", "-i", str(recording), "-vf", "fps=1", "-q:v", "3", pattern],
                           capture_output=True, timeout=TIMEOUT, creationflags=NO_WINDOW)
        except (OSError, subprocess.SubprocessError):
            return None
        images = sorted(Path(tmp).glob("*.jpg"))
        if not images:
            return None
        frames = ocr(images)
    if frames is None:
        return None
    status, count = decide(frames)
    return {"status": status, "frames": count}
