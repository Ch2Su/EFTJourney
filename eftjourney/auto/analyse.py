"""Find each finished raid's recording, describe its frames once, and localise every raid that has
a recording against the other raids of its map whose screenshots place their frames.

data/auto/frames/<raid-id>.npz  one MegaLoc descriptor a second (kept: the recording may go)
data/auto/<raid-id>.json        the raid's automatic route and the result its recording shows, read by the viewer
data/auto/recordings.json       what ffprobe said about each recording, so it asks once
data/auto/outcomes.json         the result read from each raid's recording, so it is read once
"""
from __future__ import annotations

import io
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable

import numpy as np

from ..builder import _iso, _wall_clock, game_running
from ..mapdata import entry_for
from ..store import OUTCOME_STATUSES, RaidStore, _atomic_write_json, _read_json, _while_in_use
from . import outcome as oc
from . import path as pm
from . import video
from .config import Tools, auto_dir
from .model import BATCH, SIZE, MegaLoc, identity

SCHEMA = 1
GENERATOR = "eftjourney"
MAX_GAP = 30.0  # seconds between two screenshots across which a frame's position is interpolated
MIN_OVERLAP = 30.0  # seconds a recording must share with a raid to be its recording
MIN_PLACES = 30  # reference places a map needs before its raids are localised
GAME_CHECK = 30.0  # seconds between looks for the game while frames are described
MAX_SURE = 999.0  # sureness where nothing else is in the running (infinity) is stored as this
PROGRESS = 120  # frames between progress lines


@dataclass
class AutoReport:
    recordings: int = 0
    raids: int = 0  # finished raids with a recording
    described: list[str] = field(default_factory=list)
    routes: list[str] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)  # raid id -> why it got no route
    checks: dict[str, dict] = field(default_factory=dict)  # raid id -> its route against its screenshots
    outcomes: dict[str, str] = field(default_factory=dict)  # raid id -> the result its recording shows
    unread: int = 0  # recordings whose result screens could not be read this time (no Windows OCR)
    stopped: str | None = None


@dataclass
class Raid:
    id: str
    doc: dict
    map_key: str
    start: datetime
    end: datetime
    recording: Path | None = None
    video_start: datetime | None = None
    probe: dict | None = None
    t: np.ndarray | None = None  # seconds since the raid's start, one a frame
    desc: np.ndarray | None = None
    truth: np.ndarray | None = None  # positions from the screenshots, NaN where unknown
    outcome: dict | None = None  # the result its recording shows


def truth_at(times: np.ndarray, track: list[dict], max_gap: float = MAX_GAP) -> np.ndarray:
    """Positions at `times` (seconds since the raid's start) interpolated between the exactly timed
    screenshots around each, at most `max_gap` s apart; NaN elsewhere (before the first one too)."""
    pts = sorted((s["t"], s["x"], s["y"], s["z"]) for s in track
                 if not s.get("approx") and all(isinstance(s.get(k), (int, float)) and not isinstance(s.get(k), bool)
                                                for k in ("t", "x", "y", "z")))
    out = np.full((len(times), 3), np.nan)
    if len(pts) < 2:
        return out
    tv = np.array([p[0] for p in pts], float)
    sp = np.array([p[1:] for p in pts], float)
    for i, t in enumerate(times):
        j = int(np.searchsorted(tv, t))
        if j == 0 or j >= len(tv) or tv[j] - tv[j - 1] > max_gap:
            continue
        gap = tv[j] - tv[j - 1]
        out[i] = sp[j - 1] + (sp[j] - sp[j - 1]) * ((t - tv[j - 1]) / gap if gap > 0 else 0.0)
    return out


def _probed(tools: Tools, files: list[Path], cache_file: Path) -> dict[Path, dict]:
    """ffprobe's answer for every recording, asked once per file version (name, size, time)."""
    cache = _read_json(cache_file)
    cache = cache if isinstance(cache, dict) else {}
    out, fresh = {}, {}
    for path in files:
        st = path.stat()
        key = f"{path.name}|{st.st_size}|{int(st.st_mtime)}"
        info = cache.get(key)
        if not isinstance(info, dict) and tools.ffprobe:
            info = video.probe(tools.ffprobe, path)
        if isinstance(info, dict):
            fresh[key] = info
            out[path] = info
    if fresh != cache:
        _atomic_write_json(cache_file, fresh)
    return out


def _finished_raids(store: RaidStore, maps: dict) -> list[Raid]:
    raids = []
    for raid_id in store.list_ids():
        doc = store.merged(raid_id)
        if not doc or doc.get("demo"):
            continue
        entry = entry_for(maps, doc.get("map"))
        start, end = _wall_clock(doc.get("startedAt")), _wall_clock(doc.get("endedAt"))
        if entry and start and end and end > start:
            raids.append(Raid(raid_id, doc, entry["key"], start, end))
    return raids


def _match(raids: list[Raid], probed: dict[Path, dict]) -> None:
    """Give each raid the recording that covers most of it (at least MIN_OVERLAP s)."""
    spans = []
    for path, info in probed.items():
        try:
            first = video.first_frame(path, info.get("tag"))
        except OSError:
            continue
        spans.append((path, info, first, first + timedelta(seconds=info["duration"])))
    for raid in raids:
        best = None
        for path, info, lo, hi in spans:
            overlap = (min(hi, raid.end) - max(lo, raid.start)).total_seconds()
            if overlap >= MIN_OVERLAP and (best is None or overlap > best[0]):
                best = overlap, path, info, lo
        if best:
            _, raid.recording, raid.probe, raid.video_start = best


def _frames_file(data_dir: Path, raid_id: str) -> Path:
    return auto_dir(data_dir) / "frames" / f"{raid_id}.npz"


def _cache_meta(raid: Raid, model: dict) -> dict:
    return {"raidStart": raid.doc.get("startedAt"), "model": model, "size": SIZE, "step": pm.STEP}


def _load_frames(data_dir: Path, raid: Raid, model: dict | None) -> bool:
    """The raid's descriptors from an earlier run, when they came from this model and raid start."""
    try:
        with np.load(_frames_file(data_dir, raid.id)) as z:
            meta = json.loads(str(z["meta"]))
            want = _cache_meta(raid, model) if model else None
            if want is not None and any(meta.get(k) != want[k] for k in want):
                return False
            raid.t, raid.desc = z["t"], z["desc"]
    except (OSError, ValueError, KeyError):
        return False
    return len(raid.t) == len(raid.desc) > 0


def _save_frames(data_dir: Path, raid: Raid, model: dict) -> None:
    target = _frames_file(data_dir, raid.id)
    target.parent.mkdir(parents=True, exist_ok=True)
    meta = dict(_cache_meta(raid, model), video=raid.recording.name, videoStart=_iso(raid.video_start))
    buf = io.BytesIO()
    np.savez_compressed(buf, t=raid.t, desc=raid.desc, meta=json.dumps(meta, ensure_ascii=False))
    tmp = target.with_name(target.name + ".tmp")
    tmp.write_bytes(buf.getvalue())
    _while_in_use(os.replace, tmp, target)


def _describe(raid: Raid, tools: Tools, megaloc: MegaLoc, log: Callable[[str], None],
              stop: Callable[[], bool]) -> bool:
    """One descriptor a second of the raid's part of its recording; False when stopped midway."""
    offset = (raid.start - raid.video_start).total_seconds()  # raid time = recording time - offset
    lo = max(0.0, offset)
    hi = min(raid.probe["duration"], offset + (raid.end - raid.start).total_seconds())
    expected = int((hi - lo) // pm.STEP) + 1
    log(f"  {raid.id}: {raid.recording.name}에서 {expected}초 분량을 분석합니다")
    frames = video.frames(tools.ffmpeg, raid.recording, raid.probe["pix_fmt"], lo, hi - lo, SIZE, pm.STEP)
    parts, done, checked = [], 0, time.monotonic()
    for desc in megaloc.describe(frames):
        parts.append(desc)
        done += len(desc)
        if done // PROGRESS != (done - len(desc)) // PROGRESS:
            log(f"    {done}/{expected}")
        if time.monotonic() - checked >= GAME_CHECK:
            checked = time.monotonic()
            if stop():
                return False
    if not parts:
        return False
    raid.desc = np.concatenate(parts)
    raid.t = lo - offset + np.arange(len(raid.desc)) * pm.STEP
    return True


def _read_outcomes(raids: list[Raid], tools: Tools, data_dir: Path, report: AutoReport, read: Callable[[Raid], bool],
                   log: Callable[[str], None]) -> None:
    """The result each raid's recording shows after its end, read once per recording (outcomes.json);
    `read` says whether a raid's screens may be read now (else only what was read before counts)."""
    cache_file = auto_dir(data_dir) / "outcomes.json"
    cache = _read_json(cache_file)
    cache = cache if isinstance(cache, dict) else {}
    changed = False
    for raid in raids:
        known = cache.get(raid.id)
        fresh = isinstance(known, dict) and raid.recording is not None and known.get("video") == raid.recording.name
        if raid.recording is not None and tools.ffmpeg is not None and not fresh and read(raid):
            log(f"  {raid.id}: 녹화의 결과 화면을 읽습니다")
            end = (raid.end - raid.video_start).total_seconds()
            seen = oc.read_result(tools.ffmpeg, raid.recording, end, raid.probe["duration"])
            if seen is None:
                report.unread += 1
            else:
                known = cache[raid.id] = dict(seen, video=raid.recording.name)
                changed = True
        if isinstance(known, dict) and known.get("status") in OUTCOME_STATUSES:
            raid.outcome = {"status": known["status"], "source": "video"}
            report.outcomes[raid.id] = known["status"]
    if changed:
        _atomic_write_json(cache_file, cache)


def _route_doc(raid: Raid, refs: list[Raid], pos: np.ndarray, est: np.ndarray, sure: np.ndarray) -> dict:
    sure = np.minimum(sure, MAX_SURE)
    doc = {
        "schema": SCHEMA,
        "generator": GENERATOR,
        "raid": raid.id,
        "video": raid.recording.name if raid.recording else None,
        "model": "MegaLoc",
        "step": pm.STEP,
        "reach": pm.REACH,
        "radius": pm.RADIUS,
        "sureTiers": list(pm.SURE),
        "references": {"raids": [r.id for r in refs], "places": int(len(pos))},
        "route": [{"t": round(float(t), 1), "x": round(float(p[0]), 2), "y": round(float(p[1]), 2),
                   "z": round(float(p[2]), 2), "sure": round(float(s), 1)} for t, p, s in zip(raid.t, est, sure)],
        "createdAt": _iso(datetime.now()),
    }
    ok = ~np.isnan(raid.truth[:, 0])
    if ok.sum() >= 10:  # its own screenshots tell how right the route is
        dxz = np.hypot(est[ok, 0] - raid.truth[ok, 0], est[ok, 2] - raid.truth[ok, 2])
        hit = (dxz <= pm.RADIUS) & (np.abs(est[ok, 1] - raid.truth[ok, 1]) <= pm.FLOOR)
        sure_ok = sure[ok] >= pm.SURE[1]
        doc["check"] = {"seconds": int(ok.sum()), "within": round(float(hit.mean()), 3),
                        "medianError": round(float(np.median(dxz)), 1),
                        "sureShare": round(float(sure_ok.mean()), 3),
                        "sureWithin": round(float(hit[sure_ok].mean()), 3) if sure_ok.any() else None}
    return doc


def analyse(store: RaidStore, maps: dict, data_dir: Path, tools: Tools, log: Callable[[str], None] = print,
            only: set[str] | None = None, game: Callable[[], bool | None] = game_running) -> AutoReport:
    """Describe the frames still missing, then write every localisable raid's route."""
    report = AutoReport()
    raids = _finished_raids(store, maps)
    files = video.list_recordings(tools.videos)
    report.recordings = len(files)
    _match(raids, _probed(tools, files, auto_dir(data_dir) / "recordings.json"))
    model = identity(tools.model) if tools.model.is_file() else None
    megaloc = None
    stop = lambda: bool(game())  # noqa: E731

    for raid in raids:
        if not _load_frames(data_dir, raid, model) and raid.recording is not None:
            if model is None or tools.ffmpeg is None:
                report.skipped[raid.id] = "모델이나 FFmpeg가 없어 영상을 분석하지 못했습니다"
                continue
            if stop():
                report.stopped = "게임이 실행 중이라 영상 분석을 멈췄습니다. 게임을 끈 뒤 다시 실행하면 이어서 합니다."
                break
            if megaloc is None:
                megaloc = MegaLoc(tools.model)
            if not _describe(raid, tools, megaloc, log, stop):
                report.stopped = "게임이 시작되어 영상 분석을 멈췄습니다. 게임을 끈 뒤 다시 실행하면 이어서 합니다."
                raid.t = raid.desc = None
                break
            _save_frames(data_dir, raid, model)
            report.described.append(raid.id)
    stopped = report.stopped is not None
    _read_outcomes(raids, tools, data_dir, report, lambda r: not stopped and (not only or r.id in only) and not stop(), log)
    for raid in raids:
        if raid.desc is not None:
            raid.truth = truth_at(raid.t, raid.doc.get("track") or [])
    have = [r for r in raids if r.desc is not None]
    report.raids = len(have)

    for raid in have:
        if only and raid.id not in only:
            continue
        refs = [r for r in have if r.map_key == raid.map_key and r is not raid and (~np.isnan(r.truth[:, 0])).any()]
        known = [~np.isnan(r.truth[:, 0]) for r in refs]
        places = int(sum(k.sum() for k in known))
        if places < MIN_PLACES:
            report.skipped[raid.id] = "같은 맵에서 위치를 아는 다른 레이드 영상이 부족합니다"
            continue
        pos = np.concatenate([r.truth[k] for r, k in zip(refs, known)])
        ref_desc = np.concatenate([r.desc[k] for r, k in zip(refs, known)]).astype(np.float32)
        S = raid.desc.astype(np.float32) @ ref_desc.T
        _, est, sure = pm.locate(S, pos)
        doc = _route_doc(raid, [r for r, k in zip(refs, known) if k.any()], pos, est, sure)
        if raid.recording is None:  # described earlier; the recording has gone since
            meta = _read_meta(data_dir, raid.id)
            doc["video"] = meta.get("video")
        if raid.outcome:
            doc["outcome"] = raid.outcome
        _atomic_write_json(store.auto_file(raid.id), doc)
        report.routes.append(raid.id)
        if "check" in doc:
            report.checks[raid.id] = doc["check"]

    for raid in raids:  # a result without a route of its own (no references on its map): kept all the same
        if raid.outcome is None or raid.id in report.routes or (only and raid.id not in only):
            continue
        doc = _read_json(store.auto_file(raid.id))
        if not isinstance(doc, dict):
            doc = {"schema": SCHEMA, "generator": GENERATOR, "raid": raid.id,
                   "video": raid.recording.name if raid.recording else None, "route": []}
        if doc.get("outcome") != raid.outcome:
            doc["outcome"] = raid.outcome
            _atomic_write_json(store.auto_file(raid.id), doc)
    return report


def _read_meta(data_dir: Path, raid_id: str) -> dict:
    try:
        with np.load(_frames_file(data_dir, raid_id)) as z:
            meta = json.loads(str(z["meta"]))
    except (OSError, ValueError, KeyError):
        return {}
    return meta if isinstance(meta, dict) else {}
