"""eftjourney.auto: the path model, positions from screenshots, recordings, and a whole analysis
with a stand-in model. The parts that need numpy are skipped without it (it is an optional extra)."""
import argparse
import contextlib
import http.client
import importlib.util
import io
import json
import os
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from eftjourney import cli
from eftjourney.auto import config, outcome, video
from eftjourney.mapdata import load_maps
from eftjourney.server import AppContext, make_server
from eftjourney.store import RaidStore

HAS_NUMPY = importlib.util.find_spec("numpy") is not None
needs_numpy = unittest.skipUnless(HAS_NUMPY, "numpy is an optional extra (pip install eftjourney[auto])")
if HAS_NUMPY:
    import numpy as np

    from eftjourney.auto import analyse
    from eftjourney.auto import path as pm


def corridor(n=101, spacing=2.0, z=0.0, y=0.0):
    """Places along x every `spacing` m."""
    return np.array([[i * spacing, y, z] for i in range(n)], float)


def similarity(truth, places, width=4.0, seed=0):
    """How alike a frame at each true position looks to each place: a bump around it, plus noise."""
    rng = np.random.default_rng(seed)
    d = np.hypot(truth[:, None, 0] - places[None, :, 0], truth[:, None, 2] - places[None, :, 2])
    return (np.exp(-d ** 2 / (2 * width ** 2)) + 0.05 * rng.standard_normal(d.shape)).astype(np.float32)


@needs_numpy
class PathModelTest(unittest.TestCase):
    def walk(self, seconds=120, speed=1.5):
        return np.array([[10 + speed * t, 0.0, 0.0] for t in range(seconds)])

    def test_follows_a_walk(self):
        places = corridor()
        truth = self.walk()
        path, est, sure = pm.locate(similarity(truth, places), places)
        err = np.hypot(est[:, 0] - truth[:, 0], est[:, 2] - truth[:, 2])
        self.assertLess(np.median(err), 2.0)
        self.assertTrue((err <= pm.RADIUS).all())
        self.assertEqual(int((np.abs(np.diff(est[:, 0])) > pm.REACH).sum()), 0)  # no jumps
        self.assertTrue((sure > pm.SURE[1]).mean() > 0.9)  # nothing far away looks the same

    def test_look_alike_corridor_is_unsure(self):
        # a second corridor 100 m away that looks exactly the same: the path can't tell which it is
        near, far = corridor(), corridor(z=100.0)
        places = np.concatenate([near, far])
        truth = self.walk()
        S = similarity(truth, near)
        _, _, sure = pm.locate(np.concatenate([S, S], axis=1), places)
        self.assertTrue((sure < pm.SURE[0]).all())

    def test_smooth_never_crosses_a_jump(self):
        est = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0], [100, 0, 0], [101, 0, 0], [102, 0, 0]], float)
        out = pm.smooth(est, 3)
        self.assertEqual(out[2, 0], 1.5)  # the mean of 1 and 2 only: 100 is past a jump
        self.assertEqual(out[3, 0], 100.5)
        self.assertEqual(pm.smooth(est, 1) is est, True)

    def test_neighbours_in_chunks_are_the_same(self):
        rng = np.random.default_rng(1)
        pos = np.column_stack([rng.uniform(0, 60, 300), rng.choice([0.0, 5.0], 300), rng.uniform(0, 60, 300)])
        whole = pm.neighbours(pos)
        with mock.patch.object(pm, "CHUNK", 7):
            parts = pm.neighbours(pos)
        self.assertTrue(np.array_equal(whole[0], parts[0]) and np.array_equal(whole[1], parts[1]))
        d = np.hypot(pos[:, None, 0] - pos[None, :, 0], pos[:, None, 2] - pos[None, :, 2])
        for j in (0, 150, 299):  # the places within reach on the same floor, in index order
            want = np.flatnonzero((d[:, j] <= pm.REACH) & (np.abs(pos[:, 1] - pos[j, 1]) <= pm.FLOOR))
            self.assertEqual(whole[0][j][whole[0][j] >= 0].tolist(), want.tolist())

    def test_nothing_far_away_is_sure(self):
        places = corridor(n=10, spacing=1.0)  # every place within FAR of every other
        truth = np.array([[3.0, 0, 0]] * 5)
        _, _, sure = pm.locate(similarity(truth, places), places)
        self.assertTrue(np.isinf(sure).all())

    def test_tiers(self):
        self.assertEqual([pm.tier(s) for s in (5, 10, 19.9, 20, float("inf"))], ["unsure", "medium", "medium", "sure", "sure"])


@needs_numpy
class TruthTest(unittest.TestCase):
    def test_between_screenshots_only(self):
        track = [{"t": 10.0, "x": 0.0, "y": 1.0, "z": 0.0}, {"t": 20.0, "x": 10.0, "y": 1.0, "z": 0.0},
                 {"t": 25.0, "x": 99.0, "y": 1.0, "z": 0.0, "approx": True},  # minute precision: not used
                 {"t": 60.0, "x": 50.0, "y": 1.0, "z": 0.0},  # 40 s after the last: too far to interpolate
                 {"t": 70.0, "x": 60.0, "y": None, "z": 0.0}]  # no height: not used
        out = analyse.truth_at(np.array([5.0, 10.0, 15.0, 20.0, 30.0]), track)
        self.assertTrue(np.isnan(out[0]).all() and np.isnan(out[1]).all())  # before or at the first one
        self.assertEqual(out[2].tolist(), [5.0, 1.0, 0.0])
        self.assertEqual(out[3].tolist(), [10.0, 1.0, 0.0])
        self.assertTrue(np.isnan(out[4]).all())

    def test_needs_two(self):
        self.assertTrue(np.isnan(analyse.truth_at(np.array([1.0]), [{"t": 0, "x": 0, "y": 0, "z": 0}])).all())


class VideoTest(unittest.TestCase):
    def test_obs_folder_from_its_profile(self):
        with tempfile.TemporaryDirectory() as tmp:
            profile = Path(tmp) / "obs-studio" / "basic" / "profiles" / "제목_없음"
            profile.mkdir(parents=True)
            ini = "\ufeff[Output]\nMode=Simple\n\n[SimpleOutput]\nFilePath=E:/obs-studio/Recording\n\n[AdvOut]\nRecFilePath=C:\\\\Users\\\\me\\\\Videos\n"
            (profile / "basic.ini").write_text(ini, encoding="utf-8")
            self.assertEqual(video.obs_recordings_dir(tmp), Path("E:/obs-studio/Recording"))
            (profile / "basic.ini").write_text(ini.replace("Mode=Simple", "Mode=Advanced"), encoding="utf-8")
            self.assertEqual(video.obs_recordings_dir(tmp), Path("C:\\Users\\me\\Videos"))
            self.assertIsNone(video.obs_recordings_dir(str(Path(tmp) / "none")))

    def test_finished_recordings_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            for name in ("a.mp4", "b.MKV", "c.txt", "d.mp4"):
                (folder / name).write_bytes(b"x")
            old = time.time() - 3600
            for name in ("a.mp4", "b.MKV", "c.txt"):
                os.utime(folder / name, (old, old))
            # d.mp4 was written just now: OBS may still be recording into it
            self.assertEqual([p.name for p in video.list_recordings(folder)], ["a.mp4", "b.MKV"])
            self.assertEqual(video.list_recordings(None), [])
            self.assertEqual(video.list_recordings(folder / "missing"), [])

    def test_first_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = Path(tmp) / "r.mp4"
            f.write_bytes(b"x")
            created = datetime.fromtimestamp(video._created(f.stat()))
            self.assertEqual(video.first_frame(f), created)
            near = (created + timedelta(seconds=1)).astimezone().isoformat()
            self.assertEqual(video.first_frame(f, near), created)  # the creation time has milliseconds
            far = datetime(2026, 11, 2, 11, 17, 55, tzinfo=__import__("datetime").timezone.utc)
            self.assertEqual(video.first_frame(f, "2026-11-02T11:17:55.000000Z"),
                             far.astimezone().replace(tzinfo=None))  # a copied file: the tag wins
            self.assertEqual(video.first_frame(f, "not a time"), created)

    def test_find_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            exe = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
            (Path(tmp) / "bin").mkdir()
            (Path(tmp) / "bin" / exe).write_bytes(b"")
            self.assertEqual(video.find_tool("ffmpeg", tmp), Path(tmp) / "bin" / exe)
            self.assertIsNone(video.find_tool("ffprobe", tmp))


class OutcomeTest(unittest.TestCase):
    # what Windows OCR read on real result screens (1.3.0, Korean client; the nickname line left out)
    SUMMARY_KIA = ["레0[드 종료", "|른 탈출", "933", "레이드 시간.", "작전 중 사망", "0 002202", "EXP+", "다음", "메인 메뉴"]
    XP_KIA = ["@33", "현재: 2 079 322", "아이템 획득 경험치", "작전 중 사망 (KIA)", "90 • 0.75 • 102 • 12 = 830", "〉〉〉다음"]
    XP_SURVIVED = ["탈출 보너스", "* 치료 경험치", "O 생존함", "7727 * 1.5 * 1.02 * 1.2 = 14160", "뒤로"]
    TREATMENT = ["* 치료되지 않은 피해 0 사망원인", "0 최대 체력까지 남은 시간: 00:41:16"]  # no result on it

    def test_one_frame(self):
        self.assertEqual(outcome.result_of(self.SUMMARY_KIA), "killed")
        self.assertEqual(outcome.result_of(self.XP_KIA), "killed")
        self.assertEqual(outcome.result_of(self.XP_SURVIVED), "survived")
        self.assertEqual(outcome.result_of(["작전중 실종 (MIA)"]), "mia")
        self.assertEqual(outcome.result_of(["작전 중 이탈"]), "left")
        self.assertEqual(outcome.result_of(["생존"]), "survived")  # the summary's band, alone on its line
        self.assertEqual(outcome.result_of(["도주."]), "runthrough")
        self.assertEqual(outcome.result_of(["Killed in action (KIA)"]), "killed")
        self.assertEqual(outcome.result_of(["SURVIVED"]), "survived")
        for lines in (self.TREATMENT, ["생존자 3명"], ["탈출까지 : 0.4", "탈출구 도작"], []):
            self.assertIsNone(outcome.result_of(lines), lines)

    def test_most_frames_win(self):
        frames = [[], self.SUMMARY_KIA, self.SUMMARY_KIA, self.TREATMENT, self.XP_KIA, ["생존"], []]
        self.assertEqual(outcome.decide(frames), ("killed", 3))
        self.assertEqual(outcome.decide([["생존"], ["작전 중 사망"]]), ("survived", 1))  # a tie: the earlier
        self.assertEqual(outcome.decide([[], self.TREATMENT]), (None, 0))

    def test_reading_a_recording(self):
        with mock.patch.object(outcome.subprocess, "run") as ffmpeg, mock.patch.object(outcome, "ocr") as ocr:
            # the recording stops as the raid ends: nothing to read, and nothing to try again
            self.assertEqual(outcome.read_result(Path("ffmpeg"), Path("r.mp4"), 150.0, 152.0), {"status": None, "frames": 0})
            ffmpeg.assert_not_called()

            def frames(cmd, **kw):  # FFmpeg writes three frames
                for i in (1, 2, 3):
                    Path(cmd[-1].replace("%03d", f"{i:03d}")).write_bytes(b"jpg")
            ffmpeg.side_effect = frames
            ocr.return_value = [[], self.XP_KIA, self.TREATMENT]
            self.assertEqual(outcome.read_result(Path("ffmpeg"), Path("r.mp4"), 150.0, 180.0), {"status": "killed", "frames": 1})
            span = ffmpeg.call_args[0][0]
            self.assertEqual(span[span.index("-ss") + 1], "148.000")  # from just before the end
            self.assertEqual(span[span.index("-t") + 1], "32.000")  # to the end of the recording
            ocr.return_value = None  # no Windows OCR: worth trying again later
            self.assertIsNone(outcome.read_result(Path("ffmpeg"), Path("r.mp4"), 150.0, 180.0))

    @unittest.skipUnless(os.name == "nt", "Windows OCR")
    def test_windows_says_its_languages(self):
        languages = outcome.ocr_languages()
        self.assertTrue(languages is None or isinstance(languages, list))


class ConfigTest(unittest.TestCase):
    def test_remembers_what_was_given(self):
        with tempfile.TemporaryDirectory() as tmp:
            data = Path(tmp) / "data"
            model = Path(tmp) / "megaloc.onnx"
            model.write_bytes(b"onnx")
            config.remember(data, model=str(model), videos=None)
            tools = config.resolve(data)
            self.assertEqual(tools.model, model)
            self.assertTrue(tools.model_found)
            self.assertEqual(config.resolve(data, model=str(Path(tmp) / "other.onnx")).model, Path(tmp) / "other.onnx")
            (config.auto_dir(data) / config.SETTINGS_FILE).write_text("[1, 2]", encoding="utf-8")
            self.assertEqual(config.load_settings(data), {})  # a broken file: nothing remembered
            self.assertEqual(config.resolve(data).model, config.auto_dir(data) / config.MODEL_FILE)


class StoreAndServerTest(unittest.TestCase):
    RAID = "20261102-202719-laboratory"

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.store = RaidStore(self.root)
        self.store.save_raid({"id": self.RAID, "map": "laboratory", "track": []})

    def test_route_file(self):
        self.assertFalse(self.store.summaries()[0]["auto"])
        self.assertIsNone(self.store.load_auto(self.RAID))
        target = self.store.auto_file(self.RAID)
        target.parent.mkdir(parents=True)
        target.write_text('{"route": [{"t": 0, "x": 1, "y": NaN, "z": 2, "sure": 1e999}, {"t": 1, "x": "?", "z": 0}]}',
                          encoding="utf-8")
        doc = self.store.load_auto(self.RAID)
        self.assertEqual(doc["route"], [{"t": 0, "x": 1, "y": None, "z": 2, "sure": None}])
        self.assertTrue(self.store.summaries()[0]["auto"])
        with self.assertRaises(KeyError):
            self.store.auto_file("../x")

    def test_endpoint(self):
        server = make_server(AppContext(self.store, {"maps": []}, None, self.root / "cache"), port=0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)

        def get(path):
            conn = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=5)
            try:
                conn.request("GET", path)
                res = conn.getresponse()
                return res.status, res.read()
            finally:
                conn.close()

        self.assertEqual(get(f"/api/raids/{self.RAID}/auto")[0], 404)
        self.assertEqual(get("/api/raids/nope/auto")[0], 404)
        target = self.store.auto_file(self.RAID)
        target.parent.mkdir(parents=True)
        target.write_text(json.dumps({"raid": self.RAID, "route": [{"t": 0, "x": 1, "y": 0, "z": 2, "sure": 30}]}),
                          encoding="utf-8")
        status, body = get(f"/api/raids/{self.RAID}/auto")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["route"][0]["sure"], 30)


class CommandTest(unittest.TestCase):
    def rt(self, tmp):
        return SimpleNamespace(data_dir=Path(tmp) / "data", logs_dir=None, shots_dir=Path(tmp) / "shots",
                               maps={"maps": []}, store=RaidStore(Path(tmp) / "data"))

    def run_auto(self, rt, **opts):
        args = argparse.Namespace(**{"videos": None, "ffmpeg": None, "model": None, "raid": None, **opts})
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            code = cli.cmd_auto(rt, args)
        return code, out.getvalue()

    def test_says_what_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.rt(tmp)
            with mock.patch("eftjourney.auto.config.missing_modules", return_value=["onnxruntime"]):
                code, out = self.run_auto(rt)
            self.assertEqual(code, 1)
            self.assertIn('pip install "eftjourney[auto]"', out)
            with mock.patch("eftjourney.auto.config.missing_modules", return_value=[]), \
                    mock.patch("eftjourney.auto.video.find_tool", return_value=None):
                code, out = self.run_auto(rt)
            self.assertEqual(code, 1)
            self.assertIn("FFmpeg", out)
            self.assertIn("MegaLoc 모델이 없습니다", out)
            self.assertFalse((config.auto_dir(rt.data_dir) / config.SETTINGS_FILE).exists())  # nothing that failed is kept

    def test_not_while_the_game_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            rt = self.rt(tmp)
            model = Path(tmp) / "m.onnx"
            model.write_bytes(b"x")
            with mock.patch("eftjourney.auto.config.missing_modules", return_value=[]), \
                    mock.patch("eftjourney.auto.video.find_tool", return_value=Path(tmp) / "ffmpeg"), \
                    mock.patch("eftjourney.builder.game_running", return_value=True):
                code, out = self.run_auto(rt, videos=tmp, ffmpeg=tmp, model=str(model))
            self.assertEqual(code, 1)
            self.assertIn("게임이 실행 중", out)
            self.assertEqual(config.load_settings(rt.data_dir)["model"], str(model))  # the paths worked: kept

    def test_doctor_lists_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                self.assertEqual(cli.cmd_doctor(self.rt(tmp), argparse.Namespace()), 0)
            self.assertIn("영상 자동 경로", out.getvalue())
            self.assertIn("자동 경로    : 0개", out.getvalue())

    def test_parser(self):
        args = cli.build_parser().parse_args(["auto", "--raid", "a", "--raid", "b", "--model", "m.onnx"])
        self.assertEqual((args.cmd, args.raid, args.model), ("auto", ["a", "b"], "m.onnx"))


@needs_numpy
class AnalyseTest(unittest.TestCase):
    """Two Labs raids walking the same corridor, each placed by the other's frames; a third without
    screenshots placed by both. A stand-in model turns each frame's true spot into a descriptor."""

    START = datetime(2026, 11, 2, 20, 0, 0)

    @staticmethod
    def spot(raid_index, t):
        """Where raid `raid_index` is `t` s into it: along a 200 m corridor, the third one backwards."""
        x = -250.0 + (1.2 * t if raid_index < 2 else 200 - 1.2 * t) + 0.5 * raid_index
        return x, 0.0, -300.0

    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.store = RaidStore(self.root / "data")
        self.videos = self.root / "videos"
        self.videos.mkdir()
        self.raids, self.recordings = [], {}
        for i in range(3):
            start = self.START + timedelta(minutes=10 * i)
            raid_id = f"{start:%Y%m%d-%H%M%S}-laboratory"
            track = [] if i == 2 else [
                {"t": float(t), "x": self.spot(i, t)[0], "y": 0.0, "z": -300.0, "shot": f"{i}-{t}.png"}
                for t in range(0, 161, 10)]
            self.store.save_raid({"id": raid_id, "map": "laboratory", "startedAt": start.astimezone().isoformat(),
                                  "endedAt": (start + timedelta(seconds=160)).astimezone().isoformat(), "track": track})
            video_file = self.videos / f"{start:%Y-%m-%d %H-%M-%S}.mp4"
            video_file.write_bytes(b"not really a video")
            old = time.time() - 3600
            os.utime(video_file, (old, old))
            # the recording starts 5 s before the raid and runs on 20 s past it
            self.recordings[video_file.name] = (i, start - timedelta(seconds=5))
            self.raids.append(raid_id)
        self.model = self.root / "megaloc.onnx"
        self.model.write_bytes(b"stand-in")
        self.tools = config.Tools(videos=self.videos, ffmpeg=Path("ffmpeg"), ffprobe=Path("ffprobe"),
                                  model=self.model, remembered={})
        self.anchors = np.array([[-260.0 + 4 * k, -300.0] for k in range(60)])
        self.results = ["killed", None, "survived"]  # what each recording's result screens say
        self.results_read = []

    def descriptor(self, x, z):
        d = np.exp(-((self.anchors[:, 0] - x) ** 2 + (self.anchors[:, 1] - z) ** 2) / (2 * 6.0 ** 2))
        return d / np.linalg.norm(d)

    def fakes(self, described):
        test = self

        def frames(ffmpeg, path, pix_fmt, first, seconds, size, step=1.0):
            i, video_start = test.recordings[path.name]
            start = test.START + timedelta(minutes=10 * i)
            for k in range(int(seconds // step) + 1):
                t = first + k * step - (start - video_start).total_seconds()
                x, _, z = test.spot(i, t)
                yield np.array([x, z]).tobytes()

        class FakeMegaLoc:
            def __init__(self, path):
                pass

            def describe(self, frames):
                for buf in frames:
                    described.append(1)
                    x, z = np.frombuffer(buf, np.float64)
                    yield test.descriptor(x, z)[None, :].astype(np.float16)

        def read_result(ffmpeg, recording, end, duration):
            test.results_read.append(recording.name)
            i = test.recordings[recording.name][0]
            return {"status": test.results[i], "frames": 2 if test.results[i] else 0}

        return [mock.patch.object(analyse.video, "frames", frames),
                mock.patch.object(analyse.video, "probe", lambda ffprobe, p: {"duration": 185.0, "pix_fmt": "yuv420p", "tag": None}),
                mock.patch.object(analyse.video, "first_frame", lambda p, tag=None: test.recordings[p.name][1]),
                mock.patch.object(analyse.oc, "read_result", read_result),
                mock.patch.object(analyse, "MegaLoc", FakeMegaLoc)]

    def run_analysis(self, game=lambda: False, read_result=None, **kw):
        described = []
        with contextlib.ExitStack() as stack:
            for patch in self.fakes(described):
                stack.enter_context(patch)
            if read_result is not None:
                stack.enter_context(mock.patch.object(analyse.oc, "read_result", read_result))
            report = analyse.analyse(self.store, load_maps(), self.store.root, self.tools, log=lambda line: None,
                                     game=game, **kw)
        return report, len(described)

    def test_routes_from_recordings(self):
        report, described = self.run_analysis()
        self.assertEqual(report.recordings, 3)
        self.assertEqual(report.raids, 3)
        self.assertEqual(sorted(report.described), sorted(self.raids))
        self.assertEqual(described, 3 * 161)  # one frame a second over each raid's 160 s
        self.assertEqual(sorted(report.routes), sorted(self.raids))
        for raid_id in self.raids[:2]:  # their screenshots say how right they are
            self.assertGreater(report.checks[raid_id]["within"], 0.95)
        doc = self.store.load_auto(self.raids[2])  # no screenshots of its own
        self.assertNotIn("check", doc)
        self.assertEqual(sorted(doc["references"]["raids"]), self.raids[:2])
        self.assertEqual(len(doc["route"]), 161)
        errors = [abs(p["x"] - self.spot(2, p["t"])[0]) for p in doc["route"]]
        self.assertLess(sorted(errors)[len(errors) // 2], 3.0)
        self.assertEqual(doc["route"][0]["t"], 0.0)

        # a second run describes nothing again: the frames were kept, and the recordings may go
        for f in self.videos.iterdir():
            f.unlink()
        report, described = self.run_analysis()
        self.assertEqual((described, report.described, report.raids), (0, [], 3))
        self.assertEqual(sorted(report.routes), sorted(self.raids))
        self.assertEqual(self.store.load_auto(self.raids[2])["video"], "2026-11-02 20-20-00.mp4")

    def test_results_from_recordings(self):
        report, _ = self.run_analysis()
        self.assertEqual(report.outcomes, {self.raids[0]: "killed", self.raids[2]: "survived"})
        self.assertEqual(self.store.load_auto(self.raids[0])["outcome"], {"status": "killed", "source": "video"})
        self.assertNotIn("outcome", self.store.load_auto(self.raids[1]))  # its screens showed no result
        self.assertEqual(self.store.merged(self.raids[0])["outcome"], {"status": "killed", "source": "video"})
        rows = {row["id"]: row for row in self.store.summaries()}
        self.assertEqual((rows[self.raids[0]]["outcome"], rows[self.raids[1]]["outcome"]), ("killed", "unknown"))

        # read once per recording: a second run asks nobody
        self.results_read.clear()
        report, _ = self.run_analysis()
        self.assertEqual(self.results_read, [])
        self.assertEqual(report.outcomes[self.raids[2]], "survived")

        # what the user typed wins over what the recording showed
        self.store.save_edits(self.raids[0], {"outcome": {"status": "survived"}})
        self.assertEqual(self.store.merged(self.raids[0])["outcome"]["status"], "survived")

    def test_a_result_without_a_route(self):
        for raid_id in self.raids[:2]:  # no references: no routes, but the results still count
            doc = self.store.load_raid(raid_id)
            doc["track"] = []
            self.store.save_raid(doc)
        report, _ = self.run_analysis()
        self.assertEqual(report.routes, [])
        doc = self.store.load_auto(self.raids[0])
        self.assertEqual((doc["route"], doc["outcome"]["status"]), ([], "killed"))
        rows = {row["id"]: row for row in self.store.summaries()}
        self.assertEqual((rows[self.raids[0]]["outcome"], rows[self.raids[0]]["auto"]), ("killed", False))

    def test_results_wait_for_windows_ocr(self):
        report, _ = self.run_analysis(read_result=lambda *a: None)  # no Windows OCR here
        self.assertEqual((report.unread, report.outcomes), (3, {}))
        self.assertFalse((config.auto_dir(self.store.root) / "outcomes.json").exists())  # nothing to keep: tried again later
        report, _ = self.run_analysis()
        self.assertEqual(len(report.outcomes), 2)

    def test_stops_when_the_game_starts(self):
        report, described = self.run_analysis(game=lambda: True)
        self.assertIn("게임", report.stopped)
        self.assertEqual((described, report.routes), (0, []))

    def test_a_map_without_references(self):
        for raid_id in self.raids[:2]:  # no screenshots anywhere: nothing places any frame
            doc = self.store.load_raid(raid_id)
            doc["track"] = []
            self.store.save_raid(doc)
        report, _ = self.run_analysis()
        self.assertEqual(report.routes, [])
        self.assertEqual(set(report.skipped), set(self.raids))


if __name__ == "__main__":
    unittest.main()
