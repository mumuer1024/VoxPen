"""
Pipeline 单元测试

大部分测试用 FakeTranscriber / FakeVAD 避免真实加载模型，
仅 test_real_end_to_end 用真实 Transcriber（需要 GPU）。
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import soundfile as sf
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voxpen.asr.types import NotLoadedError, TranscriberOOMError
from voxpen.config import PipelineConfig, VADConfig, load_config
from voxpen.pipeline import (
    TranscribedSegment,
    run_pipeline,
)

# ── Fake 实现 ─────────────────────────────────────────────


class FakeTranscriber:
    """模拟 Transcriber 接口，用于测试 pipeline 逻辑。"""

    def __init__(
        self,
        behavior: list | None = None,
        loaded: bool = True,
        has_aligner: bool = False,
    ):
        """
        Args:
            behavior: 每个元素是 callable(audio, context, language, return_time_stamps)
                      → 返回值 或 抛异常。按调用顺序消费；消费完后默认成功返回空文本。
            loaded: is_loaded() 返回值。
            has_aligner: has_aligner() 返回值。
        """
        self._behavior = behavior or []
        self._loaded = loaded
        self._has_aligner = has_aligner
        self._call_count = 0

    def is_loaded(self) -> bool:
        return self._loaded

    def has_aligner(self) -> bool:
        return self._has_aligner

    def transcribe(
        self,
        audio,
        context="",
        language=None,
        return_time_stamps=False,
    ):
        if self._call_count < len(self._behavior):
            action = self._behavior[self._call_count]
            self._call_count += 1
            if callable(action):
                return action(audio, context, language, return_time_stamps)
            if isinstance(action, Exception):
                raise action
            return action

        self._call_count += 1
        return SimpleNamespace(
            text=f"fake_seg_{self._call_count}",
            language="Chinese",
            time_stamps=None,
        )


class FakeVAD:
    """模拟 SileroVAD，返回预设段列表。"""

    def __init__(self, segments: list | None = None):
        self._segments = segments or []

    def detect(self, wav_tensor, sr):
        return self._segments


# ── fixtures ──────────────────────────────────────────────


def _make_pipeline_config(retry_times: int = 3) -> PipelineConfig:
    return PipelineConfig(retry_times=retry_times)


def _make_vad_config() -> VADConfig:
    cfg = load_config()
    return cfg.vad


def _dummy_wav(duration_sec: float = 10.0, sr: int = 16000) -> np.ndarray:
    return np.zeros(int(duration_sec * sr), dtype=np.float32)


def _three_segments() -> list:
    """三段 VAD 输出：2s, 3s, 2s。"""
    return [
        {"start": 0.0, "end": 2.0},
        {"start": 2.5, "end": 5.5},
        {"start": 6.0, "end": 8.0},
    ]


# ── 测试用例 ──────────────────────────────────────────────


class TestBasic:
    """基础成功路径测试。"""

    def test_basic_success(self):
        fake = FakeTranscriber()
        vad = FakeVAD(_three_segments())
        vad_cfg = _make_vad_config()

        results = run_pipeline(
            wav=_dummy_wav(10, 16000),
            sr=16000,
            transcriber=fake,
            vad=vad,
            vad_config=vad_cfg,
            pipeline_config=_make_pipeline_config(),
        )

        assert len(results) == 3
        for r in results:
            assert r.failed is False
            assert r.text  # 非空

    def test_progress_callback(self):
        fake = FakeTranscriber()
        vad = FakeVAD(_three_segments())
        calls = []

        def cb(seg, done, total):
            calls.append((done, total))

        run_pipeline(
            wav=_dummy_wav(10, 16000),
            sr=16000,
            transcriber=fake,
            vad=vad,
            vad_config=_make_vad_config(),
            pipeline_config=_make_pipeline_config(),
            progress_callback=cb,
        )

        assert len(calls) == 3
        assert calls[0] == (1, 3)
        assert calls[1] == (2, 3)
        assert calls[2] == (3, 3)


class TestCancel:
    """取消机制测试。"""

    def test_cancel_before_start(self):
        fake = FakeTranscriber()
        vad = FakeVAD(_three_segments())
        cancel = threading.Event()
        cancel.set()  # 提前 set

        results = run_pipeline(
            wav=_dummy_wav(10, 16000),
            sr=16000,
            transcriber=fake,
            vad=vad,
            vad_config=_make_vad_config(),
            pipeline_config=_make_pipeline_config(),
            cancel_event=cancel,
        )

        # VAD 跑完（3 段），但第一段循环前就 break
        assert len(results) == 0

    def test_cancel_mid_pipeline(self):
        cancel = threading.Event()

        def second_behavior(audio, ctx, lang, ts):
            cancel.set()
            return SimpleNamespace(text="seg2", language="Chinese", time_stamps=None)

        fake = FakeTranscriber(behavior=[
            SimpleNamespace(text="seg1", language="Chinese", time_stamps=None),
            second_behavior,
        ])
        vad = FakeVAD([
            {"start": 0.0, "end": 2.0},
            {"start": 3.0, "end": 5.0},
            {"start": 6.0, "end": 8.0},
            {"start": 9.0, "end": 11.0},
            {"start": 12.0, "end": 14.0},
        ])

        results = run_pipeline(
            wav=_dummy_wav(15, 16000),
            sr=16000,
            transcriber=fake,
            vad=vad,
            vad_config=_make_vad_config(),
            pipeline_config=_make_pipeline_config(),
            cancel_event=cancel,
        )

        # 段 2 推理中 cancel.set()，段 3 循环开始前检查到 → 恰好 2 段
        assert len(results) == 2
        assert results[0].text == "seg1"
        assert results[1].text == "seg2"


class TestRetry:
    """重试机制测试。"""

    def test_retry_success(self):
        """第 1/2 次抛 RuntimeError，第 3 次成功。"""
        fake = FakeTranscriber(behavior=[
            RuntimeError("fail1"),
            RuntimeError("fail2"),
            SimpleNamespace(text="third_try", language="Chinese", time_stamps=None),
        ])
        vad = FakeVAD([{"start": 0.0, "end": 2.0}])
        vad_cfg = _make_vad_config()

        results = run_pipeline(
            wav=_dummy_wav(3, 16000),
            sr=16000,
            transcriber=fake,
            vad=vad,
            vad_config=vad_cfg,
            pipeline_config=_make_pipeline_config(retry_times=3),
        )

        assert len(results) == 1
        r = results[0]
        assert r.failed is False
        assert r.retry_count == 2  # 0=首次, 1=第1次重试, 2=第2次重试成功
        assert r.text == "third_try"

    def test_retry_exhausted(self):
        """连续 4 次 RuntimeError（retry_times=3，总 4 次尝试）→ 失败占位。"""
        fake = FakeTranscriber(behavior=[
            RuntimeError("e1"),
            RuntimeError("e2"),
            RuntimeError("e3"),
            RuntimeError("e4"),
        ])
        vad = FakeVAD([{"start": 0.0, "end": 2.0}])
        vad_cfg = _make_vad_config()

        results = run_pipeline(
            wav=_dummy_wav(3, 16000),
            sr=16000,
            transcriber=fake,
            vad=vad,
            vad_config=vad_cfg,
            pipeline_config=_make_pipeline_config(retry_times=3),
        )

        assert len(results) == 1
        r = results[0]
        assert r.failed is True
        assert r.text == "[??? 推理失败 ???]"
        assert r.retry_count == 3
        assert "RuntimeError" in r.last_error

    def test_oom_not_retried(self):
        """TranscriberOOMError 直接抛出，不进重试。"""
        fake = FakeTranscriber(behavior=[TranscriberOOMError("OOM!")])
        vad = FakeVAD([{"start": 0.0, "end": 2.0}])

        with pytest.raises(TranscriberOOMError):
            run_pipeline(
                wav=_dummy_wav(3, 16000),
                sr=16000,
                transcriber=fake,
                vad=vad,
                vad_config=_make_vad_config(),
                pipeline_config=_make_pipeline_config(),
            )


class TestErrors:
    """错误场景测试。"""

    def test_not_loaded_error(self):
        fake = FakeTranscriber(loaded=False)
        vad = FakeVAD([{"start": 0.0, "end": 2.0}])

        with pytest.raises(NotLoadedError):
            run_pipeline(
                wav=_dummy_wav(3, 16000),
                sr=16000,
                transcriber=fake,
                vad=vad,
                vad_config=_make_vad_config(),
                pipeline_config=_make_pipeline_config(),
            )

    def test_timestamp_without_aligner(self):
        """return_time_stamps=True 但 aligner 未加载 → 入口就抛 ValueError。"""
        fake = FakeTranscriber(loaded=True, has_aligner=False)
        # FakeVAD 不打 detonate——如果 VAD 被调了，也不会在测试路径中报错
        vad = FakeVAD([{"start": 0.0, "end": 2.0}])

        with pytest.raises(ValueError, match="Aligner 未加载"):
            run_pipeline(
                wav=_dummy_wav(3, 16000),
                sr=16000,
                transcriber=fake,
                vad=vad,
                vad_config=_make_vad_config(),
                pipeline_config=_make_pipeline_config(),
                return_time_stamps=True,
            )

    def test_empty_vad_result(self):
        fake = FakeTranscriber()
        vad = FakeVAD([])  # 空段

        results = run_pipeline(
            wav=_dummy_wav(3, 16000),
            sr=16000,
            transcriber=fake,
            vad=vad,
            vad_config=_make_vad_config(),
            pipeline_config=_make_pipeline_config(),
        )

        assert results == []

    def test_invalid_wav_ndim(self):
        fake = FakeTranscriber()
        vad = FakeVAD()

        with pytest.raises(ValueError, match="1-D"):
            run_pipeline(
                wav=np.zeros((2, 16000), dtype=np.float32),
                sr=16000,
                transcriber=fake,
                vad=vad,
                vad_config=_make_vad_config(),
                pipeline_config=_make_pipeline_config(),
            )

    def test_callback_exception_swallowed(self):
        """callback 抛异常 → pipeline 继续，不中断。"""
        fake = FakeTranscriber()
        vad = FakeVAD(_three_segments())

        def exploding_cb(seg, done, total):
            raise RuntimeError("callback boom")

        results = run_pipeline(
            wav=_dummy_wav(10, 16000),
            sr=16000,
            transcriber=fake,
            vad=vad,
            vad_config=_make_vad_config(),
            pipeline_config=_make_pipeline_config(),
            progress_callback=exploding_cb,
        )

        # 3 段全部完成（callback 异常被捕获）
        assert len(results) == 3
        for r in results:
            assert r.failed is False


@pytest.mark.skipif(not torch.cuda.is_available(), reason="需要 GPU")
class TestRealEndToEnd:
    """真实模型端到端测试。"""

    def test_real_end_to_end(self):
        from voxpen.asr.transcriber import Transcriber
        from voxpen.config import load_config
        from voxpen.vad.silero_vad import SileroVAD

        cfg = load_config()
        wav_path = PROJECT_ROOT / "output" / "test_phase2" / "test_phase2.wav"
        if not wav_path.exists():
            pytest.skip(f"测试 WAV 不存在: {wav_path}")

        wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
        wav = np.asarray(wav, dtype=np.float32)

        tc = Transcriber(cfg.model)
        tc.load(with_aligner=False)

        vad = SileroVAD(cfg.vad, device="cuda")

        results = run_pipeline(
            wav=wav,
            sr=sr,
            transcriber=tc,
            vad=vad,
            vad_config=cfg.vad,
            pipeline_config=cfg.pipeline,
            language="Chinese",
        )

        tc.unload()

        assert len(results) > 0, "应有至少一个转录段"
        for r in results:
            assert r.failed is False
            assert r.text, f"段 {r.index} 文本不应为空"
