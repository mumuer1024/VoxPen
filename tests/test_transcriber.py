"""
Transcriber 单元测试

需要真实加载模型，会消耗显存和时间。
如果环境无 GPU，跳过需要 GPU 的测试。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voxpen.asr.transcriber import Transcriber
from voxpen.asr.types import NotLoadedError, TranscriberOOMError
from voxpen.config import ModelConfig, load_config


# ── helpers ────────────────────────────────────────────────

def _has_gpu() -> bool:
    return torch.cuda.is_available()


def _make_config(**overrides) -> ModelConfig:
    """用默认值创建 ModelConfig，可部分覆盖。"""
    cfg = load_config()
    model = cfg.model
    for k, v in overrides.items():
        setattr(model, k, v)
    return model


def _load_test_wav() -> tuple[np.ndarray, int]:
    """加载 Phase 2.0 产出的 30 秒测试 WAV。"""
    wav_path = PROJECT_ROOT / "output" / "test_phase2" / "test_phase2.wav"
    if not wav_path.exists():
        pytest.skip(f"测试 WAV 不存在: {wav_path}")
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    return np.asarray(data, dtype=np.float32), int(sr)


# ── 测试用例 ───────────────────────────────────────────────

@pytest.mark.skipif(not _has_gpu(), reason="需要 GPU")
class TestLifecycle:
    """模型加载/卸载生命周期测试"""

    def test_lifecycle(self):
        cfg = _make_config()
        tc = Transcriber(cfg)

        # 初始状态
        assert tc.is_loaded() is False
        assert tc.has_aligner() is False

        # 加载
        tc.load(with_aligner=False)
        assert tc.is_loaded() is True
        assert tc.has_aligner() is False

        mem_after_load = torch.cuda.memory_allocated()

        # 幂等 no-op
        tc.load()

        # 卸载
        tc.unload()
        assert tc.is_loaded() is False
        assert tc.has_aligner() is False

        # 显存释放验证
        mem_after_unload = torch.cuda.memory_allocated()
        assert mem_after_unload < mem_after_load * 0.1, (
            f"显存应释放到 10% 以下: {mem_after_unload} >= {mem_after_load * 0.1}"
        )

        # 幂等 no-op
        tc.unload()

    def test_load_with_aligner(self):
        cfg = _make_config()
        tc = Transcriber(cfg)

        tc.load(with_aligner=True)
        assert tc.is_loaded() is True
        assert tc.has_aligner() is True

        tc.unload()
        assert tc.is_loaded() is False
        assert tc.has_aligner() is False


class TestErrorCases:
    """错误场景测试（不需要 GPU）"""

    def test_transcribe_not_loaded(self):
        cfg = _make_config()
        tc = Transcriber(cfg)
        dummy_audio = (np.zeros(16000, dtype=np.float32), 16000)

        with pytest.raises(NotLoadedError, match="未加载"):
            tc.transcribe(audio=dummy_audio)

    def test_invalid_dtype(self):
        cfg = _make_config(dtype="fp8")
        tc = Transcriber(cfg)

        with pytest.raises(ValueError, match="dtype"):
            tc.load()

    def test_invalid_path(self):
        cfg = _make_config(
            asr_local_path="C:/nonexistent/path",
            asr_model_id="",
        )
        tc = Transcriber(cfg)

        with pytest.raises(FileNotFoundError, match="路径不存在"):
            tc.load()


@pytest.mark.skipif(not _has_gpu(), reason="需要 GPU")
class TestTranscription:
    """真实推理测试（需要 GPU）"""

    def test_timestamp_without_aligner(self):
        cfg = _make_config()
        tc = Transcriber(cfg)
        tc.load(with_aligner=False)

        audio = _load_test_wav()

        with pytest.raises(ValueError, match="Aligner 未加载"):
            tc.transcribe(audio=audio, return_time_stamps=True)

        tc.unload()

    def test_real_transcription(self):
        cfg = _make_config()
        tc = Transcriber(cfg)
        tc.load(with_aligner=False)

        audio = _load_test_wav()

        result = tc.transcribe(audio=audio, language="Chinese")

        assert result.text, "转录文本不应为空"
        assert result.language == "Chinese", f"语言应为 Chinese，实际: {result.language}"

        tc.unload()
