"""
Qwen3-ASR Transcriber 封装

完整的模型生命周期管理：load / unload / is_loaded / has_aligner / transcribe。
基于 Phase 2.1 实测验证通过的 qwen_asr 接口。

设计原则：
- 显式加载/卸载（用户手动管理），不做单例/自动重载
- OOM 直接报错，不做自动降级重试
- Aligner 由调用方传 with_aligner 参数决定
"""

from __future__ import annotations

import gc
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from qwen_asr import Qwen3ASRModel

from voxpen.asr.types import NotLoadedError, TranscriberOOMError
from voxpen.config import ModelConfig
from voxpen.utils.logger import get_logger

logger = get_logger("asr.transcriber")


class Transcriber:
    """
    Qwen3-ASR 转录器。

    用法：
        tc = Transcriber(config.model)
        tc.load()
        result = tc.transcribe(audio=(wav_ndarray, sr), language="Chinese")
        tc.unload()
    """

    def __init__(self, config: ModelConfig) -> None:
        """
        初始化 Transcriber。

        Args:
            config: 模型配置（来自 VoxPenConfig.model）。

        注意：__init__ 不做路径校验、不加载模型。
        """
        self._config = config
        self._model: Optional[Qwen3ASRModel] = None
        self._has_aligner: bool = False

    # ── 生命周期 ──────────────────────────────────────────

    def load(self, with_aligner: bool = False) -> None:
        """
        加载 Qwen3-ASR 模型（及可选的 ForcedAligner）。

        - 已加载时直接返回（幂等 no-op）。
        - 路径优先级：asr_local_path > asr_model_id（Aligner 同理）。
        - dtype 从配置转换（"bf16"→bfloat16, "fp16"→float16）。

        Args:
            with_aligner: 是否同时加载 ForcedAligner 模块。

        Raises:
            FileNotFoundError: 本地路径不存在或缺少 config.json。
            ValueError: 不支持的 dtype。
            RuntimeError: 模型加载失败。
        """
        if self.is_loaded():
            return  # 幂等

        # ── 解析 ASR 路径 ──
        asr_path = self._resolve_path(
            local=self._config.asr_local_path,
            remote=self._config.asr_model_id,
            label="ASR",
        )

        # ── dtype 转换 ──
        torch_dtype = self._dtype_to_torch(self._config.dtype)

        # ── 构建 kwargs ──
        kwargs = dict(
            dtype=torch_dtype,
            device_map=self._config.device,
            max_new_tokens=512,
        )

        # ── Aligner ──
        if with_aligner:
            aligner_path = self._resolve_path(
                local=self._config.aligner_local_path,
                remote=self._config.aligner_model_id,
                label="Aligner",
            )
            kwargs["forced_aligner"] = aligner_path
            kwargs["forced_aligner_kwargs"] = dict(
                dtype=torch_dtype,
                device_map=self._config.device,
            )
            logger.info(f"将同时加载 ForcedAligner: {aligner_path}")

        # ── 加载 ──
        logger.info(f"加载模型: {asr_path} (dtype={self._config.dtype}, device={self._config.device})")
        t0 = time.perf_counter()

        try:
            self._model = Qwen3ASRModel.from_pretrained(asr_path, **kwargs)
        except Exception as e:
            raise RuntimeError(f"模型加载失败: {e}") from e

        elapsed = time.perf_counter() - t0
        self._has_aligner = with_aligner

        vram_mb = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        logger.info(f"模型加载完成 ({elapsed:.1f}s, 显存占用 {vram_mb:.0f} MB, aligner={with_aligner})")

    def unload(self) -> None:
        """
        卸载模型，释放显存。

        - 未加载时直接返回（幂等 no-op）。
        - 执行：del model → empty_cache → gc.collect。
        """
        if not self.is_loaded():
            return  # 幂等

        mem_before = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0

        del self._model
        self._model = None
        self._has_aligner = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        mem_after = torch.cuda.memory_allocated() / (1024**2) if torch.cuda.is_available() else 0
        logger.info(f"模型已卸载 (显存: {mem_before:.0f} → {mem_after:.0f} MB)")

    def is_loaded(self) -> bool:
        """模型是否已加载。"""
        return self._model is not None

    def has_aligner(self) -> bool:
        """当前加载的模型是否包含 ForcedAligner。"""
        return self._has_aligner

    # ── 推理 ──────────────────────────────────────────────

    def transcribe(
        self,
        audio: tuple[np.ndarray, int],
        context: str = "",
        language: str | None = None,
        return_time_stamps: bool = False,
    ) -> "ASRTranscription":
        """
        对单段音频执行转录。

        Args:
            audio: (waveform_float32_1d, sample_rate) 元组。
            context: 热词/上下文（传给 ASR 模型）。
            language: 强制语言（None=自动检测），如 "Chinese"。
            return_time_stamps: 是否返回时间戳（需要 Aligner）。

        Returns:
            ASRTranscription: 转录结果（含 language / text / time_stamps）。

        Raises:
            NotLoadedError: 模型未加载。
            ValueError: 需要时间戳但 Aligner 未加载；audio 格式非法。
            TranscriberOOMError: CUDA OOM（附段长和显存诊断）。
        """
        if not self.is_loaded():
            raise NotLoadedError("Transcriber 未加载，请先调用 load()")

        if return_time_stamps and not self.has_aligner():
            raise ValueError(
                "需要时间戳但 Aligner 未加载，请 unload() 后 load(with_aligner=True)"
            )

        # audio 校验
        if not isinstance(audio, tuple) or len(audio) != 2:
            raise ValueError(f"audio 必须是 (np.ndarray, int) 元组，收到: {type(audio)}")
        wav_arr, sr = audio
        if not isinstance(wav_arr, np.ndarray):
            raise ValueError(f"audio[0] 必须是 np.ndarray，收到: {type(wav_arr)}")
        if wav_arr.ndim != 1:
            raise ValueError(f"audio[0] 必须是 1-D 数组，ndim={wav_arr.ndim}")
        if wav_arr.dtype != np.float32:
            raise ValueError(f"audio[0] dtype 必须是 float32，收到: {wav_arr.dtype}")
        if not isinstance(sr, int):
            raise ValueError(f"audio[1] 必须是 int，收到: {type(sr)}")

        try:
            results = self._model.transcribe(
                audio=audio,
                context=context,
                language=language,
                return_time_stamps=return_time_stamps,
            )
        except torch.cuda.OutOfMemoryError as e:
            duration = len(wav_arr) / sr
            raise TranscriberOOMError(
                f"OOM: 段长 {duration:.1f}s, "
                f"已分配 {torch.cuda.memory_allocated() / 1e9:.2f}GB, "
                f"峰值 {torch.cuda.max_memory_allocated() / 1e9:.2f}GB"
            ) from e

        # qwen_asr 总返回 list，单条输入取第 0 个
        return results[0]

    # ── 内部工具 ──────────────────────────────────────────

    @staticmethod
    def _resolve_path(local: str | None, remote: str, label: str) -> str:
        """解析模型路径：本地优先，否则用 remote ID。"""
        if local and local.strip():
            p = Path(local)
            if not p.exists():
                raise FileNotFoundError(f"{label} 模型路径不存在: {p}")
            if not (p / "config.json").exists():
                raise FileNotFoundError(f"{label} 模型路径无效（缺少 config.json）: {p}")
            return str(p)
        return str(remote)

    @staticmethod
    def _dtype_to_torch(dtype_str: str) -> torch.dtype:
        """将配置字符串转为 torch.dtype。"""
        s = dtype_str.strip().lower()
        if s in ("bf16", "bfloat16"):
            return torch.bfloat16
        if s in ("fp16", "float16", "half"):
            return torch.float16
        raise ValueError(f"不支持的 dtype: {dtype_str}，请使用 bf16 或 fp16")
