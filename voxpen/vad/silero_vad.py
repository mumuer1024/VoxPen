"""
Silero VAD 封装

使用 silero-vad pip 包（非 torch.hub 在线下载方式），
对 16kHz 单声道 PCM 音频做语音活动检测。
"""

from __future__ import annotations

import torch

from voxpen.config import VADConfig
from voxpen.utils.logger import get_logger

logger = get_logger("vad.silero")


class SileroVAD:
    """
    Silero VAD 语音活动检测器。

    使用 silero-vad pip 包加载预训练模型。
    优先从本地缓存加载，不触发在线下载。
    """

    def __init__(self, config: VADConfig, device: str = "cuda"):
        """
        加载 Silero VAD 模型。

        Args:
            config: VAD 配置对象。
            device: 推理设备（"cuda" 或 "cpu"）。

        Raises:
            ImportError: silero-vad 未安装。
            RuntimeError: 模型加载失败。
        """
        self.config = config
        self.device = device

        try:
            from silero_vad import load_silero_vad
        except ImportError:
            raise ImportError(
                "silero-vad 未安装。请运行: pip install silero-vad>=5.1"
            )

        try:
            self._model = load_silero_vad(onnx=False)
        except Exception as e:
            raise RuntimeError(f"Silero VAD 模型加载失败: {e}") from e

        # 显式迁移到目标设备，失败时回退 CPU
        try:
            self._model = self._model.to(device)
            self.device = device
            logger.info(f"Silero VAD 模型加载成功 (device={device})")
        except Exception as e:
            logger.warning(f"Silero VAD 无法加载到 {device}，回退 CPU: {e}")
            self._model = self._model.to("cpu")
            self.device = "cpu"
            logger.info(f"Silero VAD 模型加载成功 (device=cpu, 已回退)")

    def detect(
        self,
        wav_tensor: torch.Tensor,
        sample_rate: int = 16000,
    ) -> list[dict]:
        """
        检测语音段。

        Args:
            wav_tensor: 单声道浮点 PCM (shape=[samples])。
            sample_rate: 采样率（默认 16000）。

        Returns:
            语音段列表，每段: {"start": float_seconds, "end": float_seconds}。

        Raises:
            ValueError: 音频长度不足。
        """
        from silero_vad import get_speech_timestamps

        wav_tensor = wav_tensor.float().to(self.device)

        # Silero VAD 要求输入在 [-1, 1] 范围
        peak = wav_tensor.abs().max()
        if peak > 1.0:
            wav_tensor = wav_tensor / peak

        if wav_tensor.numel() < sample_rate * 0.1:  # 不足 100ms
            raise ValueError("音频长度不足（< 100ms），无法进行 VAD 检测。")

        threshold = self.config.threshold
        logger.debug(
            f"VAD detect: samples={wav_tensor.numel()}, "
            f"sr={sample_rate}, threshold={threshold}"
        )

        segments = get_speech_timestamps(
            wav_tensor,
            self._model,
            threshold=threshold,
            sampling_rate=sample_rate,
            return_seconds=True,
        )

        result = [
            {"start": float(s["start"]), "end": float(s["end"])}
            for s in segments
        ]

        logger.info(f"VAD 检测到 {len(result)} 个原始语音段")
        return result
