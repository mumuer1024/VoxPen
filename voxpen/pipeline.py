"""
VoxPen 顺序流水线

单线程顺序处理：wav → VAD 段切片 → 逐段转录 → 收集结果。
不做生产者-消费者并行（PRD v1.3 §2.5）。
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional

import numpy as np
import torch

from voxpen.asr.transcriber import Transcriber
from voxpen.asr.types import TranscriberOOMError, NotLoadedError
from voxpen.config import PipelineConfig, VADConfig
from voxpen.vad.silero_vad import SileroVAD
from voxpen.vad.segmenter import merge_and_split_segments
from voxpen.utils.logger import get_logger

logger = get_logger("pipeline")

# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class TranscribedSegment:
    """单段转录结果。"""

    index: int                          # 段序号 0..N-1
    start: float                        # 在原始音频中的起始秒
    end: float                          # 在原始音频中的结束秒
    text: str                           # 转录文本（失败时为 "[??? 推理失败 ???]"）
    language: str                       # 检测到的语言（失败时为 ""）
    time_stamps: Any | None = None      # ForcedAlignResult 或 None
    failed: bool = False                # 是否标记为失败段
    retry_count: int = 0                # 实际重试次数（0 表示首次成功）
    last_error: str = ""                # 失败段记录最后一次异常的 type+message


ProgressCallback = Callable[[TranscribedSegment, int, int], None]
"""进度回调签名：(刚完成的段, 已完成数, 总段数)"""

# ── 主函数 ────────────────────────────────────────────────


def run_pipeline(
    wav: np.ndarray,
    sr: int,
    transcriber: Transcriber,
    vad: SileroVAD,
    vad_config: VADConfig,
    pipeline_config: PipelineConfig,
    context: str = "",
    language: str | None = None,
    return_time_stamps: bool = False,
    progress_callback: ProgressCallback | None = None,
    cancel_event: threading.Event | None = None,
) -> List[TranscribedSegment]:
    """
    顺序流水线：wav → VAD 段切片 → 逐段转录 → 收集结果。

    Args:
        wav: 1-D float32 音频 ndarray。
        sr: 采样率。
        transcriber: 已加载的 Transcriber 实例。
        vad: SileroVAD 实例。
        vad_config: VAD 配置。
        pipeline_config: 流水线配置（含 retry_times）。
        context: 热词/上下文。
        language: 强制语言（None=自动检测）。
        return_time_stamps: 是否返回时间戳（需 Aligner）。
        progress_callback: 每段完成后的回调。
        cancel_event: 取消信号，set 后在下一段循环开始前退出。

    Returns:
        List[TranscribedSegment]: 转录结果列表（可能为空）。

    Raises:
        ValueError: wav 格式错误；需要时间戳但 Aligner 未加载。
        NotLoadedError: Transcriber 未加载。
        TranscriberOOMError: 推理时 OOM（不会进入重试，直接抛出）。
    """
    # ── 入口校验 ──
    if not isinstance(wav, np.ndarray) or wav.ndim != 1:
        raise ValueError(f"wav 必须是 1-D float32 ndarray，当前 ndim={getattr(wav, 'ndim', 'N/A')}")
    if wav.dtype != np.float32:
        raise ValueError(f"wav dtype 必须是 float32，当前: {wav.dtype}")
    if not isinstance(sr, int):
        raise ValueError(f"sr 必须是 int，当前: {type(sr)}")

    if not transcriber.is_loaded():
        raise NotLoadedError("Transcriber 未加载，请调用方先 load()")

    if return_time_stamps and not transcriber.has_aligner():
        raise ValueError("需要时间戳但 Aligner 未加载，请 unload() 后 load(with_aligner=True)")

    # ── VAD 阶段 ──
    logger.info(f"VAD start: wav samples={len(wav)}, sr={sr}")
    wav_tensor = torch.from_numpy(wav).float()

    raw_segments = vad.detect(wav_tensor, sr)
    segments = merge_and_split_segments(
        raw_segments,
        min_length=vad_config.min_segment_length,
        max_length=vad_config.max_segment_length,
        merge_gap=vad_config.merge_gap,
        min_speech_duration=vad_config.min_speech_duration,
    )

    if not segments:
        logger.warning("VAD 未检测到任何语音段，返回空结果。")
        return []

    total = len(segments)

    # 运行时统计
    failed_count = 0
    success_count = 0

    # ── 逐段推理 ──
    logger.info(f"推理开始: {total} 段, retry_times={pipeline_config.retry_times}")
    t_start = time.perf_counter()
    results: List[TranscribedSegment] = []

    for idx, seg in enumerate(segments):
        # 取消检查
        if cancel_event and cancel_event.is_set():
            logger.info(f"流水线在第 {idx+1}/{total} 段被取消（已完成 {len(results)} 段）")
            break

        seg_start = float(seg["start"])
        seg_end = float(seg["end"])

        # 切片
        start_sample = int(seg_start * sr)
        end_sample = int(seg_end * sr)
        wav_slice = wav[start_sample:end_sample].astype(np.float32, copy=False)

        # 推理（含重试）
        result = _transcribe_with_retry(
            transcriber=transcriber,
            wav_slice=wav_slice,
            sr=sr,
            index=idx,
            start=seg_start,
            end=seg_end,
            context=context,
            language=language,
            return_time_stamps=return_time_stamps,
            max_retries=pipeline_config.retry_times,
        )

        if result.failed:
            failed_count += 1
        else:
            success_count += 1

        results.append(result)

        # 触发回调
        if progress_callback:
            try:
                progress_callback(result, len(results), total)
            except Exception as e:
                logger.error(f"进度回调异常（已忽略）: {e}")

    elapsed = time.perf_counter() - t_start
    logger.info(
        f"推理完成: {len(results)}/{total} 段, "
        f"成功 {success_count}, 失败 {failed_count}, "
        f"耗时 {elapsed:.1f}s "
        f"(取消={cancel_event is not None and cancel_event.is_set()})"
    )

    return results


# ── 私有辅助 ──────────────────────────────────────────────


def _transcribe_with_retry(
    transcriber: Transcriber,
    wav_slice: np.ndarray,
    sr: int,
    index: int,
    start: float,
    end: float,
    context: str,
    language: str | None,
    return_time_stamps: bool,
    max_retries: int,
) -> TranscribedSegment:
    """
    对单段执行推理 + 重试。

    - 总尝试次数 = max_retries + 1（首次 + 重试）。
    - TranscriberOOMError：直接抛出，不进重试。
    - 其他异常：记录后重试。
    - 全部失败：返回占位段。

    Args:
        transcriber: Transcriber 实例。
        wav_slice: 音频切片 ndarray。
        sr: 采样率。
        index: 段序号。
        start: 起始秒。
        end: 结束秒。
        context: 热词。
        language: 语言。
        return_time_stamps: 时间戳开关。
        max_retries: 最大重试次数。

    Returns:
        TranscribedSegment（成功或失败占位）。

    Raises:
        TranscriberOOMError: OOM 不重试，直接抛出。
    """
    last_error = ""
    max_attempts = max_retries + 1

    for attempt in range(max_attempts):
        try:
            asr_result = transcriber.transcribe(
                audio=(wav_slice, sr),
                context=context,
                language=language,
                return_time_stamps=return_time_stamps,
            )
            # 成功
            seg = TranscribedSegment(
                index=index,
                start=start,
                end=end,
                text=asr_result.text or "",
                language=asr_result.language or "",
                time_stamps=getattr(asr_result, "time_stamps", None),
                failed=False,
                retry_count=attempt,
                last_error="",
            )
            if attempt > 0:
                logger.info(f"段 {index}: 第 {attempt + 1} 次尝试成功")
            return seg

        except TranscriberOOMError:
            # OOM 不进重试，直接向上抛
            raise

        except Exception as e:
            last_error = f"{type(e).__name__}: {e}"
            logger.warning(
                f"段 {index} 第 {attempt + 1}/{max_attempts} 次尝试失败: {last_error}"
            )

    # 全部失败
    logger.error(f"段 {index} 全部 {max_attempts} 次尝试均失败: {last_error}")
    return TranscribedSegment(
        index=index,
        start=start,
        end=end,
        text="[??? 推理失败 ???]",
        language="",
        time_stamps=None,
        failed=True,
        retry_count=max_retries,
        last_error=last_error,
    )
