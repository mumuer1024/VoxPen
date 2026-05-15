"""
段合并与重叠去重

基于 PRD v1.4 §2.6 / §4.3.3：
- 后缀-前缀最长公共子串去重（非 LCS）
- 失败段不参与去重，前后用换行隔开
- 时间戳偏移修正 + 同步修剪
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import List, Optional

from voxpen.pipeline import TranscribedSegment
from voxpen.utils.logger import get_logger

logger = get_logger("postproc.merger")

# ── 常量 ──────────────────────────────────────────────────

FAILURE_PLACEHOLDER = "[??? 推理失败 ???]"
MIN_OVERLAP = 2  # 最小重叠字符数，低于此值不去重（避免"的"等单字巧合）

# ── 数据结构 ──────────────────────────────────────────────


@dataclass
class GlobalTimeStamp:
    """全局时间戳条目（段内时间戳偏移修正后）。"""

    text: str
    start: float  # 全局秒数
    end: float


@dataclass
class MergedTranscription:
    """整段音频合并后的转录结果。"""

    full_text: str                                       # 拼接去重后的完整文本（含失败占位）
    segments: List[TranscribedSegment]                   # 原始段列表（formatter 复用）
    time_stamps: Optional[List[GlobalTimeStamp]] = None  # 全局对齐后的字级时间戳，无 Aligner 时为 None
    has_failures: bool = False                           # 是否有失败段
    failed_count: int = 0                                # 失败段数
    language: str = ""                                   # 主语言（取出现最多的；若无成功段则为 ""）

# ── 主函数 ────────────────────────────────────────────────


def merge_transcribed_segments(
    segments: List[TranscribedSegment],
    overlap_window: int = 10,
) -> MergedTranscription:
    """
    合并 Phase 2.3 输出的段列表：

    - 相邻成功段之间做后缀-前缀最长公共子串去重（窗口大小 overlap_window 字符）
    - 失败段保留占位符，前后用换行隔开
    - 成功段之间用单个空格连接
    - 时间戳同步修剪（若存在 ForcedAlignResult）
    - 时间戳偏移修正：段内相对秒 → 全局绝对秒

    Args:
        segments: Phase 2.3 输出的 TranscribedSegment 列表（按时间排序）。
        overlap_window: 后缀-前缀匹配的窗口大小（字符数），默认 10。

    Returns:
        MergedTranscription
    """
    if not segments:
        return MergedTranscription(full_text="", segments=[])

    merged_text = ""
    merged_ts: Optional[List[GlobalTimeStamp]] = None
    has_any_ts = any(_segment_has_timestamps(s) for s in segments)

    if has_any_ts:
        merged_ts = []

    failed_count = 0

    for seg in segments:
        if seg.failed:
            failed_count += 1
            # 失败段：前后换行
            if merged_text and not merged_text.endswith("\n"):
                merged_text += "\n"
            merged_text += FAILURE_PLACEHOLDER + "\n"
            continue  # 不修改 merged_ts

        # ── 成功段：提取全局时间戳 ──
        seg_global_ts = _extract_global_timestamps(seg)

        if merged_text == "" or merged_text.endswith("\n"):
            # 首段 或 前一段是失败段 → 直接追加
            merged_text += seg.text
        else:
            # 去重拼接
            merged_text, seg_global_ts = _dedupe_and_join(
                merged_text=merged_text,
                seg_text=seg.text,
                seg_time_stamps=seg_global_ts,
                overlap_window=overlap_window,
            )

        # 累加时间戳
        if merged_ts is not None and seg_global_ts is not None:
            merged_ts.extend(seg_global_ts)

    # ── 末尾清理 ──
    merged_text = merged_text.strip("\n").strip()

    # ── language 统计 ──
    success_langs = [s.language for s in segments if not s.failed and s.language]
    if success_langs:
        language = Counter(success_langs).most_common(1)[0][0]
    else:
        language = ""

    return MergedTranscription(
        full_text=merged_text,
        segments=segments,
        time_stamps=merged_ts if has_any_ts else None,
        has_failures=failed_count > 0,
        failed_count=failed_count,
        language=language,
    )

# ── 辅助函数 ──────────────────────────────────────────────


def _longest_common_substring_at_boundary(
    prev_tail: str,
    next_head: str,
) -> int:
    """
    在 prev_tail 末尾和 next_head 开头之间找最长公共子串。

    必须满足：
      - 子串是 prev_tail 的**后缀**
      - 子串是 next_head 的**前缀**

    Returns:
        子串长度（0 表示无重叠）。
        返回值 ≥ MIN_OVERLAP 才算有效重叠，调用方自行处理阈值。
    """
    if not prev_tail or not next_head:
        return 0

    max_k = min(len(prev_tail), len(next_head))
    for k in range(max_k, 0, -1):
        if prev_tail[-k:] == next_head[:k]:
            return k
    return 0


def _dedupe_and_join(
    merged_text: str,
    seg_text: str,
    seg_time_stamps: Optional[List[GlobalTimeStamp]],
    overlap_window: int,
) -> tuple[str, Optional[List[GlobalTimeStamp]]]:
    """
    将 seg_text 拼接到 merged_text，必要时做后缀-前缀去重。

    1. 取 merged_text 末尾 overlap_window 字符 + seg_text 开头 overlap_window 字符
    2. 调用 _longest_common_substring_at_boundary 求重叠长度 k
    3. 若 k >= MIN_OVERLAP：删除 seg_text 前 k 字符 + 对应的前 k 个时间戳条目
    4. 用 " " 空格连接

    Args:
        merged_text: 已合并文本。
        seg_text: 当前段文本。
        seg_time_stamps: 当前段时间戳列表（全局偏移已修正）。
        overlap_window: 匹配窗口大小。

    Returns:
        (new_merged_text, trimmed_time_stamps)
    """
    prev_tail = merged_text[-overlap_window:] if len(merged_text) >= overlap_window else merged_text
    next_head = seg_text[:overlap_window] if len(seg_text) >= overlap_window else seg_text

    k = _longest_common_substring_at_boundary(prev_tail, next_head)

    if k >= MIN_OVERLAP:
        logger.debug(f"去重: 删除重叠 {k} 字符 '{seg_text[:k]}'")
        seg_text = seg_text[k:].lstrip()  # 清除 dedup 后残留的前导空格（如英文场景）
        if seg_time_stamps:
            seg_time_stamps = seg_time_stamps[k:]

    new_merged = merged_text + " " + seg_text if merged_text else seg_text
    return new_merged, seg_time_stamps


def _segment_has_timestamps(seg: TranscribedSegment) -> bool:
    """检查段是否有有效时间戳。"""
    ts = seg.time_stamps
    if ts is None:
        return False
    items = getattr(ts, "items", None)
    return items is not None and len(items) > 0


def _extract_global_timestamps(
    seg: TranscribedSegment,
) -> Optional[List[GlobalTimeStamp]]:
    """
    从 TranscribedSegment 提取全局时间戳。

    将段内相对秒数（ForcedAlignItem.start_time / end_time）
    加上 seg.start 偏移为全局绝对秒数。

    Returns:
        GlobalTimeStamp 列表，若无时间戳返回 None。
    """
    ts = seg.time_stamps
    if ts is None:
        return None

    items = getattr(ts, "items", None)
    if not items:
        return None

    result: List[GlobalTimeStamp] = []
    for item in items:
        result.append(GlobalTimeStamp(
            text=getattr(item, "text", ""),
            start=float(getattr(item, "start_time", 0)) + seg.start,
            end=float(getattr(item, "end_time", 0)) + seg.start,
        ))
    return result
