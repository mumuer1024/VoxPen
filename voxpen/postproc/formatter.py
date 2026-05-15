"""
格式化输出模块

基于 PRD v1.4 §2.7 和 MergedTranscription 实现：
- txt（三种模式：plain / timestamped / paragraph）
- srt（字幕，依赖 Aligner 时间戳）
- md（Markdown 文稿）
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, List

from voxpen.postproc.merger import MergedTranscription, GlobalTimeStamp, FAILURE_PLACEHOLDER
from voxpen.utils.logger import get_logger

logger = get_logger("postproc.formatter")

# ── 常量 ──────────────────────────────────────────────────

STRONG_PUNCTUATIONS = set("。！？.!?;；")
WEAK_PUNCTUATIONS = set("，,、")

TxtMode = Literal["plain", "timestamped", "paragraph"]


def _is_ascii_word(s: str) -> bool:
    """判断文本片段是否是 ASCII 词（英文、数字、英文标点）。

    用于决定字幕行拼接时是否需要在 item 之间补空格：
    - 英文词级对齐：items 是单词（如 "hello"），需要空格分隔
    - 中文字级对齐：items 是单字（如 "今"），不需要空格
    """
    if not s:
        return False
    return all(ord(c) < 128 for c in s)

# ── 时间格式化工具 ────────────────────────────────────────


def _format_timestamp_hms(seconds: float) -> str:
    """秒数 → HH:MM:SS（用于 txt timestamped 模式和 md）。"""
    if seconds < 0:
        seconds = 0.0
    total_sec = round(seconds)
    s = total_sec % 60
    m = (total_sec // 60) % 60
    h = total_sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d}"


def _format_timestamp_srt(seconds: float) -> str:
    """秒数 → HH:MM:SS,mmm（SRT 标准格式，逗号分隔毫秒）。"""
    if seconds < 0:
        seconds = 0.0
    total_ms = round(seconds * 1000)
    ms = total_ms % 1000
    total_sec = total_ms // 1000
    s = total_sec % 60
    m = (total_sec // 60) % 60
    h = total_sec // 3600
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def _format_srt_time_range(start: float, end: float) -> str:
    """生成 SRT 时间范围字符串：00:00:00,000 --> 00:00:05,000。"""
    return f"{_format_timestamp_srt(start)} --> {_format_timestamp_srt(end)}"

# ── txt 输出 ──────────────────────────────────────────────


def to_txt(
    merged: MergedTranscription,
    mode: TxtMode = "plain",
) -> str:
    """
    输出 .txt 格式。

    Args:
        merged: Phase 2.4.1 输出的合并结果。
        mode:
            - "plain": 纯文稿，直接返回 merged.full_text。
            - "timestamped": 每段开头加 [HH:MM:SS] 时间戳前缀。
            - "paragraph": 每段独立成段落，段间空行。

    Returns:
        字符串（供调用方写文件）。

    Raises:
        ValueError: mode 取值非法。
    """
    if mode not in ("plain", "timestamped", "paragraph"):
        raise ValueError(f"不支持的 txt mode: {mode!r}，请使用 plain / timestamped / paragraph")

    if not merged.segments:
        return ""

    if mode == "plain":
        return merged.full_text

    if mode == "timestamped":
        lines: List[str] = []
        for seg in merged.segments:
            ts = _format_timestamp_hms(seg.start)
            text = seg.text if not seg.failed else FAILURE_PLACEHOLDER
            lines.append(f"[{ts}] {text}")
        return "\n".join(lines)

    if mode == "paragraph":
        paragraphs: List[str] = []
        for seg in merged.segments:
            paragraphs.append(seg.text if not seg.failed else FAILURE_PLACEHOLDER)
        return "\n\n".join(paragraphs)

    return ""  # unreachable

# ── srt 输出 ──────────────────────────────────────────────


def to_srt(
    merged: MergedTranscription,
    max_chars_per_line: int = 20,
    max_duration: float = 5.0,
) -> str:
    """
    输出 .srt 字幕格式。

    依赖时间戳对齐（必须有 Aligner）。无 time_stamps 直接抛 ValueError。

    字幕行聚合规则（强标点优先 → 软兜底）：
      1. 遇到强标点（。!?;；）立即分行
      2. 累积时长超过 max_duration 时，在最近的弱标点（,、，）分行
      3. 极端无标点场景：按 max_chars_per_line 强制截断

    失败段：在 srt 中保留，文本为 [??? 推理失败 ???]，时间用 segment.start/end。

    Args:
        merged: 合并结果（time_stamps 必须非 None）。
        max_chars_per_line: 兜底硬截断字符数。
        max_duration: 触发软兜底的最大字幕条目时长（秒）。

    Returns:
        SRT 格式字符串。

    Raises:
        ValueError: time_stamps 为 None（未加载 Aligner）。
    """
    if merged.time_stamps is None:
        raise ValueError("SRT 输出需要时间戳对齐，请先加载 Aligner 后重新转录")

    # 聚合字幕行（仅成功段时间戳）
    lines = _aggregate_subtitle_lines(merged.time_stamps, max_chars_per_line, max_duration)

    # 插入失败段为独立字幕条目
    entries: List[tuple[float, float, str]] = list(lines)
    for seg in merged.segments:
        if seg.failed:
            entries.append((seg.start, seg.end, FAILURE_PLACEHOLDER))

    # 按 start 时间排序
    entries.sort(key=lambda e: e[0])

    # 生成 SRT
    result_parts: List[str] = []
    for idx, (start, end, text) in enumerate(entries, 1):
        result_parts.append(str(idx))
        result_parts.append(_format_srt_time_range(start, end))
        result_parts.append(text)
        result_parts.append("")  # 空行分隔

    return "\n".join(result_parts)


def _aggregate_subtitle_lines(
    time_stamps: List[GlobalTimeStamp],
    max_chars_per_line: int,
    max_duration: float,
) -> List[tuple[float, float, str]]:
    """
    按"强标点优先 → 软兜底"规则聚合字幕行。

    Args:
        time_stamps: 全局时间戳列表（按时间排序）。
        max_chars_per_line: 兜底硬截断字符数。
        max_duration: 最大字幕条时长（秒）。

    Returns:
        字幕行列表，每个元素为 (start_sec, end_sec, text)。
    """
    result: List[tuple[float, float, str]] = []
    buf_items: List[GlobalTimeStamp] = []  # 当前缓冲的时间戳条目
    buf_chars: List[str] = []              # 对应的字符列表（与 buf_items 一一对应）

    def _flush() -> None:
        """输出当前缓冲为一条字幕行。"""
        if buf_items:
            line_start = buf_items[0].start
            line_end = buf_items[-1].end
            line_text = "".join(buf_chars).strip()
            result.append((line_start, line_end, line_text))
            buf_items.clear()
            buf_chars.clear()

    for item in time_stamps:
        buf_items.append(item)

        # ASCII 词间补空格（英文词级对齐场景）
        if buf_chars and _is_ascii_word(item.text) and _is_ascii_word(buf_chars[-1]):
            buf_chars.append(" " + item.text)
        else:
            buf_chars.append(item.text)

        full_text = "".join(buf_chars)

        # 检查 1: 强标点立即分行
        if item.text and item.text[-1] in STRONG_PUNCTUATIONS:
            _flush()
            continue

        duration = buf_items[-1].end - buf_items[0].start

        # 检查 2: 超时软兜底
        if duration > max_duration:
            # 在缓冲区文本中回找最后一个弱标点位置
            last_weak = -1
            for i in range(len(full_text) - 1, -1, -1):
                if full_text[i] in WEAK_PUNCTUATIONS:
                    last_weak = i
                    break

            if last_weak > 0:
                # 在弱标点处切分：前半 flush，后半留下
                # 计算切分位置对应的时间戳索引
                char_count = 0
                split_idx = 0
                for j, ch in enumerate(buf_chars):
                    char_count += len(ch)
                    if char_count > last_weak:
                        split_idx = j + 1  # 包含弱标点本身
                        break

                # flush 前半
                if split_idx > 0:
                    keep_items = buf_items[split_idx:]
                    keep_chars = buf_chars[split_idx:]
                    buf_items = buf_items[:split_idx]
                    buf_chars = buf_chars[:split_idx]
                    _flush()
                    buf_items = keep_items
                    buf_chars = keep_chars
                else:
                    # 极端情况：第一个字符就是弱标点 → 不切
                    pass
            else:
                # 无弱标点 → 硬截断
                _hard_truncate(buf_items, buf_chars, max_chars_per_line, result)

        # 检查 3: 超长硬截断
        if len(full_text) > max_chars_per_line:
            _hard_truncate(buf_items, buf_chars, max_chars_per_line, result)

    # 循环结束，flush 剩余
    _flush()
    return result


def _hard_truncate(
    buf_items: List[GlobalTimeStamp],
    buf_chars: List[str],
    max_chars: int,
    result: List[tuple[float, float, str]],
) -> None:
    """
    按 max_chars 强制截断缓冲。

    截断后前半 flush 到 result，后半保留在 buf_items/buf_chars 中。
    """
    full_text = "".join(buf_chars)
    if len(full_text) <= max_chars:
        return

    # 找截断点对应的时间戳索引
    char_count = 0
    split_idx = len(buf_items)
    for j, ch in enumerate(buf_chars):
        char_count += len(ch)
        if char_count >= max_chars:
            split_idx = j + 1
            break

    if split_idx >= len(buf_items):
        return  # 全部保留

    # flush 前半
    flush_start = buf_items[0].start
    flush_end = buf_items[split_idx - 1].end
    flush_text = "".join(buf_chars[:split_idx])
    result.append((flush_start, flush_end, flush_text))

    # 后半留存
    remaining_items = buf_items[split_idx:]
    remaining_chars = buf_chars[split_idx:]
    buf_items.clear()
    buf_chars.clear()
    buf_items.extend(remaining_items)
    buf_chars.extend(remaining_chars)

# ── md 输出 ──────────────────────────────────────────────


def to_md(
    merged: MergedTranscription,
) -> str:
    """
    输出 .md 格式，带段级时间戳锚点。

    格式：
        # 转录文稿

        **主语言**：Chinese
        **总段数**：6
        **生成时间**：2026-05-14 18:23:43

        ---

        [00:00:00] 第一段文本

        [00:00:05] 第二段文本

        [00:00:09] [??? 推理失败 ???]

    Args:
        merged: 合并结果。

    Returns:
        Markdown 字符串。
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    total = len(merged.segments)

    header = (
        f"# 转录文稿\n\n"
        f"**主语言**：{merged.language or '未知'}\n"
        f"**总段数**：{total}\n"
        f"**生成时间**：{now}\n\n"
        f"---\n"
    )

    if not merged.segments:
        return header + "\n"

    body_parts: List[str] = []
    for seg in merged.segments:
        ts = _format_timestamp_hms(seg.start)
        text = seg.text if not seg.failed else FAILURE_PLACEHOLDER
        body_parts.append(f"[{ts}] {text}")

    return header + "\n\n".join(body_parts) + "\n"
