"""
VAD 段后处理

核心原则：不丢任何 VAD 检测到的语音段（短语气词「嗯/对/啊/卧槽/哈哈」承载语义）。
流程：合并相邻 → 短段救援（强制并入最近邻居）→ 切分超长 → 兜底过滤(<min_speech_duration)。
"""

from __future__ import annotations

from voxpen.utils.logger import get_logger

logger = get_logger("vad.segmenter")


def merge_and_split_segments(
    raw_segments: list[dict],
    min_length: float = 1.0,
    max_length: float = 30.0,
    merge_gap: float = 0.5,
    min_speech_duration: float = 0.2,
) -> list[dict]:
    """
    对 VAD 原始段做四步后处理：

    a) merge_adjacent: 相邻段间隔 < merge_gap → 合并。
    b) rescue_short_segments: 短段（< min_length）强制并入最近邻居，不丢弃。
    c) split_long_segments: 超长段（> max_length）在中点递归切分。
    d) filter_micro_segments: 仅丢弃 < min_speech_duration 的段（误触发兜底）。

    Args:
        raw_segments: VAD 原始段列表 [{"start": float, "end": float}, ...]。
        min_length: 短段救援目标长度（秒），低于此值的段强制并入邻居。
        max_length: 最大段长（秒），超过的强制切分。
        merge_gap: 合并间隔阈值（秒）。
        min_speech_duration: 最小语音时长（秒），低于此值视为误触发并丢弃。

    Returns:
        处理后的段列表，格式同输入。
    """
    if not raw_segments:
        logger.warning("输入段列表为空，跳过处理。")
        return []

    # ── a) 排序 + 合并相邻段 ──────────────────────────────
    sorted_segs = sorted(raw_segments, key=lambda s: s["start"])

    merged: list[dict] = []
    current = dict(sorted_segs[0])
    for seg in sorted_segs[1:]:
        gap = seg["start"] - current["end"]
        if gap < merge_gap:
            current["end"] = max(current["end"], seg["end"])
            logger.debug(f"  合并: gap={gap:.2f}s → [{current['start']:.1f}, {current['end']:.1f}]")
        else:
            merged.append(current)
            current = dict(seg)
    merged.append(current)

    a_count = len(merged)
    logger.debug(f"  (a) 合并相邻: {len(sorted_segs)} → {a_count}")

    # ── b) 短段救援：强制并入最近邻居 ──────────────────────
    merged = _rescue_short_segments(merged, min_length)
    b_count = len(merged)
    logger.debug(f"  (b) 短段救援: {a_count} → {b_count}")

    # ── c) 切分超长段（中点递归） ──────────────────────────
    def _split_long(seg: dict) -> list[dict]:
        duration = seg["end"] - seg["start"]
        if duration <= max_length:
            return [seg]
        mid = (seg["start"] + seg["end"]) / 2.0
        logger.debug(f"  切分: [{seg['start']:.1f}, {seg['end']:.1f}] ({duration:.1f}s) → 中点 {mid:.1f}")
        return _split_long({"start": seg["start"], "end": mid}) + _split_long({"start": mid, "end": seg["end"]})

    split: list[dict] = []
    for seg in merged:
        split.extend(_split_long(seg))

    c_count = len(split)
    logger.debug(f"  (c) 切分超长: {b_count} → {c_count}")

    # ── d) 兜底过滤：仅丢弃 < min_speech_duration ──────────
    result: list[dict] = []
    for seg in split:
        duration = seg["end"] - seg["start"]
        if duration >= min_speech_duration:
            result.append(seg)
        else:
            logger.debug(f"  丢弃微段: [{seg['start']:.2f}, {seg['end']:.2f}] ({duration:.3f}s < {min_speech_duration}s)")

    d_count = len(result)
    logger.info(
        f"段后处理: {len(raw_segments)} 原始 → "
        f"{a_count} 合并 → {b_count} 救援 → "
        f"{c_count} 切分 → {d_count} 最终"
    )
    return result


def _rescue_short_segments(
    segments: list[dict],
    min_length: float,
) -> list[dict]:
    """
    短段救援：将时长 < min_length 的段强制并入最近邻居。

    规则（按优先级）：
    1. 如果该段是唯一段：保留不动。
    2. 如果只有一侧有邻居：并入该邻居。
    3. 如果两侧都有邻居：并入 gap 更近的那个。
    4. 如果两侧 gap 相同：并入自身时长较短的那个邻居（让段长均衡）。

    迭代执行直到没有可救援的短段或无法继续合并。

    Args:
        segments: 已排序的段列表。
        min_length: 短段阈值（秒）。

    Returns:
        救援后的段列表。
    """
    if len(segments) <= 1:
        return segments

    changed = True
    while changed:
        changed = False
        segs = list(segments)

        for i in range(len(segs)):
            seg = segs[i]
            duration = seg["end"] - seg["start"]
            if duration >= min_length:
                continue

            # 确定左右邻居
            has_left = i > 0
            has_right = i < len(segs) - 1

            if not has_left and not has_right:
                # 唯一段，保留
                continue

            # 计算 gaps
            left_gap = seg["start"] - segs[i - 1]["end"] if has_left else float("inf")
            right_gap = segs[i + 1]["start"] - seg["end"] if has_right else float("inf")

            # 选择并入目标
            if has_left and not has_right:
                target_idx = i - 1
            elif has_right and not has_left:
                target_idx = i + 1
            else:
                # 两侧都有邻居
                if left_gap < right_gap:
                    target_idx = i - 1
                elif right_gap < left_gap:
                    target_idx = i + 1
                else:
                    # gap 相同 → 选自身时长较短的邻居
                    left_dur = segs[i - 1]["end"] - segs[i - 1]["start"]
                    right_dur = segs[i + 1]["end"] - segs[i + 1]["start"]
                    target_idx = i - 1 if left_dur <= right_dur else i + 1

            # 执行并入
            target = segs[target_idx]
            new_start = min(target["start"], seg["start"])
            new_end = max(target["end"], seg["end"])

            logger.debug(
                f"  救援: 短段 [{seg['start']:.2f}, {seg['end']:.2f}] "
                f"({duration:.2f}s) → 并入邻居 [{target['start']:.2f}, {target['end']:.2f}] "
                f"(left_gap={left_gap:.2f}, right_gap={right_gap:.2f})"
            )

            # 更新目标段，删除当前段
            new_segs = []
            for j, s in enumerate(segs):
                if j == target_idx:
                    new_segs.append({"start": new_start, "end": new_end})
                elif j != i:
                    new_segs.append(s)

            segments = new_segs
            changed = True
            break  # 从头开始迭代，避免索引混乱

    return segments
