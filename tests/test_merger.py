"""
Merger 单元测试

测试后缀-前缀去重、失败段处理、时间戳偏移/修剪、language 统计。
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voxpen.pipeline import TranscribedSegment
from voxpen.postproc.merger import (
    FAILURE_PLACEHOLDER,
    MIN_OVERLAP,
    GlobalTimeStamp,
    MergedTranscription,
    _dedupe_and_join,
    _longest_common_substring_at_boundary,
    merge_transcribed_segments,
)

# ── 测试辅助 ──────────────────────────────────────────────


def _make_segment(
    index: int,
    start: float,
    end: float,
    text: str,
    language: str = "Chinese",
    time_stamps_items: list[dict] | None = None,
    failed: bool = False,
) -> TranscribedSegment:
    """构造测试用 TranscribedSegment。

    Args:
        index: 段序号。
        start: 起始秒。
        end: 结束秒。
        text: 转录文本。
        language: 语言。
        time_stamps_items: [{"text": "字", "start_time": 0.0, "end_time": 0.3}, ...]，
                          None 表示无 Aligner。
        failed: 是否失败段。
    """
    ts = None
    if time_stamps_items is not None:
        ts = SimpleNamespace(
            items=[
                SimpleNamespace(
                    text=item["text"],
                    start_time=item["start_time"],
                    end_time=item["end_time"],
                )
                for item in time_stamps_items
            ]
        )
    return TranscribedSegment(
        index=index,
        start=start,
        end=end,
        text=text,
        language=language,
        time_stamps=ts,
        failed=failed,
        retry_count=0,
        last_error="",
    )


# ── Low-level 单元测试 ────────────────────────────────────


class TestLongestCommonSubstringAtBoundary:
    """_longest_common_substring_at_boundary 函数测试。"""

    def test_exact_overlap(self):
        assert _longest_common_substring_at_boundary("hello world how", "how are you") == 3

    def test_chinese_overlap(self):
        assert _longest_common_substring_at_boundary("今天我们来聊聊", "聊聊机器学习") == 2

    def test_no_overlap(self):
        assert _longest_common_substring_at_boundary("段一文本", "段二文本") == 0

    def test_one_side_empty(self):
        assert _longest_common_substring_at_boundary("hello", "") == 0
        assert _longest_common_substring_at_boundary("", "hello") == 0

    def test_full_match(self):
        assert _longest_common_substring_at_boundary("abc", "abc") == 3

    def test_partial_head(self):
        # "how" 是 prev 后缀 也是 next 前缀
        assert _longest_common_substring_at_boundary("show how", "how now") == 3


class TestDedupeAndJoin:
    """_dedupe_and_join 函数测试。"""

    def test_no_overlap_join(self):
        new_text, new_ts = _dedupe_and_join(
            merged_text="段一文本", seg_text="段二文本",
            seg_time_stamps=None, overlap_window=10,
        )
        assert new_text == "段一文本 段二文本"

    def test_overlap_join_chinese(self):
        new_text, new_ts = _dedupe_and_join(
            merged_text="今天我们来聊聊", seg_text="聊聊机器学习",
            seg_time_stamps=None, overlap_window=10,
        )
        assert new_text == "今天我们来聊聊 机器学习"

    def test_single_char_no_dedup(self):
        """单字重叠不去重（MIN_OVERLAP=2）。"""
        new_text, new_ts = _dedupe_and_join(
            merged_text="这是测试的", seg_text="的另一个测试",
            seg_time_stamps=None, overlap_window=10,
        )
        assert new_text == "这是测试的 的另一个测试"


# ── 主函数测试 ────────────────────────────────────────────


class TestEmptyAndSingle:
    """空输入和单段输入。"""

    def test_empty_input(self):
        result = merge_transcribed_segments([])
        assert result.full_text == ""
        assert result.segments == []
        assert result.time_stamps is None
        assert result.has_failures is False
        assert result.failed_count == 0

    def test_single_segment_no_dedup(self):
        seg = _make_segment(0, 0.0, 5.0, "一段文本")
        result = merge_transcribed_segments([seg])
        assert result.full_text == "一段文本"
        assert result.language == "Chinese"


class TestBasicJoin:
    """基础拼接和去重。"""

    def test_basic_join_no_overlap(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "段一文本"),
            _make_segment(1, 3.0, 5.0, "段二文本"),
        ]
        result = merge_transcribed_segments(segs)
        assert result.full_text == "段一文本 段二文本"

    def test_overlap_dedup_chinese(self):
        segs = [
            _make_segment(0, 0.0, 3.0, "今天我们来聊聊"),
            _make_segment(1, 3.0, 6.0, "聊聊机器学习"),
        ]
        result = merge_transcribed_segments(segs)
        assert result.full_text == "今天我们来聊聊 机器学习"

    def test_overlap_dedup_english(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "hello world how", language="English"),
            _make_segment(1, 2.0, 4.0, "how are you", language="English"),
        ]
        result = merge_transcribed_segments(segs)
        assert result.full_text == "hello world how are you"

    def test_min_overlap_threshold(self):
        """单字符重叠不去重。"""
        segs = [
            _make_segment(0, 0.0, 2.0, "这是测试的"),
            _make_segment(1, 3.0, 5.0, "的另一个测试"),
        ]
        result = merge_transcribed_segments(segs)
        assert result.full_text == "这是测试的 的另一个测试"


class TestFailedSegments:
    """失败段处理。"""

    def test_failed_segment_in_middle(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "段一"),
            _make_segment(1, 3.0, 5.0, "", failed=True),
            _make_segment(2, 6.0, 8.0, "段三"),
        ]
        result = merge_transcribed_segments(segs)
        assert result.full_text == f"段一\n{FAILURE_PLACEHOLDER}\n段三"
        assert result.has_failures is True
        assert result.failed_count == 1

    def test_all_failed_segments(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "", failed=True),
            _make_segment(1, 3.0, 5.0, "", failed=True),
        ]
        result = merge_transcribed_segments(segs)
        assert FAILURE_PLACEHOLDER in result.full_text
        assert result.language == ""
        assert result.failed_count == 2

    def test_first_segment_failed(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "", failed=True),
            _make_segment(1, 3.0, 5.0, "第二段"),
        ]
        result = merge_transcribed_segments(segs)
        assert result.full_text == f"{FAILURE_PLACEHOLDER}\n第二段"
        # 不应有前导 \n
        assert not result.full_text.startswith("\n")

    def test_last_segment_failed(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "第一段"),
            _make_segment(1, 3.0, 5.0, "", failed=True),
        ]
        result = merge_transcribed_segments(segs)
        assert result.full_text == f"第一段\n{FAILURE_PLACEHOLDER}"
        # 不应有尾部 \n
        assert not result.full_text.endswith("\n")

    def test_dedup_does_not_cross_failure(self):
        """失败段是天然边界，前后段不做去重。"""
        segs = [
            _make_segment(0, 0.0, 2.0, "段一聊聊"),
            _make_segment(1, 3.0, 5.0, "", failed=True),
            _make_segment(2, 6.0, 8.0, "聊聊段三"),
        ]
        result = merge_transcribed_segments(segs)
        assert "段一聊聊" in result.full_text
        assert "聊聊段三" in result.full_text
        # "聊聊" 应保留在两处（未去重）
        assert result.full_text.count("聊聊") == 2


class TestTimeStamps:
    """时间戳处理。"""

    def test_global_time_stamp_offset(self):
        segs = [
            _make_segment(0, 0.0, 5.0, "段一文本", time_stamps_items=[
                {"text": "段", "start_time": 0.0, "end_time": 0.5},
                {"text": "一", "start_time": 0.5, "end_time": 1.0},
            ]),
            _make_segment(1, 5.0, 10.0, "段二文本", time_stamps_items=[
                {"text": "段", "start_time": 0.0, "end_time": 0.5},
                {"text": "二", "start_time": 0.5, "end_time": 1.0},
            ]),
        ]
        result = merge_transcribed_segments(segs)
        assert result.time_stamps is not None
        # 段 2 的 "段" 全局 start = 0.0 + 5.0 = 5.0
        ts_seg2_first = result.time_stamps[2]
        assert ts_seg2_first.text == "段"
        assert ts_seg2_first.start == pytest.approx(5.0)
        assert ts_seg2_first.end == pytest.approx(5.5)

    def test_time_stamp_trim_on_dedup(self):
        """去重时同步修剪时间戳。"""
        segs = [
            _make_segment(0, 0.0, 5.0, "今天我们来聊聊", time_stamps_items=[
                {"text": "今", "start_time": 0.0, "end_time": 0.3},
                {"text": "聊", "start_time": 4.5, "end_time": 4.7},
                {"text": "聊", "start_time": 4.7, "end_time": 4.9},
            ]),
            _make_segment(1, 5.0, 10.0, "聊聊机器学习", time_stamps_items=[
                {"text": "聊", "start_time": 0.0, "end_time": 0.2},
                {"text": "聊", "start_time": 0.2, "end_time": 0.4},
                {"text": "机", "start_time": 0.4, "end_time": 0.7},
            ]),
        ]
        result = merge_transcribed_segments(segs)
        assert result.time_stamps is not None
        # "聊聊" 重叠 2 字符，段 2 前 2 个时间戳应被删除
        # 剩余：段 1 的 3 个 + 段 2 的 "机"
        assert len(result.time_stamps) == 4

    def test_no_aligner_no_timestamps(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "段一"),
            _make_segment(1, 3.0, 5.0, "段二"),
        ]
        result = merge_transcribed_segments(segs)
        assert result.time_stamps is None


class TestLanguageStats:
    """language 统计。"""

    def test_mixed_language_priority(self):
        segs = [
            _make_segment(i, float(i * 3), float(i * 3 + 2), f"text_{i}", language="Chinese")
            for i in range(5)
        ]
        segs.append(_make_segment(5, 15.0, 17.0, "english text", language="English"))
        result = merge_transcribed_segments(segs)
        assert result.language == "Chinese"

    def test_all_failed_language_empty(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "", failed=True, language="Chinese"),
        ]
        result = merge_transcribed_segments(segs)
        assert result.language == ""
