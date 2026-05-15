"""
Formatter 单元测试

测试 txt/srt/md 三种格式输出和时间格式化工具。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from tests.test_merger import _make_segment
from voxpen.postproc.merger import (
    GlobalTimeStamp,
    MergedTranscription,
)
from voxpen.postproc.formatter import (
    STRONG_PUNCTUATIONS,
    WEAK_PUNCTUATIONS,
    _aggregate_subtitle_lines,
    _is_ascii_word,
    _format_timestamp_hms,
    _format_timestamp_srt,
    _format_srt_time_range,
    to_md,
    to_srt,
    to_txt,
)

# ── 辅助 ──────────────────────────────────────────────────


def _make_merged(
    segments: list,
    full_text: str | None = None,
    time_stamps: list[GlobalTimeStamp] | None = None,
    language: str = "Chinese",
) -> MergedTranscription:
    """构造测试用 MergedTranscription。

    Args:
        segments: TranscribedSegment 列表。
        full_text: 完整文本，不传则自动用 segments 拼接。
        time_stamps: 全局时间戳列表。
        language: 主语言。
    """
    if full_text is None:
        full_text = " ".join(s.text for s in segments if not s.failed)
    failed_count = sum(1 for s in segments if s.failed)
    return MergedTranscription(
        full_text=full_text,
        segments=segments,
        time_stamps=time_stamps,
        has_failures=failed_count > 0,
        failed_count=failed_count,
        language=language,
    )


# ── TxtFormat 测试 ────────────────────────────────────────


class TestTxtFormat:
    """to_txt 测试。"""

    def test_txt_plain(self):
        segs = [_make_segment(0, 0.0, 2.0, "你好世界")]
        merged = _make_merged(segs)
        result = to_txt(merged, "plain")
        assert result == "你好世界"

    def test_txt_timestamped_basic(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "段一"),
            _make_segment(1, 5.0, 7.0, "段二"),
            _make_segment(2, 10.0, 13.0, "段三"),
        ]
        merged = _make_merged(segs)
        result = to_txt(merged, "timestamped")
        lines = result.split("\n")
        assert "[00:00:00] 段一" in lines[0]
        assert "[00:00:05] 段二" in lines[1]
        assert "[00:00:10] 段三" in lines[2]

    def test_txt_paragraph(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "段落一"),
            _make_segment(1, 3.0, 5.0, "段落二"),
            _make_segment(2, 6.0, 8.0, "段落三"),
        ]
        merged = _make_merged(segs)
        result = to_txt(merged, "paragraph")
        assert result == "段落一\n\n段落二\n\n段落三"

    def test_txt_with_failed_segment(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "正常段"),
            _make_segment(1, 3.0, 5.0, "", failed=True),
        ]
        merged = _make_merged(segs)
        result = to_txt(merged, "timestamped")
        assert "[??? 推理失败 ???]" in result

    def test_txt_invalid_mode(self):
        merged = _make_merged([])
        with pytest.raises(ValueError, match="txt mode"):
            to_txt(merged, "xxx")  # type: ignore[arg-type]

    def test_txt_empty(self):
        merged = _make_merged([])
        assert to_txt(merged, "plain") == ""
        assert to_txt(merged, "timestamped") == ""
        assert to_txt(merged, "paragraph") == ""


# ── SrtFormat 测试 ────────────────────────────────────────


class TestSrtFormat:
    """to_srt 测试。"""

    def test_srt_no_aligner_raises(self):
        segs = [_make_segment(0, 0.0, 2.0, "文本")]
        merged = _make_merged(segs, time_stamps=None)
        with pytest.raises(ValueError, match="Aligner"):
            to_srt(merged)

    def test_srt_strong_punct_split(self):
        """强标点立即分行。"""
        ts = [
            GlobalTimeStamp("你", 0.0, 0.5),
            GlobalTimeStamp("好", 0.5, 1.0),
            GlobalTimeStamp("。", 1.0, 1.2),
            GlobalTimeStamp("世", 1.5, 2.0),
            GlobalTimeStamp("界", 2.0, 2.5),
            GlobalTimeStamp("。", 2.5, 2.7),
        ]
        merged = _make_merged([], full_text="你好。世界。", time_stamps=ts)
        result = to_srt(merged, max_chars_per_line=20, max_duration=5.0)
        # 应被强标点切成 2 条
        assert result.count("-->") == 2

    def test_srt_duration_soft_split(self):
        """超时在弱标点分行。"""
        ts = [
            GlobalTimeStamp("长", 0.0, 1.0),
            GlobalTimeStamp("文", 1.0, 2.0),
            GlobalTimeStamp("本", 2.0, 3.0),
            GlobalTimeStamp("，", 3.0, 3.5),
            GlobalTimeStamp("继", 3.5, 4.5),
            GlobalTimeStamp("续", 4.5, 5.5),
            GlobalTimeStamp("。", 5.5, 6.0),
        ]
        merged = _make_merged([], full_text="长文本，继续。", time_stamps=ts)
        result = to_srt(merged, max_chars_per_line=20, max_duration=3.0)
        # duration > 3.0s 时应在逗号处分行
        assert result.count("-->") >= 2

    def test_srt_hard_truncate(self):
        """无标点长串硬截断。"""
        ts = [GlobalTimeStamp("A" * 30, i, i + 1) for i in range(30)]
        merged = _make_merged([], full_text="A" * 30, time_stamps=ts)
        result = to_srt(merged, max_chars_per_line=10, max_duration=60.0)
        # 应被硬截断为至少 2 条
        assert result.count("-->") >= 2

    def test_srt_failed_segment_inserted(self):
        """失败段作为独立字幕条目插入。"""
        ts = [
            GlobalTimeStamp("正", 0.0, 0.5),
            GlobalTimeStamp("常", 0.5, 1.0),
            GlobalTimeStamp("。", 1.0, 1.2),
        ]
        segs = [
            _make_segment(0, 0.0, 2.0, "正常。"),
            _make_segment(1, 3.0, 5.0, "", failed=True),
        ]
        merged = _make_merged(segs, full_text="正常。", time_stamps=ts)
        result = to_srt(merged)
        assert "[??? 推理失败 ???]" in result
        # 失败段时间戳应来自 seg.start/end
        assert "00:00:03" in result

    def test_srt_format_correctness(self):
        """验证 SRT 格式符合规范。"""
        ts = [
            GlobalTimeStamp("测", 0.0, 0.5),
            GlobalTimeStamp("试", 0.5, 1.0),
            GlobalTimeStamp("。", 1.0, 1.2),
        ]
        merged = _make_merged([], full_text="测试。", time_stamps=ts)
        result = to_srt(merged)
        lines = result.split("\n")
        assert lines[0] == "1"
        assert "00:00:00,000" in lines[1]
        assert "-->" in lines[1]
        assert "00:00:01,200" in lines[1]
        assert lines[2] == "测试。"
        assert lines[3] == ""  # 空行分隔

    def test_srt_english_word_spacing(self):
        """英文词级对齐：items 间应补空格。"""
        ts = [
            GlobalTimeStamp("Hey", 0.0, 0.5),
            GlobalTimeStamp(",", 0.5, 0.6),
            GlobalTimeStamp("have", 0.7, 1.0),
            GlobalTimeStamp("you", 1.0, 1.2),
            GlobalTimeStamp("seen", 1.2, 1.5),
            GlobalTimeStamp("my", 1.5, 1.8),
            GlobalTimeStamp("gray", 1.8, 2.2),
            GlobalTimeStamp("T", 2.2, 2.3),
            GlobalTimeStamp("-", 2.3, 2.4),
            GlobalTimeStamp("shirt", 2.4, 2.8),
            GlobalTimeStamp("?", 2.8, 2.9),
        ]
        merged = _make_merged([], full_text="Hey, have you seen my gray T-shirt?", time_stamps=ts)
        result = to_srt(merged, max_chars_per_line=40, max_duration=5.0)
        # 应包含空格分隔的英文
        assert "have you seen" in result
        # 首尾不应有多余空格
        for line in result.split("\n"):
            if line and not line.startswith(("0", "1", "2")):
                stripped = line.strip()
                if stripped:
                    assert stripped == line  # 首尾无空格


# ── MdFormat 测试 ─────────────────────────────────────────


class TestMdFormat:
    """to_md 测试。"""

    def test_md_metadata_header(self):
        segs = [_make_segment(0, 0.0, 2.0, "文本")]
        merged = _make_merged(segs)
        result = to_md(merged)
        assert "转录文稿" in result
        assert "**主语言**" in result
        assert "**总段数**" in result
        assert "**生成时间**" in result
        assert "---" in result

    def test_md_segment_anchors(self):
        segs = [
            _make_segment(0, 0.0, 2.0, "第一段"),
            _make_segment(1, 5.0, 7.0, "第二段"),
        ]
        merged = _make_merged(segs)
        result = to_md(merged)
        assert "[00:00:00] 第一段" in result
        assert "[00:00:05] 第二段" in result

    def test_md_failed_segment(self):
        segs = [_make_segment(0, 0.0, 2.0, "", failed=True)]
        merged = _make_merged(segs)
        result = to_md(merged)
        assert "[??? 推理失败 ???]" in result

    def test_md_empty_input(self):
        merged = _make_merged([])
        result = to_md(merged)
        assert "转录文稿" in result
        assert "**总段数**：0" in result


# ── TimestampFormat 测试 ──────────────────────────────────


class TestTimestampFormat:
    """时间格式化工具测试。"""

    def test_format_hms(self):
        assert _format_timestamp_hms(3661.0) == "01:01:01"
        assert _format_timestamp_hms(0.0) == "00:00:00"
        assert _format_timestamp_hms(59.4) == "00:00:59"

    def test_format_srt(self):
        assert _format_timestamp_srt(0.5) == "00:00:00,500"
        assert _format_timestamp_srt(3661.123) == "01:01:01,123"

    def test_format_negative_seconds(self):
        assert _format_timestamp_hms(-5.0) == "00:00:00"
        assert _format_timestamp_srt(-5.0) == "00:00:00,000"
