"""
FFmpeg 运行器

封装 ffmpeg / ffprobe 子进程调用，提供：
- 音频提取（任意格式 → 16kHz 单声道 WAV）
- 媒体信息探测
- ffmpeg 可用性检测

在 Windows 上优先使用项目内置的 bin/ffmpeg.exe，
否则回退到系统 PATH 中的 ffmpeg。
"""

from __future__ import annotations

import json
import subprocess
import shutil
from pathlib import Path
from typing import Optional


# ── ffmpeg 路径解析 ─────────────────────────────────────────

def find_ffmpeg() -> Optional[Path]:
    """
    查找 ffmpeg 可执行文件路径。

    优先级：
    1. 项目内置 bin/ffmpeg.exe
    2. 系统 PATH 中的 ffmpeg

    Returns:
        ffmpeg 路径或 None
    """
    # 1. 项目内置
    project_root = Path(__file__).resolve().parent.parent.parent
    bundled = project_root / "bin" / "ffmpeg.exe"
    if bundled.exists():
        return bundled

    # 2. 系统 PATH
    sys_ffmpeg = shutil.which("ffmpeg")
    if sys_ffmpeg:
        return Path(sys_ffmpeg)

    return None


def find_ffprobe() -> Optional[Path]:
    """
    查找 ffprobe 可执行文件路径。
    与 find_ffmpeg 逻辑相同。
    """
    project_root = Path(__file__).resolve().parent.parent.parent
    bundled = project_root / "bin" / "ffprobe.exe"
    if bundled.exists():
        return bundled

    sys_ffprobe = shutil.which("ffprobe")
    if sys_ffprobe:
        return Path(sys_ffprobe)

    return None


def check_ffmpeg_available() -> bool:
    """
    检测 ffmpeg 是否可用。

    Returns:
        True 如果 ffmpeg -version 成功执行
    """
    ffmpeg = find_ffmpeg()
    if ffmpeg is None:
        return False
    try:
        subprocess.run(
            [str(ffmpeg), "-version"],
            capture_output=True,
            timeout=10,
            check=True,
        )
        return True
    except Exception:
        return False


# ── 音频提取 ────────────────────────────────────────────────

def extract_audio_to_wav(
    input_path: str | Path,
    output_path: str | Path,
    sample_rate: int = 16000,
    channels: int = 1,
    ffmpeg_path: Optional[str | Path] = None,
    timeout: int = 600,
) -> bool:
    """
    将视频/音频文件提取为指定格式的 WAV。

    调用 ffmpeg：
      ffmpeg -y -i <input> -ac <channels> -ar <sample_rate>
             -sample_fmt s16 -c:a pcm_s16le <output>

    Args:
        input_path: 输入文件路径（视频或音频）
        output_path: 输出 WAV 路径
        sample_rate: 采样率（默认 16000）
        channels: 声道数（默认 1 单声道）
        ffmpeg_path: ffmpeg 路径（可选，不传则自动查找）
        timeout: 子进程超时秒数

    Returns:
        True 如果成功

    Raises:
        FileNotFoundError: ffmpeg 不可用
        subprocess.TimeoutExpired: 提取超时
        RuntimeError: ffmpeg 返回非零退出码
    """
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        raise FileNotFoundError(
            "ffmpeg 未找到。请将 ffmpeg.exe 放入 bin/ 目录，或添加到系统 PATH。"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        str(ffmpeg_path),
        "-y",                      # 覆盖已有文件
        "-i", str(input_path),
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-sample_fmt", "s16",
        "-c:a", "pcm_s16le",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=timeout,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg 音频提取失败 (exit={result.returncode}):\n{stderr[-500:]}"
        )

    return output_path.exists()


# ── 媒体探测 ────────────────────────────────────────────────

def probe_media(
    input_path: str | Path,
    ffprobe_path: Optional[str | Path] = None,
) -> dict:
    """
    使用 ffprobe 探测媒体文件信息。

    返回结构：
    {
        "duration": float (秒),
        "sample_rate": int (Hz),
        "codec": str,
        "channels": int,
        "format": str,
    }

    Args:
        input_path: 媒体文件路径
        ffprobe_path: ffprobe 路径（可选）

    Returns:
        媒体信息字典

    Raises:
        FileNotFoundError: ffprobe 不可用
        RuntimeError: 探测失败
    """
    if ffprobe_path is None:
        ffprobe_path = find_ffprobe()
    if ffprobe_path is None:
        raise FileNotFoundError(
            "ffprobe 未找到。请将 ffprobe.exe 放入 bin/ 目录，或添加到系统 PATH。"
        )

    cmd = [
        str(ffprobe_path),
        "-v", "quiet",
        "-print_format", "json",
        "-show_format",
        "-show_streams",
        str(input_path),
    ]

    result = subprocess.run(cmd, capture_output=True, timeout=30)

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(f"ffprobe 探测失败:\n{stderr}")

    raw = json.loads(result.stdout.decode("utf-8"))

    # 提取音频流信息
    audio_stream = None
    for stream in raw.get("streams", []):
        if stream.get("codec_type") == "audio":
            audio_stream = stream
            break

    fmt = raw.get("format", {})

    return {
        "duration": float(fmt.get("duration", 0)),
        "sample_rate": int(audio_stream.get("sample_rate", 0)) if audio_stream else 0,
        "codec": audio_stream.get("codec_name", "unknown") if audio_stream else "unknown",
        "channels": int(audio_stream.get("channels", 0)) if audio_stream else 0,
        "format": fmt.get("format_name", "unknown"),
    }
