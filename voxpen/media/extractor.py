"""
音频提取与重采样

提供：
- 音视频类型判断（按扩展名）
- ffmpeg 音频提取 → 16kHz 单声道 PCM WAV
- WAV 时长/张量读取

依赖 voxpen.utils.ffmpeg_runner 查找 ffmpeg，
依赖 soundfile 读取 WAV。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch

from voxpen.utils.ffmpeg_runner import find_ffmpeg
from voxpen.utils.logger import get_logger

logger = get_logger("media")

# ── 格式判断 ────────────────────────────────────────────────

VIDEO_EXTENSIONS = {".mp4", ".mkv", ".mov", ".avi", ".flv", ".webm", ".wmv", ".m4v", ".ts"}
AUDIO_EXTENSIONS = {
    ".wav", ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus",
    ".wma", ".ape", ".aiff", ".au", ".ra", ".amr", ".ac3",
}


def is_video_file(path: Path) -> bool:
    """根据扩展名判断是否为视频文件。"""
    return path.suffix.lower() in VIDEO_EXTENSIONS


def is_audio_file(path: Path) -> bool:
    """根据扩展名判断是否为音频文件。"""
    return path.suffix.lower() in AUDIO_EXTENSIONS


def is_media_file(path: Path) -> bool:
    """判断是否支持的音视频文件。"""
    return is_video_file(path) or is_audio_file(path)


# ── 音频提取 ────────────────────────────────────────────────

def extract_audio_to_wav(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    mono: bool = True,
    ffmpeg_path: Optional[Path] = None,
) -> Path:
    """
    用 ffmpeg 将任意音视频提取为单声道 PCM WAV。

    ffmpeg 命令：
        ffmpeg -y -i <input> -vn -ac 1 -ar 16000 -acodec pcm_s16le <output.wav>

    Args:
        input_path: 输入文件路径。
        output_path: 输出 WAV 路径（自动创建父目录）。
        sample_rate: 目标采样率（默认 16000 Hz）。
        mono: 是否转为单声道（默认 True）。
        ffmpeg_path: ffmpeg 可执行文件路径，不传则自动查找。

    Returns:
        output_path（提取成功）。

    Raises:
        FileNotFoundError: ffmpeg 不可用，附带下载指引。
        subprocess.TimeoutExpired: 提取超时（>600s）。
        RuntimeError: ffmpeg 返回非零退出码。
        ValueError: 输入文件不存在或非媒体格式。
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"输入文件不存在: {input_path}")
    if not is_media_file(input_path):
        raise ValueError(f"不支持的媒体格式: {input_path.suffix}")

    # 找 ffmpeg
    if ffmpeg_path is None:
        ffmpeg_path = find_ffmpeg()
    if ffmpeg_path is None:
        raise FileNotFoundError(
            "ffmpeg 未找到。请将 ffmpeg.exe 放入 VoxPen/bin/ 目录，\n"
            "或安装到系统 PATH。\n"
            "下载地址：https://www.gyan.dev/ffmpeg/builds/ (推荐 ffmpeg-release-essentials.zip)"
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    channels = 1 if mono else 2
    cmd = [
        str(ffmpeg_path),
        "-y",
        "-i", str(input_path),
        "-vn",                       # 丢弃视频流
        "-ac", str(channels),
        "-ar", str(sample_rate),
        "-acodec", "pcm_s16le",
        str(output_path),
    ]

    logger.debug(f"ffmpeg: {' '.join(cmd)}")

    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=600,
    )

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        raise RuntimeError(
            f"ffmpeg 音频提取失败 (exit={result.returncode}):\n{stderr[-500:]}"
        )

    logger.info(f"音频提取完成: {output_path} ({output_path.stat().st_size / 1024:.0f} KB)")
    return output_path


# ── WAV 信息 ────────────────────────────────────────────────

def get_audio_duration(wav_path: Path) -> float:
    """
    读取 WAV 文件时长（秒），使用 soundfile 不重新解码压缩格式。

    Args:
        wav_path: WAV 文件路径。

    Returns:
        时长（秒）。

    Raises:
        FileNotFoundError: 文件不存在。
        RuntimeError: soundfile 读取失败。
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV 文件不存在: {wav_path}")

    try:
        info = sf.info(str(wav_path))
        return float(info.duration)
    except Exception as e:
        raise RuntimeError(f"无法读取 WAV 时长: {wav_path}") from e


def load_wav_as_tensor(wav_path: Path) -> tuple[torch.Tensor, int]:
    """
    加载 WAV 为 torch.Tensor（float32, shape=[samples]）。

    Args:
        wav_path: WAV 文件路径。

    Returns:
        (tensor, sample_rate)。

    Raises:
        FileNotFoundError: 文件不存在。
    """
    wav_path = Path(wav_path)
    if not wav_path.exists():
        raise FileNotFoundError(f"WAV 文件不存在: {wav_path}")

    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    tensor = torch.from_numpy(data.copy()).float()

    # 确保是 1D
    if tensor.ndim > 1:
        tensor = tensor.mean(dim=-1)

    return tensor, int(sr)
