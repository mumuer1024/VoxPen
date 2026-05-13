"""
Phase 2.0 自测脚本

验证"输入音视频 → 16kHz WAV → VAD 切分 → 语音段列表"完整链路。

用法:
    python scripts/test_phase2_media_vad.py <音频或视频文件路径>
    python scripts/test_phase2_media_vad.py test_audio.mp3
"""

from __future__ import annotations

import sys
from pathlib import Path

# 确保项目根目录在 sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voxpen.config import load_config
from voxpen.media.extractor import (
    extract_audio_to_wav,
    get_audio_duration,
    load_wav_as_tensor,
)
from voxpen.vad.segmenter import merge_and_split_segments
from voxpen.vad.silero_vad import SileroVAD
from voxpen.utils.logger import setup_root_logger, get_logger

logger = get_logger("test_phase2")


def main(input_path: Path) -> int:
    """
    自测主流程。

    Args:
        input_path: 输入音频或视频文件路径。

    Returns:
        0 表示成功，非 0 表示失败。
    """
    setup_root_logger()
    logger.info(f"=== Phase 2.0 自测开始 ===")
    logger.info(f"输入文件: {input_path}")

    # ── 1. 加载配置 ──
    try:
        config = load_config()
        logger.info(f"配置加载成功 (vad.threshold={config.vad.threshold})")
    except Exception as e:
        logger.error(f"配置加载失败: {e}")
        return 1

    # ── 2. 音频提取 ──
    output_dir = Path(PROJECT_ROOT) / "output" / "test_phase2"
    output_dir.mkdir(parents=True, exist_ok=True)
    wav_path = output_dir / "test_phase2.wav"

    try:
        logger.info("提取音频…")
        extract_audio_to_wav(input_path, wav_path)
        logger.info(f"WAV 文件: {wav_path}")
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.error(f"音频提取失败: {e}")
        return 1

    # ── 3. 音频信息 ──
    try:
        duration = get_audio_duration(wav_path)
        logger.info(f"音频时长: {duration:.1f} 秒 ({duration / 60:.1f} 分钟)")
    except Exception as e:
        logger.error(f"读取时长失败: {e}")
        return 1

    # ── 4. 加载为 tensor ──
    try:
        wav_tensor, sr = load_wav_as_tensor(wav_path)
        logger.info(f"加载 WAV tensor: shape={wav_tensor.shape}, sr={sr}")
    except Exception as e:
        logger.error(f"加载 WAV 失败: {e}")
        return 1

    # ── 5. VAD 检测 ──
    try:
        vad = SileroVAD(config.vad, device="cuda")
    except ImportError as e:
        logger.error(str(e))
        return 1
    except Exception as e:
        logger.warning(f"CUDA 不可用，回退到 CPU: {e}")
        try:
            vad = SileroVAD(config.vad, device="cpu")
        except Exception as e2:
            logger.error(f"VAD 模型加载失败: {e2}")
            return 1

    try:
        raw_segments = vad.detect(wav_tensor, sr)
    except Exception as e:
        logger.error(f"VAD 检测失败: {e}")
        return 1

    # ── 6. 段后处理 ──
    try:
        processed = merge_and_split_segments(
            raw_segments,
            min_length=config.vad.min_segment_length,
            max_length=config.vad.max_segment_length,
        )
    except Exception as e:
        logger.error(f"段后处理失败: {e}")
        return 1

    # ── 7. 输出结果 ──
    logger.info(f"原始段数: {len(raw_segments)}")
    logger.info(f"后处理段数: {len(processed)}")
    logger.info(f"前 {min(5, len(processed))} 段:")

    for i, seg in enumerate(processed[:5]):
        dur = seg["end"] - seg["start"]
        logger.info(
            f"  段 {i + 1}: start={seg['start']:7.2f}s  "
            f"end={seg['end']:7.2f}s  "
            f"duration={dur:5.2f}s"
        )

    logger.info(f"=== Phase 2.0 自测完成 ===")
    return 0


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"用法: python {sys.argv[0]} <音频或视频文件路径>")
        print(f"示例: python {sys.argv[0]} test_audio.mp3")
        sys.exit(1)

    input_file = Path(sys.argv[1])
    if not input_file.exists():
        print(f"错误: 文件不存在 — {input_file}")
        sys.exit(1)

    sys.exit(main(input_file))
