"""
Phase 2.1：Qwen3-ASR 最小推理验证脚本

加载 Qwen3-ASR-1.7B 模型，对单段 WAV 执行转录，打印结果。

用法：
    python scripts/test_qwen3_asr_minimal.py <wav_path> [--context "热词"] [--language zh]

要求：
    - config/default.yaml 中 model.asr_local_path 已填本地模型路径
    - transformers 后端，不加载 ForcedAligner，不启用时间戳
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# 确保项目根目录在 sys.path 最前
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import soundfile as sf
import torch

from voxpen.config import load_config
from voxpen.utils.logger import setup_root_logger, get_logger

logger = get_logger("asr.minimal")


# ── 语言校验 ───────────────────────────────────────────────

# 从 qwen_asr 导入合法语言列表（不会触发模型加载）
try:
    from qwen_asr.inference.utils import SUPPORTED_LANGUAGES
except ImportError:
    # 备用：硬编码常用语言（以实际包导出的为准）
    SUPPORTED_LANGUAGES = [
        "Chinese", "English", "Cantonese", "Japanese", "Korean",
        "Arabic", "German", "French", "Spanish", "Portuguese",
        "Indonesian", "Italian", "Russian", "Thai", "Vietnamese",
        "Turkish", "Hindi", "Malay", "Dutch", "Swedish", "Danish",
        "Finnish", "Polish", "Czech", "Filipino", "Persian", "Greek",
        "Romanian", "Hungarian", "Macedonian",
    ]


def validate_language(lang: str | None) -> str | None:
    """校验 language 参数，非法时直接报错列出合法值。"""
    if lang is None:
        return None
    # 标准化：首字母大写
    normalized = lang.strip()[:1].upper() + lang.strip()[1:].lower()
    if normalized not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"不支持的语言: '{lang}'。\n"
            f"合法取值（首字母大写）: {', '.join(sorted(SUPPORTED_LANGUAGES))}\n"
            f"提示：请使用完整名称如 'Chinese'，不要用 'zh' / 'en' 等短码。"
        )
    return normalized


# ── dtype 转换 ─────────────────────────────────────────────

def _dtype_from_config(dtype_str: str) -> torch.dtype:
    s = dtype_str.strip().lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"不支持的 dtype: {dtype_str}")


# ── 主流程 ─────────────────────────────────────────────────

def main() -> int:
    setup_root_logger()

    parser = argparse.ArgumentParser(description="Qwen3-ASR 最小推理验证")
    parser.add_argument("wav_path", type=Path, help="输入 16kHz WAV 文件路径")
    parser.add_argument("--context", type=str, default="", help="热词/上下文（可选）")
    parser.add_argument("--language", type=str, default=None, help="强制语言，如 Chinese / English")
    args = parser.parse_args()

    wav_path = args.wav_path
    if not wav_path.exists():
        logger.error(f"WAV 文件不存在: {wav_path}")
        return 1

    # ── 校验 language ──
    language = validate_language(args.language)

    # ── 加载配置 ──
    config = load_config()
    local_path = config.model.asr_local_path
    if not local_path or not str(local_path).strip():
        raise SystemExit(
            "请先在 config/default.yaml 中填写 model.asr_local_path，\n"
            "例如: asr_local_path: \"C:/Users/mumuer/qwen-asr/Qwen3-ASR-1.7B\""
        )

    local_path = Path(local_path)
    if not local_path.exists():
        raise SystemExit(f"模型路径不存在: {local_path}")

    logger.info(f"模型路径: {local_path}")
    logger.info(f"dtype: {config.model.dtype}")
    logger.info(f"device: {config.model.device}")
    logger.info(f"context: {args.context!r}")
    logger.info(f"language: {language}")

    # ── 加载 WAV ──
    logger.info(f"读取 WAV: {wav_path}")
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    logger.info(f"  shape={wav.shape}, sr={sr}, duration={len(wav)/sr:.1f}s")

    # ── 加载模型 ──
    logger.info("加载 Qwen3-ASR 模型…")
    t0 = time.perf_counter()

    from qwen_asr import Qwen3ASRModel

    asr = Qwen3ASRModel.from_pretrained(
        str(local_path),
        dtype=_dtype_from_config(config.model.dtype),
        device_map="cuda:0",
    )
    t_load = time.perf_counter() - t0
    logger.info(f"模型加载耗时: {t_load:.1f}s")

    # ── 推理 ──
    logger.info("开始推理…")
    torch.cuda.reset_peak_memory_stats()
    t_infer_start = time.perf_counter()

    results = asr.transcribe(
        audio=(wav, sr),
        context=args.context,
        language=language,
        return_time_stamps=False,
    )

    t_infer = time.perf_counter() - t_infer_start
    peak_vram = torch.cuda.max_memory_allocated() / (1024**3)

    logger.info(f"推理耗时: {t_infer:.1f}s")
    logger.info(f"推理显存峰值: {peak_vram:.2f} GB")

    # ── 输出结果 ──
    if not results:
        logger.warning("transcribe() 返回空结果列表")
        return 1

    r = results[0]
    logger.info(f"检测语言: {r.language!r}")
    logger.info(f"转录文本: {r.text!r}")
    logger.info(f"原始返回结构 repr: {repr(r)}")

    # 额外打印文本便于肉眼阅读
    print(f"\n{'='*60}")
    print(f"语言: {r.language}")
    print(f"文本: {r.text}")
    print(f"{'='*60}")
    print(f"加载耗时: {t_load:.1f}s")
    print(f"推理耗时: {t_infer:.1f}s")
    print(f"显存峰值: {peak_vram:.2f} GB")
    print(f"{'='*60}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
