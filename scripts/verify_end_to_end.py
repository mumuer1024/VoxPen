"""
端到端验证脚本(Phase 2 收官验证)

把所有底层模块串起来跑一遍真实音视频,产出 txt/srt/md 文件。
覆盖 Phase 2.0-2.5 全链路:extractor → VAD → segmenter → ASR → pipeline → merger → formatter。

用法:
    python scripts/verify_end_to_end.py <input_file>

示例:
    python scripts/verify_end_to_end.py "C:\\Users\\mumuer\\Desktop\\test.mp4"
    python scripts/verify_end_to_end.py output/test_phase2/test_phase2.wav
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voxpen.asr.transcriber import Transcriber
from voxpen.config import load_config
from voxpen.media.extractor import extract_audio_to_wav, is_media_file
from voxpen.pipeline import run_pipeline
from voxpen.postproc.formatter import to_md, to_srt, to_txt
from voxpen.postproc.merger import merge_transcribed_segments
from voxpen.vad.silero_vad import SileroVAD


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    input_path = Path(sys.argv[1]).resolve()
    if not input_path.exists():
        print(f"❌ 输入文件不存在: {input_path}")
        sys.exit(1)
    if not is_media_file(input_path):
        print(f"❌ 不支持的媒体格式: {input_path.suffix}")
        sys.exit(1)

    output_dir = PROJECT_ROOT / "output" / "verify_end_to_end" / input_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("端到端验证开始")
    print(f"{'=' * 60}")
    print(f"输入文件: {input_path}")
    print(f"输出目录: {output_dir}\n")

    # ── 1. 音频提取(extractor 模块) ──
    print(f"[1/6] 音频提取(任意格式 → 16kHz 单声道 WAV)...")
    t0 = time.perf_counter()
    wav_path = output_dir / f"{input_path.stem}.preprocessed.wav"
    extract_audio_to_wav(input_path, wav_path)
    print(f"      提取完成,耗时 {time.perf_counter() - t0:.1f}s\n")

    # ── 加载 WAV ──
    wav, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    wav = np.asarray(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1).astype(np.float32)
    duration = len(wav) / sr
    print(f"音频时长: {duration:.1f}s ({duration/60:.1f}min), 采样率: {sr}\n")

    cfg = load_config()

    # ── 2. 加载模型(含 Aligner,验证 srt 路径) ──
    print(f"[2/6] 加载 ASR + Aligner...")
    t0 = time.perf_counter()
    tc = Transcriber(cfg.model)
    tc.load(with_aligner=True)
    print(f"      模型加载完成,耗时 {time.perf_counter() - t0:.1f}s\n")

    # ── 3. 加载 VAD ──
    print(f"[3/6] 加载 VAD...")
    vad = SileroVAD(cfg.vad, device="cuda")
    print()

    # ── 4. 流水线 ──
    print(f"[4/6] 跑流水线(VAD + 转录,带时间戳)...")
    t0 = time.perf_counter()
    results = run_pipeline(
        wav=wav,
        sr=sr,
        transcriber=tc,
        vad=vad,
        vad_config=cfg.vad,
        pipeline_config=cfg.pipeline,
        language=None,  # 自动检测,避免强制语言导致跨语言翻译
        return_time_stamps=True,
    )
    elapsed = time.perf_counter() - t0
    rtf = elapsed / duration if duration > 0 else 0
    print(f"      流水线完成,耗时 {elapsed:.1f}s,产出 {len(results)} 段,RTF={rtf:.3f}\n")

    # ── 5. 合并 ──
    print(f"[5/6] 段合并和去重...")
    merged = merge_transcribed_segments(results)
    print(f"      full_text 长度: {len(merged.full_text)} 字符")
    print(f"      失败段数: {merged.failed_count}, 主语言: {merged.language}")
    if merged.time_stamps:
        print(f"      时间戳条目数: {len(merged.time_stamps)}\n")
    else:
        print()

    # ── 6. 格式化输出 ──
    print(f"[6/6] 格式化输出 (txt/srt/md)...")
    txt_plain = to_txt(merged, mode="plain")
    txt_timestamped = to_txt(merged, mode="timestamped")
    txt_paragraph = to_txt(merged, mode="paragraph")
    srt = to_srt(
        merged,
        max_chars_per_line=cfg.output.srt_max_chars_per_line,
        max_duration=cfg.output.srt_max_duration,
    )
    md = to_md(merged)

    stem = input_path.stem
    (output_dir / f"{stem}.plain.txt").write_text(txt_plain, encoding="utf-8")
    (output_dir / f"{stem}.timestamped.txt").write_text(txt_timestamped, encoding="utf-8")
    (output_dir / f"{stem}.paragraph.txt").write_text(txt_paragraph, encoding="utf-8")
    (output_dir / f"{stem}.srt").write_text(srt, encoding="utf-8")
    (output_dir / f"{stem}.md").write_text(md, encoding="utf-8")
    print(f"      5 个文件写入完成\n")

    # ── 卸载 ──
    tc.unload()

    print(f"{'=' * 60}")
    print(f"✅ 端到端验证完成")
    print(f"{'=' * 60}")
    print(f"产出文件:")
    for f in sorted(output_dir.glob("*")):
        size_kb = f.stat().st_size / 1024
        print(f"  - {f.name} ({size_kb:.1f} KB)")
    print()


if __name__ == "__main__":
    main()