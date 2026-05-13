"""临时调试：查看 VAD 原始段 + 不同 min_length 阈值下的段分布"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from voxpen.config import load_config
from voxpen.media.extractor import extract_audio_to_wav, load_wav_as_tensor, get_audio_duration
from voxpen.vad.silero_vad import SileroVAD
from voxpen.vad.segmenter import merge_and_split_segments

input_path = Path(r"G:\AItools\VoxPen\test.mp3")
wav_path = Path(r"G:\AItools\VoxPen\output\test_phase2\test_phase2.wav")

if not wav_path.exists():
    extract_audio_to_wav(input_path, wav_path)

cfg = load_config()
duration = get_audio_duration(wav_path)
tensor, sr = load_wav_as_tensor(wav_path)
print(f"\n=== 音频时长: {duration:.2f}s ===\n")

vad = SileroVAD(cfg.vad, device="cpu")
raw = vad.detect(tensor, sample_rate=sr)

print(f"--- VAD 原始 {len(raw)} 段 ---")
for i, s in enumerate(raw, 1):
    d = s["end"] - s["start"]
    print(f"  {i:2d}: [{s['start']:6.2f}, {s['end']:6.2f}]  dur={d:5.2f}s")

total_speech = sum(s["end"] - s["start"] for s in raw)
print(f"\n总语音时长: {total_speech:.2f}s / {duration:.2f}s ({total_speech/duration*100:.1f}%)")

print("\n--- 不同 min_length 下最终段数（旧策略：直接丢弃）---")
for ml in [0.5, 1.0, 1.5, 2.0]:
    result = merge_and_split_segments(raw, min_length=ml, max_length=30.0, merge_gap=0.5)
    kept = sum(s["end"] - s["start"] for s in result)
    print(f"  min_length={ml}s → {len(result)} 段，保留语音 {kept:.2f}s ({kept/total_speech*100:.1f}%)")

print("\n--- 不同 merge_gap 下（min_length=1.0）---")
for mg in [0.3, 0.5, 0.8, 1.2]:
    result = merge_and_split_segments(raw, min_length=1.0, max_length=30.0, merge_gap=mg)
    kept = sum(s["end"] - s["start"] for s in result)
    print(f"  merge_gap={mg}s → {len(result)} 段，保留语音 {kept:.2f}s ({kept/total_speech*100:.1f}%)")

print("\n--- 新策略（短段救援）下的最终段分布 ---")
print(f"  min_segment_length=1.0s, merge_gap=0.5s, min_speech_duration=0.2s")
result = merge_and_split_segments(raw, min_length=1.0, max_length=30.0, merge_gap=0.5, min_speech_duration=0.2)
kept = sum(s["end"] - s["start"] for s in result)
print(f"  最终段数: {len(result)}，保留语音: {kept:.2f}s ({kept/total_speech*100:.1f}%)")
print()
for i, s in enumerate(result, 1):
    d = s["end"] - s["start"]
    flag = " ⚡短" if d < 0.5 else (" ◀救援" if d < 1.0 else "")
    print(f"  {i:2d}: [{s['start']:6.2f}, {s['end']:6.2f}]  dur={d:5.2f}s{flag}")
print(f"\n  丢弃的微段（<0.2s 的误触发）:")
dropped = 0
for s in raw:
    d = s["end"] - s["start"]
    if d < 0.2:
        print(f"      [{s['start']:6.2f}, {s['end']:6.2f}]  dur={d:5.2f}s")
        dropped += 1
if dropped == 0:
    print("      （无）")
