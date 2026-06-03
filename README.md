# VoxPen — Voice → Pen (🚧 Under Development)

> *Let your voice fall onto the page.*

[中文版](README_zh.md)

VoxPen is a turn-key long-form audio/video transcription tool for Windows, powered by the Qwen3-ASR model.

## Features

- 🎬 **Video & audio support**: mp4, mkv, mp3, wav, and other common formats
- 🧠 **Smart segmentation**: Silero VAD-based voice activity detection for accurate long-audio slicing
- 🗣️ **Hotword support**: provide names, domain terms, and context to boost recognition accuracy
- ⏱️ **Timestamped output**: character-level precise timestamps, SRT subtitles, and Markdown transcripts
- 🚀 **Streaming pipeline**: transcribe and display segment by segment with real-time progress
- 🪟 **Green & portable**: one-click launcher, no registry writes, no system pollution

## Development Progress

| Phase | Description | Status |
|---|---|---|
| Phase 1 | Project skeleton & infrastructure | ✅ Done |
| Phase 2.0 | Audio extraction + VAD + segment processing | ✅ Done |
| Phase 2.1–2.2 | ASR inference & encapsulation | ✅ Done |
| Phase 2.3 | Sequential pipeline + retry + cancellation | ✅ Done |
| Phase 2.4 | Post-processing (merge + formatted output) | ✅ Done |
| Phase 2.5 | Multi-source model download | ✅ Done |
| Phase 2.6 | Speaker diarization | 🚧 Up next |
| Phase 3 | Gradio UI | ⏳ Pending |
| Phase 4 | PySide6 Launcher | ⏳ Pending |
| Phase 5 | Testing & polish | ⏳ Pending |

**Low-level module test coverage: 87 unit tests — all passing**

## System Requirements

| Item | Requirement |
|---|---|
| OS | Windows 10 / 11 (64-bit) |
| Python | 3.10 |
| GPU | NVIDIA GPU, ≥ 8 GB VRAM (16 GB recommended) |
| Disk | ~2 GB tooling + ~5 GB model weights |

## Quick Start

### From Source

```bash
# 1. Install dependencies
install.bat

# 2. Launch
start.bat
```

### Pre-packaged Release

1. Download `VoxPen-*.zip` and extract
2. Double-click `VoxPen.exe` and follow the first-run setup wizard
3. Your browser opens automatically — start transcribing

## Output Formats

| Format | Use case |
|---|---|
| `.txt` | Plain text (with / without timestamps) |
| `.srt` | Video subtitles (VLC / PotPlayer / Premiere) |
| `.md` | Markdown transcript (with timestamp anchors) |

## License

The project source code is released under the MIT License.

This project is built on the [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) model. The model weights are the property of Alibaba Cloud and are subject to the Qwen official license (Apache 2.0 / Tongyi Qianwen License).

ffmpeg is distributed under the LGPL license.

## Acknowledgments

- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — ASR model
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad) — VAD model

## Developers & Tools

- **Project lead**: Nikoleta
- **Development assistance & code review**: Claude Opus 4.7 (Anthropic)
- **Code generation tool**: [Hmbown / DeepSeek-TUI](https://github.com/Hmbown/DeepSeek-TUI)
- **Code generation model**: DeepSeek-V4-Pro
