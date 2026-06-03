# VoxPen — Voice → Pen（🚧 开发中...）

> *让声音落到纸上。*

[English](README.md)

VoxPen 是一个面向 Windows 用户的开箱即用长音频/视频转录工具，基于 Qwen3-ASR 模型。

## 特性

- 🎬 **视频/音频通吃**：支持 mp4、mkv、mp3、wav 等常见格式
- 🧠 **智能分段**：基于 Silero VAD 的语音活动检测，精准切分长音频
- 🗣️ **热词支持**：可输入人名、专业术语、领域背景，提升识别准确率
- ⏱️ **时间戳输出**：字符级精确时间戳，支持 SRT 字幕、Markdown 文稿
- 🚀 **流式处理**：转一段显示一段，实时进度反馈
- 🪟 **绿色便携**：Launcher 一键启动，不写注册表，不污染系统

## 开发进度

| 阶段 | 内容 | 状态 |
|---|---|---|
| Phase 1 | 项目骨架与基础设施 | ✅ 完成 |
| Phase 2.0 | 音频提取 + VAD + 段处理 | ✅ 完成 |
| Phase 2.1-2.2 | ASR 推理与封装 | ✅ 完成 |
| Phase 2.3 | 顺序流水线 + 重试 + 取消 | ✅ 完成 |
| Phase 2.4 | 后处理(合并 + 格式化输出) | ✅ 完成 |
| Phase 2.5 | 多源模型下载 | ✅ 完成 |
| Phase 2.6 | 说话人分离 | 🚧 即将开始 |
| Phase 3 | Gradio UI | ⏳ 待开始 |
| Phase 4 | PySide6 Launcher | ⏳ 待开始 |
| Phase 5 | 测试与收尾 | ⏳ 待开始 |

底层模块测试覆盖:**87 个单元测试全部通过**

## 系统要求

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11 (64-bit) |
| Python | 3.10 |
| GPU | NVIDIA 显卡，≥ 8GB 显存（推荐 16GB） |
| 磁盘 | ~2GB 工具 + ~5GB 模型权重 |

## 快速开始

### 源码方式

```bash
# 1. 安装依赖
install.bat

# 2. 启动
start.bat
```

### 整合包方式

1. 下载 VoxPen-*.zip 并解压
2. 双击 `VoxPen.exe`，跟随首次启动向导完成安装
3. 自动打开浏览器，开始使用

## 输出格式

| 格式 | 用途 |
|---|---|
| `.txt` | 纯文本（带/不带时间戳） |
| `.srt` | 视频字幕（VLC / PotPlayer / Premiere） |
| `.md` | Markdown 文稿（带时间戳锚点） |

## 许可

本项目代码采用 MIT 许可。

本项目基于 [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) 模型，模型版权归阿里云所有，使用需遵守 Qwen 官方许可（Apache 2.0 / Tongyi Qianwen License）。

ffmpeg 采用 LGPL 许可。

## 致谢

- [QwenLM/Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — ASR 模型
- [snakers4/silero-vad](https://github.com/snakers4/silero-vad) — VAD 模型

## 开发者与工具

- **项目主导**：Nikoleta
- **开发辅助与代码审核**：Claude Opus 4.7（Anthropic）
- **代码生成工具**：[Hmbown / DeepSeek-TUI](https://github.com/Hmbown/DeepSeek-TUI)
- **代码生成模型**：DeepSeek-V4-Pro