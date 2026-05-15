# VoxPen — 产品需求文档（PRD）v1.5

> Voice → Pen，让声音落到纸上。
> 一个面向 Windows 用户的开箱即用的长音频/视频转录工具，基于 Qwen3-ASR 模型。

---

## 0. 文档信息

- 版本：v1.5
- 状态：待开发（v1 MVP）
- 更新日期：2026-05-14
- 目标平台：Windows 10 / 11 (64-bit) 原生环境
- 文档用途：交付给 Claude Code（或其他 AI 编程助手）作为完整开发依据

---

## 1. 项目概述

### 1.1 背景

阿里发布的 Qwen3-ASR 模型在中文/多语种语音识别上表现优秀，但官方仅提供基础 demo（`qwen-asr-demo` CLI），存在以下痛点：

- 显存受限：长音频/视频无法直接处理，会 OOM
- 无图形界面：非技术用户无法使用
- 无视频支持：必须用户自行用 ffmpeg 提取音频
- 无字幕生成：只输出纯文本，无法直接用于视频后期
- 无热词支持的便捷入口：专业术语识别准确率受限

### 1.2 产品定位

**VoxPen 是 Qwen3-ASR 的"上层应用层"**，把模型能力包装成普通用户能用、长音频也能跑、视频也能转、能直接出字幕的桌面工具。

### 1.3 目标用户

- **主要用户（小白）**：内容创作者、学生、研究人员、纪录片爱好者。需求是"丢个文件进去就出结果"
- **次要用户（开发者）**：需要结构化转录结果做二次开发的工程师

### 1.4 运行环境

| 项 | 要求 |
|---|---|
| 操作系统 | Windows 10 / 11 (64-bit) |
| Python | 3.10（与已跑通环境一致） |
| GPU | NVIDIA 显卡，建议 ≥ 8GB 显存（开发基准：RTX 5060 Ti 16GB） |
| 显卡驱动 | 支持 CUDA 12.x |
| 磁盘空间 | 工具本体约 2GB，模型权重额外约 5GB |

### 1.5 不在范围内（Non-Goals）

- ❌ Linux / macOS 支持（v1 不做，未来视情况）
- ❌ 实时麦克风转录（v1 不做，纯离线文件转录）
- ❌ 翻译功能（v2 考虑）
- ❌ vLLM 后端（明确只做 transformers 后端）
- ❌ 说话人 enrollment / voiceprint（用户提供参考音频识别真实姓名）—— v2 考虑

---

## 2. 核心功能需求

### 2.1 输入支持

**视频格式**：mp4, mkv, avi, mov, flv, webm
**音频格式**：mp3, wav, flac, m4a, aac, ogg, wma
**输入方式**：Gradio 界面拖拽 / 点击上传

**v1 限制**：
- 单文件转录（批量队列留给 v2）
- 文件大小无硬限制（受磁盘空间约束）

### 2.2 音频提取

- 使用项目内置的 `bin/ffmpeg.exe`（用户零配置）
- 统一转为 16kHz 单声道 WAV，作为后续处理的标准格式
- 启动时检测 ffmpeg 可用性，缺失或损坏时给出明确错误提示并提供"重新下载"按钮

### 2.3 智能切分（VAD）

**目的**：解决长音频显存爆炸问题，把长音频切成模型可处理的短段。

**核心原则**：**不丢任何 VAD 检测到的语音段**——短语气词（"嗯/对/啊/卧槽/哈哈"）在通用转录场景下承载情绪和语义信号，必须保留。

**技术选型**：Silero VAD（轻量、纯 PyTorch、无系统依赖）

**后处理流水线**：合并相邻 → 短段救援 → 切分超长 → 兜底过滤

**默认参数**：
- `threshold`: 0.5（Silero VAD 语音概率阈值）
- `min_speech_duration`: 0.2 秒（低于此长度视为 VAD 误触发，**唯一的丢弃环节**）
- `min_segment_length`: 1.0 秒（短段救援的目标长度：短于 1s 的段强制并入最近邻居，不丢弃）
- `merge_gap`: 0.5 秒（相邻段间隔小于此值则合并）
- `max_segment_length`: 30 秒（超过此长度的段在中点递归切分）
- `overlap`: 0.3 秒（段边界前后各扩展 overlap/2 秒，形成人为重叠区，供后续后缀-前缀子串去重使用（详见 §4.3.3））

**短段救援规则**：短于 `min_segment_length` 的段，计算其到左右邻居的 gap，强制并入 gap 更近（或时长更短）的邻居。唯一段不处理，边界段（仅一侧有邻居）直接并入相邻段。迭代直到无短段可救援。

所有参数在界面"高级设置"中可调。

### 2.4 模型推理

**后端**：transformers（**唯一后端，明确不支持 vLLM**）

**模型加载**：
- 单例模式，应用启动时加载一次，常驻显存
- 避免每次请求都重新加载

**精度**：
- 默认 bf16
- 界面提供 fp16 / bf16 切换（兼容老显卡）

**热词/上下文**（重点功能）：
- 界面提供多行文本框，用户可输入：
  - 人名（如"张三、李四"）
  - 专业术语（如"transformer、attention"）
  - 领域背景描述（如"这是一段关于深度学习的讲座"）
- 转录时作为 `context` 参数传入 ASR 模型

**时间戳**：
- Qwen3-ASR 本身不输出时间戳，**所有时间戳输出依赖 ForcedAligner 模块**
- 时间戳粒度：中文逐字、英文/空格分隔语言逐词，精度毫秒级
- 用户在 UI 上勾选「时间戳对齐器」时启用（详见下方 ForcedAligner 章节）

**ForcedAligner（时间戳和字幕功能的必要组件）**：
- Qwen3-ForcedAligner-0.6B 提供字符级时间对齐
- **生成 SRT 字幕、Markdown 时间戳锚点必须加载此模块**
- 默认不加载（节省显存）；UI 上勾选「输出 srt」或「输出 md（带时间戳）」时自动联动勾选「加载 Aligner」，反之亦然
- 用户可手动取消 Aligner 加载，此时 srt / md 输出选项会同步禁用
- 代价：多占约 1.5GB 显存、推理耗时增加约 20%

**语言**：
- 默认自动检测
- 界面可手动指定（中/英/日/韩等 Qwen3-ASR 支持的所有语种）

### 2.5 顺序流水线

**架构**：单线程顺序处理。VAD 一次性完成全音频切分，得到 N 个段时间戳；流水线逐段切片 → 转录 → 收集结果。

**流程**：
```
[16kHz mono wav]
    ↓
VAD 检测 + 段后处理（详见 §2.3）
    ↓
[seg₁, seg₂, ..., segₙ]  # 段时间戳列表
    ↓
对每段：
    - 切片：wav[start*sr : end*sr]
    - Transcriber.transcribe(...)
    - 收集结果 / 处理失败重试 / 触发进度回调
    - 检查取消信号
    ↓
[List[TranscribedSegment]]
```

**不采用生产者-消费者并行的原因**：
- VAD 是一次性预处理（不是流式产出），不存在"持续生产"
- 音频切片成本接近零（numpy 切片），与 GPU 推理重叠的收益不到 1%
- 引入并发会增加 bug 风险（特别是取消机制和异常传播）

**未来若需要并行优化**：可在 Phase 3+ 单独引入，本 v1 版本不实现。

**关键能力**（流水线必须实现）：
- 进度回调钩子：每段完成后回调，UI 通过此接口实现"转一段显示一段"的流式体验
- 取消信号：基于 `threading.Event`，每段循环开始时检查，确保 3 秒内响应（详见 §4.3.4）
- 单段失败重试：详见 §2.8

### 2.6 后处理与合并

- **段间拼接**：将各段文本按时间顺序合并
- **重叠去重**：对相邻段在边界附近做**后缀-前缀最长公共子串（Longest Common Substring）**匹配，删除重复部分。匹配窗口默认取段尾/段头各 10 字符，可配置
- 与 LCS（最长公共子序列）的区别：此处用的是连续子串匹配，只识别真正重叠的边界，不会误删非重叠区的偶然重合字符，在 ASR 短重叠（默认 0.3s）场景下更稳定
- **时间戳全局对齐**：段内相对时间 → 全局绝对时间
- **字幕行聚合**：按标点（。！？，、）和最大字符数聚合为字幕行

### 2.7 输出格式

| 格式 | 用途 | 关键参数 |
|---|---|---|
| `.txt` | 纯文本 | "带时间戳" / "纯文稿" / "分段落"三选一 |
| `.srt` | 视频字幕 | 每行最大字符数（默认 20）、每条最长时长（默认 5 秒） |
| `.md` | Markdown 文稿 | 带 `[HH:MM:SS]` 时间戳锚点，方便阅读和跳转 |

**输出格式联动规则**：
- `.srt`、`.md` 输出依赖 ForcedAligner，UI 上勾选这两项会自动加载 Aligner
- `.txt` 不依赖 Aligner，可以在纯文本模式下工作
- 未加载 Aligner 时勾选 srt/md：UI 提示"将同时加载 ForcedAligner（+1.5GB 显存）"，确认后联动

**输出位置**：`output/<原文件名>_<时间戳>/`，界面提供下载按钮，支持一键打开输出目录。

### 2.8 任务管理（v1）

✅ **必做**：
- 实时进度条：显示"当前第 X / 共 Y 段"、"已用时"、"预估剩余时间"
- 流式输出：每完成一段，结果实时追加显示在结果框
- 任务取消：中途可点击"取消"按钮停止
- 失败重试：单段推理失败自动重试 3 次，仍失败则标记为错误段并继续后续段
- 错误段标记：在最终输出中以 `[??? 推理失败 ???]` 占位，不影响整体进度

⏸ **延后到 v2**：
- 批量队列（多文件排队）
- 历史任务记录

**注**：OOM 错误不进入重试，直接报错（见 §4.3.2）。重试机制仅针对其他类型的推理失败。

### 2.9 说话人分离（Phase 2.6 待实施）

**目的**：对多人对话场景标注 `[Speaker 1]` / `[Speaker 2]` 等标签，提升字幕和文稿的可读性。覆盖 PRD §1.3 用户使用场景中的多人场景（B站视频、播客、影视剧、相声小品、电话录音、多人会议等）。

**当前状态**：Phase 2.6 待实施。详细设计（算法选择、speaker tag 格式、集成位置、UI 联动）将在 Phase 2.6 启动时讨论并补充本节。

**技术方向**：Qwen3-ASR 原生不支持说话人分离（已勘探，见 GitHub Issue QwenLM/Qwen3-ASR-Toolkit#13），需引入独立 diarization 模块。主要候选：
- **pyannote.audio + ModelScope 权重源（推荐）**：speaker-diarization-3.1 在 ModelScope 有镜像，绕过 HuggingFace token + 协议门槛
- **阿里 3D-Speaker（备选）**：中文权重训练更针对，但 diarization 文档质量较弱
- 预估开销：模型权重 ~150MB，显存 6-8GB，RTF ≈ 0.025（几乎无延迟开销）

---

## 3. 界面需求（Gradio）

### 3.1 主界面布局

```
┌─────────────────────────────────────────────┐
│  VoxPen  v1.0                          ⚙️🔧 │
├──────────────┬──────────────────────────────┤
│  [文件上传]   │  模型管理                     │
│   拖入或点击  │  状态：● 已加载 (bf16, 1.7B)  │
│              │  [ 加载模型 ]  [ 卸载模型 ]    │
│              │  ☑ 同时加载时间戳对齐器        │
│              │                              │
│              │  基本设置                     │
│              │   ○ 语言：[自动检测 ▼]        │
│              │   ○ 热词/上下文：              │
│              │     ┌─────────────────────┐   │
│              │     │ （输入人名、术语等） │   │
│              │     └─────────────────────┘   │
│              │   ○ 输出格式：                │
│              │     ☑txt  ☑srt  ☑md          │
│              │     (* srt/md 需加载 Aligner) │
│              │                              │
│              │  ▶ 高级设置（折叠）            │
│              │   - VAD 切分参数              │
│              │   - 模型精度（bf16/fp16）      │
│              │   - ForcedAligner（时间戳对齐器）│
│              │   - SRT 字幕参数              │
│              │                              │
│              │  [ 开始转录 ] [ 取消 ]         │
├──────────────┴──────────────────────────────┤
│  进度： [████████░░░░] 65%  (13/20 段)       │
│  已用时 02:15 / 预估剩余 01:10               │
├─────────────────────────────────────────────┤
│  转录结果（实时流式显示）                     │
│  [00:00:01] 各位朋友大家好...                 │
│  [00:00:08] 今天我们来聊聊...                 │
│  ...                                         │
├─────────────────────────────────────────────┤
│  下载：  [📄 txt]  [🎬 srt]  [📝 md]          │
│         [📂 打开输出目录]                     │
└─────────────────────────────────────────────┘
```

### 3.2 首次启动引导

首次启动时，弹出"模型下载设置"对话框（详见 §6）。

### 3.3 启动体验

- 通过 VoxPen Launcher（详见 §5）启动 → 自动激活 venv → 启动 Gradio → 自动打开浏览器
- 启动时自检：ffmpeg、CUDA、模型权重路径有效性（不加载模型本体），缺失项给出可操作的提示

---

## 4. 技术方案

### 4.1 项目结构

```
VoxPen/
├── PRD.md                              # 本文档
├── README.md                           # 项目说明
├── LICENSE
├── requirements.txt
├── .gitignore
├── start.bat                           # Windows 一键启动（源码方式）
├── install.bat                         # 首次安装脚本
│
├── bin/
│   └── ffmpeg.exe                      # 内置 ffmpeg（首次发版用脚本下载）
│
├── config/
│   └── default.yaml                    # 默认配置
│
├── voxpen/                             # 主代码包
│   ├── __init__.py
│   ├── config.py                       # 配置加载/校验（pydantic）
│   ├── pipeline.py                     # 主流水线编排
│   │
│   ├── media/
│   │   ├── extractor.py                # 视频→音频（ffmpeg 子进程）
│   │   └── probe.py                    # 时长、采样率探测
│   │
│   ├── vad/
│   │   └── silero.py                   # Silero VAD 封装
│   │
│   ├── asr/
│   │   ├── transcriber.py              # 模型加载/卸载/推理（load/unload/is_loaded/transcribe）
│   │   └── downloader.py               # 多源模型下载
│   │
│   ├── aligner/
│   │   └── forced_aligner.py           # 可选的精细对齐
│   │
│   ├── diarizer/
│   │   └── speaker_diarizer.py         # Phase 2.6: 说话人分离封装
│   │
│   ├── postproc/
│   │   ├── merger.py                   # 段间拼接、重叠去重
│   │   └── formatter.py                # txt / srt / md 输出
│   │
│   ├── ui/
│   │   ├── gradio_app.py               # 界面主入口
│   │   ├── components.py               # 可复用组件
│   │   └── first_run.py                # 首次启动引导
│   │
│   └── utils/
│       ├── ffmpeg_runner.py
│       ├── gpu.py                      # 显存监测
│       └── logger.py
│
├── launcher/                           # 启动器（PySide6）
│   ├── main.py
│   ├── env_checker.py                  # 环境自检
│   ├── installer.py                    # 依赖安装引导
│   └── ui/
│       └── main_window.py
│
├── scripts/
│   ├── download_ffmpeg.py              # 拉取 ffmpeg.exe
│   ├── check_env.py                    # 环境自检命令行版
│   └── build_launcher.py               # 用 PyInstaller 打包 launcher
│
├── tests/
│   ├── test_vad.py
│   ├── test_formatter.py
│   ├── test_merger.py
│   └── test_pipeline.py
│
└── references/                         # 参考资料（开发用，不分发）
    ├── README.md                       # 说明这些文件用途
    ├── example_qwen3_asr_transformers.py
    ├── working_requirements.txt
    └── qwen_asr_installed/             # 用户已跑通的安装包源码
```

### 4.2 关键依赖

```txt
# 核心
torch>=2.1.0  # +cu121 或 +cu128
transformers>=4.45.0
accelerate>=0.30.0

# 音频处理
silero-vad>=5.1
# Phase 2.6 引入:
# pyannote.audio>=3.1
librosa>=0.10.0
soundfile>=0.12.0

# UI
gradio>=6.14.0

# 工具
pydantic>=2.0
pyyaml
numpy
tqdm

# 模型下载
huggingface_hub>=0.20.0
modelscope>=1.10.0  # 用于魔搭社区下载
```

具体精确版本以 `references/working_requirements.txt` 为准。

### 4.3 关键技术决策

#### 4.3.1 模型加载（用户显式管理）

- 不再采用"应用启动时加载"或"首次推理时延迟加载"——改为用户在 UI 上**显式点击「加载模型」按钮**触发加载
- 「加载模型」按钮旁配 checkbox「同时加载时间戳对齐器（Aligner）」，决定本次加载是否带 Aligner
- 加载完成后，「加载模型」按钮变为「卸载模型」，Aligner checkbox 置灰；卸载后恢复
- 卸载流程：`del model → torch.cuda.empty_cache() → gc.collect()`，确保显存释放
- 配置变更（精度切换、Aligner 增减、模型路径变更）**必须先卸载再重新加载**，不做自动重载
- 模型路径在「设置」中配置，启动时仅校验路径存在性，**不预加载模型**

#### 4.3.2 显存保护

- 每段推理后调用 `torch.cuda.empty_cache()`
- 推理前检测剩余显存，不足时弹窗提示用户降低 `max_segment_length`

**OOM 处理**：
- VAD 已在前置环节将音频切分为 ≤ `max_segment_length`（默认 30s）的段
- bf16 推理实测：30s 音频段（无 Aligner）显存峰值约 4 GB
- 16GB 显存卡上正常使用 OOM 概率极低；若发生 OOM，**直接报错，不做自动降级**，错误信息包含：当前段长、当前显存占用、显存峰值
- 理由：自动减半重试会掩盖真问题（如其他进程占用显存、用户调高了切分参数）

#### 4.3.3 段间重叠去重

- **算法**：后缀-前缀最长公共子串（Longest Common Substring），非 LCS（最长公共子序列）
- **窗口**：在前段末尾 N 字符和后段开头 N 字符内匹配重复，N 默认 10，可配置
- 选择此算法的理由：连续子串匹配只识别真正的边界重叠，避免 LCS 在非重叠区误删偶然重合字符
- 时间戳同步修剪：如果文本中删除了重复字符，对应的 ForcedAlignItem（若有）同步删除，确保字幕生成不出现重复时间区间
- 失败段（`[??? 推理失败 ???]` 占位）不参与去重，前后保留换行/空格视觉分隔

#### 4.3.4 取消机制

- 基于 `threading.Event`
- 流水线各环节定期检查 `event.is_set()`
- 干净退出，释放资源

#### 4.3.5 流水线架构选择

**采用顺序流水线，不采用生产者-消费者并行**。理由见 §2.5。

此决策意味着：
- 不引入 `queue.Queue`、不引入 worker 线程池
- 取消机制只需要 pipeline 主循环检查 `Event.is_set()`，不需要协调多线程退出
- 异常处理路径单一（直接抛到 pipeline 调用方），不需要跨线程异常传播

#### 4.3.6 日志

- 每个任务独立日志文件：`output/<任务>/task.log`
- 包含：参数、各段耗时、错误堆栈、显存峰值
- 方便用户反馈问题时附带

### 4.4 配置文件示例

`config/default.yaml`：

```yaml
model:
  asr_model_id: "Qwen/Qwen3-ASR-1.7B"  # HuggingFace 模型 ID（注意：非 Qwen3-ASR-Flash API 服务名）
  aligner_model_id: "Qwen/Qwen3-ForcedAligner-0.6B"
  asr_local_path: ""                    # 本地路径（优先级高于 model_id）
  aligner_local_path: ""
  dtype: "bf16"                         # bf16 | fp16
  device: "cuda"
  use_forced_aligner: false

vad:
  max_segment_length: 30.0
  min_segment_length: 2.0
  overlap: 0.3
  vad_threshold: 0.5

pipeline:
  retry_times: 3        # 单段推理失败自动重试次数（不含 OOM）

output:
  formats: ["txt", "srt", "md"]
  srt_max_chars_per_line: 20
  srt_max_duration: 5.0
  output_dir: "./output"

download:
  source: "hf-mirror"                   # hf | hf-mirror | modelscope | local | custom
  custom_endpoint: ""
  cache_dir: "./models"
```

### 4.5 参考实现

开发时**必须**参考 `references/qwen_asr_installed/` 目录，特别是：
- `qwen_asr/core/transformers_backend/` —— 这是用户实际跑通的 transformers 后端实现，是 `voxpen/asr/transcriber.py` 的最权威接口参考
- `qwen_asr/cli/` —— CLI 入口，可以看出参数如何传递给推理函数

用户已跑通的启动命令为：
```bash
qwen-asr-demo \
  --asr-checkpoint ./Qwen3-ASR-1.7B \
  --aligner-checkpoint ./Qwen3-ForcedAligner-0.6B \
  --backend transformers \
  --cuda-visible-devices 0
```

`voxpen/asr/transcriber.py` 应该复用 `qwen_asr/core/transformers_backend/` 中的核心推理逻辑，**而不是重新实现**。

---

## 5. VoxPen Launcher（启动管理器）

### 5.1 定位

类似"绘世启动器"的体验：一个独立的 exe，负责环境管理、依赖安装、启动主程序。

### 5.2 技术栈

- Python 3.10 + PySide6
- 用 PyInstaller 打包为单个 exe（约 20~40MB）

> 选 PySide6 而不是 Tauri/Electron 的原因：和主项目同栈，维护成本低；体积可控；Windows 原生体验好。

### 5.3 功能清单

#### 5.3.1 环境自检（每次启动）

检测项：
- Python 3.10 是否存在（`runtime/python/`）
- venv 虚拟环境是否完整（`runtime/venv/`）
- CUDA 驱动是否可用
- ffmpeg.exe 是否存在
- 模型权重路径是否有效（仅校验路径和关键文件存在性，**不加载模型本体**）
- 关键 Python 依赖是否完整

任何一项失败 → 进入"修复向导"。

#### 5.3.2 首次安装向导

引导式流程：

1. **欢迎页**：介绍 VoxPen、磁盘空间需求
2. **下载源选择**：HF 官方 / HF 镜像 / ModelScope / 本地 / 自定义
3. **下载 Python 运行时**：从 python.org 拉取嵌入式 Python 3.10
4. **创建 venv 并装依赖**：基于 `requirements.txt`
5. **下载 ffmpeg.exe**：从 BtbN 的 Windows builds 拉取
6. **下载模型权重**：根据用户选择的源
7. **完成**：跳转到主界面

每一步都有进度条、失败重试、跳过（如果用户已自备）选项。

#### 5.3.3 启动主程序

- 激活 venv
- 启动 Gradio 服务（后台进程）
- 等待端口就绪
- 自动打开浏览器
- Launcher 窗口最小化到系统托盘，可显示运行状态、查看日志、停止服务

#### 5.3.4 设置面板

- 切换模型（ASR / Aligner 路径）
- 切换下载源
- 查看 / 清理日志
- 检查更新（v2 功能）
- "一键修复环境"按钮

### 5.4 目录结构（用户安装后）

```
VoxPen/                        # 用户解压到任意位置（绿色便携）
├── VoxPen.exe                 # Launcher 主程序
├── runtime/
│   ├── python/                # 嵌入式 Python 3.10
│   └── venv/                  # 虚拟环境（首次启动时创建）
├── bin/
│   └── ffmpeg.exe
├── models/
│   ├── Qwen3-ASR-1.7B/
│   └── Qwen3-ForcedAligner-0.6B/
├── voxpen/                    # 主程序源码
├── config/
├── output/                    # 用户产出
└── logs/                      # Launcher + 主程序日志
```

绿色便携：所有运行时和数据都在 VoxPen 目录下，**不污染系统、不写注册表**。

---

## 6. 模型管理与多下载源

### 6.1 支持的下载源

| 源 | 适用场景 | 实现方式 |
|---|---|---|
| **HuggingFace 官方** | 海外用户 / 有代理 | `huggingface_hub.snapshot_download` |
| **HuggingFace 镜像** (hf-mirror.com) | 国内用户（推荐） | 设置 `HF_ENDPOINT=https://hf-mirror.com` |
| **ModelScope 魔搭** | 国内用户备选 | `modelscope.snapshot_download` |
| **本地路径** | 已下载过模型的用户 | 直接指向本地目录 |
| **自定义 endpoint** | 高级用户 | 自定义 HF 兼容 endpoint |

### 6.2 首次下载界面

```
┌─ 首次使用，请设置模型下载源 ─────────────────────┐
│                                                  │
│  下载源：                                         │
│   ○ HuggingFace（国际，需代理）                   │
│   ● HuggingFace 镜像（hf-mirror.com，推荐国内）   │
│   ○ ModelScope 魔搭社区                           │
│   ○ 本地路径：[________________] [浏览]           │
│   ○ 自定义：[__________________]                  │
│                                                  │
│  模型选择：                                       │
│   ☑ Qwen3-ASR-1.7B   (约 3.4GB)                  │
│   ☑ Qwen3-ForcedAligner-0.6B  (约 1.2GB) [可选]  │
│                                                  │
│  [测试连接]  [开始下载]  [跳过（稍后下载）]       │
│                                                  │
└──────────────────────────────────────────────────┘
```

### 6.3 实现要点

- **连通性测试**：下载前 ping 一下源，超时则提示切换
- **断点续传**：HuggingFace Hub 原生支持
- **进度反馈**：下载进度条集成到 UI
- **缓存记录**：已下载的模型记录到 `models/.cache.json`，下次启动跳过
- **校验**：下载完成后校验关键文件（如 `config.json`、`*.safetensors`）

---

## 7. 分发策略

### 7.1 分发包

| 分发包 | 面向用户 | 体积 | 内容 |
|---|---|---|---|
| **源码包** (zip) | 开发者 / 进阶用户 | ~10 MB | 代码 + install.bat + start.bat，依赖和模型在线拉 |
| **整合包** (zip) | 普通用户 / 小白 | ~50 MB | 含 VoxPen Launcher exe + 主程序源码，首次启动引导安装 |

**注意**：不做单文件 onefile exe，因为 PyInstaller 对深度学习项目打包后体积巨大（5GB+）且启动慢。Launcher 思路完美避开这个问题。

### 7.2 安装流程对比

**源码包**：
1. 下载 zip 解压
2. 双击 `install.bat`：自动建 venv + 装依赖 + 下载 ffmpeg
3. 双击 `start.bat`：启动 Gradio
4. 浏览器打开 → 使用

**整合包**：
1. 下载 zip 解压
2. 双击 `VoxPen.exe`：进入 Launcher
3. 首次启动向导：下载 Python 运行时 + 装依赖 + 下载 ffmpeg + 下载模型
4. 完成 → 自动启动主程序 → 浏览器打开 → 使用

### 7.3 v2 展望

- 数字签名（避免 Windows SmartScreen 警告）
- Inno Setup 安装向导（提供"开始菜单快捷方式"等系统集成）
- 自动更新
- **AI 文稿润色（可选功能）**：基于本地小 LLM（如 Qwen3.5-A3B、Gemma4-A4B 等）对 ASR 原始输出做后处理润色：清理口语化冗余（"呃...那个..."）、修正标点、整理段落。设计原则：作为**可选的额外输出**，原始转录（保留所有语气词，符合 §1 核心原则）始终保留。用户在 UI 上 toggle 是否启用润色，输出双份文件（`*.raw.txt` + `*.polished.txt`）
- **AI 字幕分行(可选功能)**:与"AI 文稿润色"配套,基于本地小 LLM(1.7B-4B 量级,如 Qwen3-1.7B/4B、Phi-4-mini、MiniCPM-4-2B 等)对带时间戳的字幕做语义化分行。默认走 v1 的强标点 + 软兜底规则(§4.3.3-ish 待补),仅在规则失效或用户启用"语义优化"时调用 LLM。设计原则:小模型够用,避免引入重型推理依赖

特别说明：v1 实测发现，Qwen3-ASR 在英文口语转录场景下标点输出稀疏，导致 srt 字幕几乎全部走"硬截断"分行规则，字幕短句割裂（2-3 词一行）。v1 中文场景标点输出正常，字幕分行体验合格；但英文场景下用户体验为 acceptable but suboptimal。v2 的 AI 字幕分行是该问题的根本解法——通过本地小 LLM（1.7B-4B 量级）结合纯文本和精细时间戳做语义化分行，不依赖原始 ASR 输出的标点完整性。

---

## 8. 验收标准

### 8.1 功能验收

- [x] 能成功处理 60 分钟以上的视频文件（mp4）而不 OOM
- [x] 热词传入后，专业术语识别明显改善
- [x] 转录过程中 UI 不卡死，进度实时更新（每段完成 < 0.5s 内界面响应）
- [x] 取消按钮能在 3 秒内停止任务
- [x] 生成的 .srt 文件能被 VLC、PotPlayer、Premiere 正确加载
- [x] 生成的 .md 文件在 Typora、VSCode 中显示正常
- [x] Launcher 能在干净的 Windows 系统上完成全套安装

### 8.2 性能基准（参考机：RTX 5060 Ti 16GB）

- 模型加载（bf16，2-shard checkpoint）：**实测 3.2 秒**
- 30s 音频段推理（bf16，无 Aligner）：**实测 3.4 秒**，RTF ≈ 0.11
- 30s 音频段推理显存峰值（bf16，无 Aligner）：**实测 4.03 GB**
- 30 分钟音频 → 转录耗时 < 8 分钟（基于 RTF 推算）

### 8.3 兼容性验收

- Windows 10 21H2 及以上 ✓
- Windows 11 全版本 ✓
- 至少 8GB 显存的 NVIDIA 显卡 ✓

---

## 9. 已知风险与备注

### 9.1 技术风险

- **Qwen3-ASR 接口稳定性**：模型更新可能导致接口变化。开发时**严格基于 `references/qwen_asr_installed/`** 已跑通版本，不擅自升级
- **Silero VAD 对音乐/强噪音不稳**：v2 可加 WebRTC VAD 备选
- **bf16 兼容性**：Pascal/老 Turing 显卡不支持 bf16，已提供 fp16 兜底
- **PySide6 打包后体积**：约 30~50MB，可接受

### 9.2 法律与合规

- 模型权重遵循 Qwen 官方许可（Apache 2.0 或 Tongyi Qianwen License，以官方为准）
- ffmpeg 使用 LGPL 版本，避免静态链接的 GPL 风险
- VoxPen 本身建议采用 MIT 许可
- 在 README 中明确标注："本项目基于 Qwen3-ASR 模型，模型版权归阿里云所有"

### 9.3 开发顺序建议

建议 Claude Code 按以下顺序实现：

1. **基础设施层**（1~2 天）
   - 配置加载（`config.py`）
   - 日志系统（`utils/logger.py`）
   - ffmpeg 调用封装（`utils/ffmpeg_runner.py`）

2. **核心功能层**（3~4 天）
   - 音频提取（`media/extractor.py`）
   - VAD 切分（`vad/silero.py`）
   - ASR 推理（`asr/transcriber.py`）—— **重点参考 `references/`**
   - 后处理（`postproc/merger.py`、`formatter.py`）

3. **流水线层**（2 天）
   - 顺序流水线（`pipeline.py`）：VAD 段切片 + Transcriber 调用 + 重试 + 错误段标记 + 进度回调 + 取消信号

4. **UI 层**（2~3 天）
   - Gradio 界面（`ui/gradio_app.py`）
   - 流式输出
   - 取消/进度

5. **Launcher**（3~4 天，可与 UI 层并行）
   - 环境自检
   - 安装向导
   - 多下载源支持

6. **测试与打包**（2 天）
   - 单元测试
   - PyInstaller 打包 Launcher
   - 整合包构建

**总计预估：13~17 个工作日**

---

## 10. 附：开发参考资料清单

`references/` 目录包含以下资料，开发时**必读**：

1. `example_qwen3_asr_transformers.py` —— 官方 transformers 后端示例
2. `working_requirements.txt` —— 用户实际跑通的依赖版本
3. `qwen_asr_installed/` —— 用户实际安装的 qwen_asr 包源码
   - 重点关注 `core/transformers_backend/`
   - 重点关注 `cli/`（理解参数传递）

详见 `references/README.md`。

---

## Changelog

### v1.5 (2026-05-14)
- 调整：说话人分离从 v2 范围提升至 v1 范围（Phase 2.6 实施），§1.5 Non-Goals 删除对应条目，新增 §2.9 占位段说明技术方向
- 补充：§4.1 项目结构增加 `voxpen/diarizer/` 子目录，§4.2 依赖列表增加 pyannote.audio（注释形式，Phase 2.6 引入）
- 补充：§7.3 v2 展望中"AI 字幕分行"强化说明——该功能是英文口语场景字幕短句割裂问题的根本解法（已在 v1 验证中确认该问题存在）
- v1.5 范围标记 v2 Non-Goals：说话人 enrollment / voiceprint

### v1.4 (2026-05-14)
- 调整：§2.6 / §4.3.3 重叠去重算法从 LCS（最长公共子序列）改为后缀-前缀最长公共子串（Longest Common Substring），理由：连续子串匹配只识别真正的边界重叠，避免 LCS 在非重叠区误删偶然重合字符。匹配窗口默认 N=10，可配置
- 调整：§4.3.3 明确"时间戳同步修剪"规则（删除字符时同步删除对应 ForcedAlignItem）
- 调整：§4.3.3 明确"失败段不参与去重"规则
- 补充：§7.3 v2 展望追加"AI 文稿润色"作为可选功能（基于本地小 LLM，与原始转录并存输出）
- 修补：§2.3 中 overlap 参数描述里的"LCS 去重"同步改为"后缀-前缀子串去重"（与本次算法调整对齐）
- 补充:§7.3 v2 展望追加"AI 字幕分行"作为可选功能,与文稿润色配套,目标小模型(1.7B-4B 量级)

### v1.3 (2026-05-13)
- 调整：§2.5 流水线架构从「生产者-消费者并行」改为「顺序流水线」。原架构在 VAD 一次性预处理 + 切片成本接近零的场景下属于过度设计，并行收益不足 1% 且增加并发 bug 风险
- 调整：§4.1 项目结构删除 `voxpen/worker/` 子目录（含 `queue.py`、`task.py`），任务状态由 pipeline 内部 `threading.Event` 直接管理
- 补充：§4.3 新增小节（§4.3.5）说明流水线架构选择的决策依据，原 §4.3.5（日志）顺延为 §4.3.6
- 调整：§9.3 开发顺序建议同步更新（删除"生产者-消费者队列"任务）
- 清理：§4.4 配置示例和 config/default.yaml 删除 `pipeline.queue_size` 字段（顺序流水线不需要队列）；voxpen/config.py 的 PipelineConfig 同步删除

### v1.2 (2026-05-13)
- 纠正：Qwen3-ASR 本身不返回时间戳，时间戳功能完全依赖 ForcedAligner（v1.1 中 §2.4 描述错误，基于 Phase 2.1 实测纠正）
- 调整：ForcedAligner 从"可选高级功能"调整为"时间戳/字幕功能的必要组件"，UI 上与 srt/md 输出联动
- 调整：OOM 不再自动降级重试，直接报错附诊断信息（§4.3.2）
- 调整：模型加载从「单例延迟加载 + 配置变更自动重载」改为「用户手动 load/unload」（§4.3.1）
- 补充：§8.2 性能基准更新为 Phase 2.1 实测数据
- 调整：删除独立的 `asr/model_loader.py`，load/unload 职责合并到 `asr/transcriber.py`（与 §4.3.1 模型加载策略调整配套）
- 调整：§2.3 VAD 切分策略重写，反映 Phase 2.0 实现的"不丢任何 VAD 段"策略（旧策略「<2s 段合并/丢弃」违反 §1 核心原则「不丢语气词」，已在 Phase 2.0 实测纠正）

### v1.1
- 初版（开发起点）

---

**文档结束**
