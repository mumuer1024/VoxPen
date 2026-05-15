# VoxPen v1 — 开发计划

> 基于 PRD v1.5 和 references/ 参考实现。
> 每 Phase 完成后暂停，等待人工确认再进入下一 Phase。

---

## Phase 1：项目骨架与基础设施 ✅ 完成

### 1.1 目录结构
```
VoxPen/
├── bin/                          # 内置二进制（ffmpeg.exe）
├── config/
│   └── default.yaml              # 默认配置
├── voxpen/                       # 主代码包
│   ├── __init__.py
│   ├── config.py                 # Pydantic 配置加载/校验
│   ├── pipeline.py               # 主流水线编排（Phase 3）
│   ├── media/
│   │   ├── __init__.py
│   │   ├── extractor.py          # 视频→音频
│   │   └── probe.py              # 媒体探测
│   ├── vad/
│   │   ├── __init__.py
│   │   └── silero.py             # Silero VAD 封装
│   ├── asr/
│   │   ├── __init__.py
│   │   ├── types.py              # NotLoadedError / TranscriberOOMError
│   │   ├── transcriber.py        # 模型加载/卸载/推理（load/unload/is_loaded/has_aligner/transcribe）
│   │   └── downloader.py         # 多源模型下载
│   ├── aligner/
│   │   ├── __init__.py
│   │   └── forced_aligner.py     # 可选精细对齐
│   ├── postproc/
│   │   ├── __init__.py
│   │   ├── merger.py             # 段间拼接、重叠去重
│   │   └── formatter.py          # txt/srt/md 输出
│   ├── ui/
│   │   ├── __init__.py
│   │   ├── gradio_app.py         # 界面主入口
│   │   ├── components.py         # 可复用组件
│   │   └── first_run.py          # 首次启动引导
│   └── utils/
│       ├── __init__.py
│       ├── ffmpeg_runner.py      # ffmpeg 子进程封装
│       ├── gpu.py                # 显存监测
│       └── logger.py             # 日志系统
├── launcher/                     # PySide6 启动器
│   ├── __init__.py
│   ├── main.py
│   ├── env_checker.py
│   ├── installer.py
│   └── ui/
│       ├── __init__.py
│       └── main_window.py
├── scripts/
│   ├── verify_qwen_asr_contract.py  # Phase 2.0：接口契约验证
│   ├── download_ffmpeg.py
│   ├── check_env.py
│   └── build_launcher.py
└── tests/
    ├── test_config.py
    ├── test_extractor.py
    ├── test_downloader.py
    ├── test_vad.py
    ├── test_formatter.py
    ├── test_merger.py
    └── test_pipeline.py
```

### 1.3 关键配置
- asr_model_id: `Qwen/Qwen3-ASR-1.7B`（非 Qwen3-ASR-Flash）
- aligner_model_id: `Qwen/Qwen3-ForcedAligner-0.6B`
- 默认后端: transformers（不支持 vLLM）
- 默认精度: bf16

---

## Phase 2：核心处理模块

### Phase 2.0：基础组件 ✅ 完成
- `media/extractor.py`：视频/音频 → 16kHz 单声道 WAV
- `vad/silero_vad.py`：Silero VAD 封装（pip 包，非 torch.hub）
- `vad/segmenter.py`：段后处理「不丢任何 VAD 段」策略（合并→救援→切分→兜底过滤）

### Phase 2.1：ASR 接口契约勘探与最小推理验证 ✅ 完成
- 阅读 `references/qwen_asr_installed/` 真实源码，输出 `references/qwen3_asr_actual_api.md`
- `scripts/test_qwen3_asr_minimal.py`：30s wav 单段推理验证
- 实测：bf16 加载 3.2s，推理 3.4s/30s 段（RTF 0.11），显存峰值 4.03 GB

### Phase 2.2：Transcriber 模块封装 ✅ 完成
- `voxpen/asr/transcriber.py`：load / unload / is_loaded / has_aligner / transcribe 完整生命周期
- `voxpen/asr/types.py`：NotLoadedError / TranscriberOOMError 异常体系
- 用户显式管理生命周期（不做单例/自动重载）；OOM 直接报错附诊断（不自动降级）
- `tests/test_transcriber.py`：待用户在 venv 跑 pytest

### Phase 2.3：顺序流水线 ✅ 完成
- `voxpen/pipeline.py`：VAD 段切片 + Transcriber 调用 + 重试 + 错误段标记 + 进度回调 + 取消信号
- 输入：16kHz wav ndarray + 配置；输出：`List[TranscribedSegment]`
- 不做生产者-消费者并行（PRD v1.5 §2.5）
- `tests/test_pipeline.py`：13/13 通过

### Phase 2.4.1：Merger 模块 ✅ 完成
- `voxpen/postproc/merger.py`：段合并 + 后缀-前缀子串去重 + 时间戳全局对齐 + 同步修剪
- `tests/test_merger.py`：25/25 通过

### Phase 2.4.2：Formatter 模块 ✅ 完成
- `voxpen/postproc/formatter.py`：txt / srt / md 三种格式输出
- 字幕行聚合规则：强标点优先 → 软兜底（弱标点）→ 硬截断
- srt 无 Aligner 时禁用（抛 ValueError）
- `tests/test_formatter.py`：19/19 通过

### Phase 2.5：模型下载 ✅ 完成
- `voxpen/asr/downloader.py`：5 个下载源（hf / hf-mirror / modelscope / local / custom）
- 缓存检测 + 轻量校验 + 连通性测试（check_connectivity）
- `tests/test_downloader.py`：23/23 通过

**Phase 2 测试覆盖**：87 个单元测试全部通过（Transcriber 7 + Pipeline 13 + Merger 25 + Formatter 19 + Downloader 23）

### Phase 2.6：说话人分离 ⏳ 待开始（明天实施）
- `voxpen/diarizer/speaker_diarizer.py`：基于 pyannote.audio（候选）做说话人分离
- 集成位置：VAD 之后、Transcriber 调用之前，给每段打上 speaker tag
- 模型权重来源：ModelScope 镜像（绕过 HuggingFace token 门槛）
- 详细设计（speaker tag 格式、UI 联动、加载/卸载策略）在 Phase 2.6 启动时讨论
- 详见 PRD v1.5 §2.9

---

## Phase 3：Gradio UI ⏳ 待开始
- 3.1 主界面（文件上传、设置、进度、流式结果、下载）
- 3.2 可复用组件（模型加载/卸载按钮、Aligner / Diarizer toggle）
- 3.3 首次启动引导
- 3.4 流式回调集成

---

## Phase 4：Launcher 启动器
- 4.1 环境自检（每项支持"检测到已安装→跳过"）
- 4.2 安装向导（每步复用已有；首步可指定本地模型路径）
- 4.3 主窗口（系统托盘、启停、日志、设置）
- 4.4 辅助脚本（download_ffmpeg / start.bat / install.bat / build_launcher）

---

## Phase 5：测试与收尾
- 5.1 单元测试（test_config / test_extractor / test_downloader / test_vad / test_formatter / test_merger / test_pipeline）
- 5.2 端到端手动场景（5 个）：
  1. 60 分钟视频不 OOM
  2. 低显存降级
  3. 取消任务
  4. 模型缺失提示
  5. ffmpeg 缺失提示
- 5.3 README 完善

---

## 关键设计决策

| 决策 | 说明 |
|------|------|
| 不重写推理 | `transcriber.py` 适配 `qwen_asr.Qwen3ASRModel`，非替代 |
| VAD 优先级 | Silero VAD 在语义边界切分，优于库内固定时长切分 |
| 两套时间戳 | VAD 原始 + feed 扩展，LCS 去重针对人为重叠区 |
| 用户显式 load/unload | 不做单例/自动重载，模型加载由用户在 UI 上点击触发（PRD v1.3 §4.3.1） |
| 取消机制 | `threading.Event`，3 秒内响应 |
| 复用已有环境 | Launcher 每步检测已装组件，支持指定已有模型路径 |

---

*最后更新: 2026-05-14*
