# VoxPen v1 — 开发计划

> 基于 PRD v1.1 和 references/ 参考实现。
> 每 Phase 完成后暂停，等待人工确认再进入下一 Phase。

---

## Phase 1：项目骨架与基础设施 ✅ 当前

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
│   │   ├── model_loader.py       # 单例模型加载
│   │   ├── transcriber.py        # Qwen3-ASR 推理适配器
│   │   └── downloader.py         # 多源模型下载
│   ├── aligner/
│   │   ├── __init__.py
│   │   └── forced_aligner.py     # 可选精细对齐
│   ├── postproc/
│   │   ├── __init__.py
│   │   ├── merger.py             # 段间拼接、重叠去重
│   │   └── formatter.py          # txt/srt/md 输出
│   ├── worker/
│   │   ├── __init__.py
│   │   ├── queue.py              # 生产者-消费者
│   │   └── task.py               # 任务状态机
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

## Phase 2.0：接口契约验证（前置必做）
- 编写 `scripts/verify_qwen_asr_contract.py`
- 实际调用 Qwen3ASRModel，打印参数和返回值结构
- 输出保存为 `references/qwen_asr_actual_api.txt`

---

## Phase 2：核心处理模块
- 2.1 音频提取 → 16kHz 单声道 WAV
- 2.2 媒体探测
- 2.3 VAD 封装（两套时间戳：vad 原始 + feed 扩展）
- 2.4 ASR 推理适配器（context/language 参数以 Phase 2.0 为准）
- 2.5 模型下载（多源）
- 2.6 段合并（LCS 去重）
- 2.7 格式化输出（txt/srt/md）

---

## Phase 3：流水线编排
- 3.1 任务状态机（threading.Event 取消）
- 3.2 生产者-消费者队列（CPU VAD / GPU 转录）
- 3.3 主流水线编排 + 回调钩子

---

## Phase 4：Gradio 界面
- 4.1 主界面（文件上传、设置、进度、流式结果、下载）
- 4.2 可复用组件
- 4.3 首次启动引导
- 4.4 流式回调集成

---

## Phase 5：Launcher 启动器
- 5.1 环境自检（每项支持"检测到已安装→跳过"）
- 5.2 安装向导（每步复用已有；首步可指定本地模型路径）
- 5.3 主窗口（系统托盘、启停、日志、设置）
- 5.4 辅助脚本（download_ffmpeg / start.bat / install.bat / build_launcher）

---

## Phase 6：测试与收尾
- 6.1 单元测试（test_config / test_extractor / test_downloader / test_vad / test_formatter / test_merger / test_pipeline）
- 6.2 端到端手动场景（5 个）：
  1. 60 分钟视频不 OOM
  2. 低显存降级
  3. 取消任务
  4. 模型缺失提示
  5. ffmpeg 缺失提示
- 6.3 README 完善

---

## 关键设计决策

| 决策 | 说明 |
|------|------|
| 不重写推理 | `transcriber.py` 适配 `qwen_asr.Qwen3ASRModel`，非替代 |
| VAD 优先级 | Silero VAD 在语义边界切分，优于库内固定时长切分 |
| 两套时间戳 | VAD 原始 + feed 扩展，LCS 去重针对人为重叠区 |
| 单例模型 | 延迟加载，首次推理时才初始化，常驻显存 |
| 取消机制 | `threading.Event`，3 秒内响应 |
| 复用已有环境 | Launcher 每步检测已装组件，支持指定已有模型路径 |

---

*最后更新: 2026-05-13*
