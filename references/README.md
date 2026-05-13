# References — 开发参考资料

本目录包含 VoxPen 项目的开发参考资料。**这些文件不参与最终产品分发，仅供 Claude Code（或其他 AI 编程助手）和人类开发者在开发过程中参考。**

---

## 目录内容

```
references/
├── README.md                          # 本文件
├── example_qwen3_asr_transformers.py  # Qwen3-ASR 官方 transformers 后端示例
├── working_requirements.txt           # 用户实际跑通的依赖版本清单
└── qwen_asr_installed/                # 用户实际跑通的 qwen_asr 包源码
    ├── cli/
    ├── core/
    │   ├── transformers_backend/      # ★ transformers 后端实现（最重要）
    │   └── vllm_backend/              # vLLM 后端（VoxPen 不使用，仅参考）
    └── inference/
```

---

## 各文件说明与开发指引

### 1. `example_qwen3_asr_transformers.py`

来源：[QwenLM/Qwen3-ASR 官方 GitHub 仓库](https://github.com/QwenLM/Qwen3-ASR)

**用途**：
- 了解 Qwen3-ASR 模型的基本调用方式
- 查阅模型加载、推理、参数传递的官方推荐写法

**注意**：这只是个简单 demo，**不能作为生产代码的唯一依据**。生产代码请优先参考 `qwen_asr_installed/` 目录下的真实实现。

---

### 2. `working_requirements.txt`

来源：用户在已跑通环境下执行 `pip freeze` 的输出。

**生成命令**：
```bash
C:\Users\Mumuer\Qwen-ASR\env310\Scripts\activate.bat
pip freeze > working_requirements.txt
```

**用途**：
- 提供精确的依赖版本号（torch、transformers、accelerate 等）
- 避免因版本不匹配导致的接口对不上、模型加载失败等问题

**给 Claude Code 的指引**：
- 编写 `requirements.txt` 时，关键依赖（torch、transformers、accelerate、modelscope、huggingface_hub）的版本应**与本文件保持一致或不低于此版本**
- 如果某个依赖版本过新导致问题，回退到本文件的版本

---

### 3. `qwen_asr_installed/`

**来源**：用户已跑通环境的 `site-packages/qwen_asr/` 目录完整复制。

**用途**：这是 VoxPen 开发时**最权威的参考实现**。

#### 3.1 用户已跑通的启动命令

用户在原环境中通过以下命令成功跑起来过：

```bash
qwen-asr-demo \
  --asr-checkpoint ./Qwen3-ASR-1.7B \
  --aligner-checkpoint ./Qwen3-ForcedAligner-0.6B \
  --backend transformers \
  --cuda-visible-devices 0
```

这意味着：
- **transformers 后端是已验证可用的**
- **模型权重是本地路径方式加载的**（不是 HuggingFace 仓库 ID）
- **ASR 模型和 ForcedAligner 模型可以同时加载**

#### 3.2 重点目录

##### `qwen_asr_installed/core/transformers_backend/`

⭐⭐⭐ **这是最重要的参考目录**。

VoxPen 的 `voxpen/asr/transcriber.py` 应该**复用这里的核心推理逻辑**，而不是重新实现。具体做法：

- 阅读这个目录下的所有 `.py` 文件
- 理解模型加载流程、推理调用方式、返回值结构
- 在 `voxpen/asr/transcriber.py` 中以**适配器模式**封装这些逻辑
- 在外层加上 VoxPen 特有的能力：分段推理、流式输出、热词上下文、错误重试

##### `qwen_asr_installed/cli/`

包含 `qwen-asr-demo` 命令的入口实现。

阅读这里可以理解：
- 命令行参数如何映射到推理参数（如 `--backend transformers` 走哪条代码路径）
- ASR checkpoint 和 Aligner checkpoint 是如何被分别加载的
- `--cuda-visible-devices` 是如何生效的

##### `qwen_asr_installed/core/vllm_backend/`

⚠️ **VoxPen 不使用 vLLM 后端**，但保留这个目录的原因：
- 对比两个后端的接口设计，理解 transformers 后端的特性
- 万一 transformers 后端有 bug，可以参考 vLLM 后端的实现做交叉验证

**Claude Code 不要在 VoxPen 中使用任何 vLLM 相关代码。**

#### 3.3 给 Claude Code 的开发原则

1. **优先复用，而非重写**：能用 `qwen_asr_installed/` 现成函数就直接调用，避免重复造轮子
2. **遇到接口不确定时**：以 `qwen_asr_installed/` 的实际代码为准，而不是 `example_qwen3_asr_transformers.py`
3. **不要修改 `qwen_asr_installed/`**：这是参考资料，不是源码。VoxPen 的代码写在 `voxpen/` 下
4. **依赖安装**：VoxPen 的 `requirements.txt` 中可以直接添加 `qwen-asr` 包（如果 PyPI 有），或者通过 git URL 安装

---

## 模型权重位置

用户已下载的模型权重位于：

```
C:\Users\Mumuer\Qwen-ASR\Qwen3-ASR-1.7B\
C:\Users\Mumuer\Qwen-ASR\Qwen3-ForcedAligner-0.6B\
```

**注意**：
- 这些权重**没有复制到 references/**（体积过大，几个 GB）
- VoxPen 的 Launcher 应支持"指定本地已有模型路径"的选项，让用户复用已下载的权重，避免重复下载
- 默认下载位置应为 `<VoxPen 安装目录>/models/`

---

## 与 PRD 的关系

本目录的所有内容都是为了支撑 PRD（`../PRD.md`）中描述的功能实现。开发遇到问题时的优先级：

1. **PRD.md** —— 描述"做什么"
2. **references/qwen_asr_installed/** —— 描述"怎么做"（针对 ASR 推理部分）
3. **references/working_requirements.txt** —— 描述"用什么版本"
4. **references/example_qwen3_asr_transformers.py** —— 辅助理解模型基本用法

如果 PRD 和 references 之间有冲突，**以 references 为准**（因为 references 是已验证可跑通的真实代码）。

---

## 文件来源溯源

| 文件 | 来源 | 修改情况 |
|---|---|---|
| `example_qwen3_asr_transformers.py` | https://github.com/QwenLM/Qwen3-ASR | 未修改 |
| `working_requirements.txt` | 用户本地环境 `pip freeze` 输出 | 未修改 |
| `qwen_asr_installed/` | 用户本地 `site-packages/qwen_asr/` 完整复制 | 未修改，但已删除 `__pycache__` 等缓存 |

---

**本目录由人类开发者维护，AI 编程助手仅作为参考使用，不应修改本目录任何文件。**