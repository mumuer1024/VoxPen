# Qwen3-ASR 接口契约报告

> **生成方式**：阅读以下源码后逐字段核对，非凭记忆编写。
> **生成日期**：2026-05-13
> **模型**：Qwen3-ASR-1.7B（transformers 后端）
> **依赖版本**：qwen-asr==0.0.6, transformers==4.57.6

---

## 1. 阅读过的关键文件

| 文件 | 作用 |
|------|------|
| `references/qwen_asr_installed/inference/qwen3_asr.py` | ★ 主入口类 `Qwen3ASRModel`：`from_pretrained()` / `transcribe()` |
| `references/qwen_asr_installed/inference/utils.py` | 音频标准化、切分、语言列表、`parse_asr_output`、`ASRTranscription` |
| `references/qwen_asr_installed/inference/qwen3_forced_aligner.py` | `Qwen3ForcedAligner`：`from_pretrained()` / `align()` / `ForcedAlignResult` / `ForcedAlignItem` |
| `references/qwen_asr_installed/core/transformers_backend/__init__.py` | 导出：`Qwen3ASRConfig` / `Qwen3ASRForConditionalGeneration` / `Qwen3ASRProcessor` |
| `references/qwen_asr_installed/core/transformers_backend/processing_qwen3_asr.py` | `Qwen3ASRProcessor.__call__(text, audio)` → `BatchFeature` |
| `references/qwen_asr_installed/core/transformers_backend/configuration_qwen3_asr.py` | `Qwen3ASRConfig` 结构定义 |
| `references/qwen_asr_installed/cli/demo.py` | CLI 入口，展示参数如何从 argparse 映射到 `from_pretrained()` |
| `references/example_qwen3_asr_transformers.py` | 官方示例：单段/批量/时间戳/语言强制 |
| `references/working_requirements.txt` | 跑通环境的依赖版本 |

---

## 2. 模型加载入口

### 类名
```python
from qwen_asr import Qwen3ASRModel
```

### 签名
```python
@classmethod
def from_pretrained(
    cls,
    pretrained_model_name_or_path: str,         # HF repo ID 或本地目录路径
    forced_aligner: Optional[str] = None,        # ForcedAligner 的路径/repo_id（可选）
    forced_aligner_kwargs: Optional[Dict[str, Any]] = None,
    max_inference_batch_size: int = 32,           # 批量推理最大 batch size（-1=不切分）
    max_new_tokens: Optional[int] = 512,          # 每段最大生成 token 数
    **kwargs,                                      # → 转发给 AutoModel.from_pretrained()
) -> "Qwen3ASRModel":
```

### `**kwargs` 说明
`**kwargs` 直接透传给 `AutoModel.from_pretrained()`。常用参数：

| 参数名 | 类型 | 说明 | 示例 |
|--------|------|------|------|
| `dtype` | `torch.dtype` | 模型精度 | `torch.bfloat16` / `torch.float16` |
| `device_map` | `str` | 设备映射 | `"cuda:0"` / `"auto"` |
| `attn_implementation` | `str` | 注意力实现 | `"flash_attention_2"` (可选) |

**注意**：`dtype` 是 torch.dtype 对象，不是字符串。YAML 配置中的 `"bf16"` 需要在代码中转换为 `torch.bfloat16`。

### 加载示例（等价于用户跑通命令）
```python
import torch
from qwen_asr import Qwen3ASRModel

asr = Qwen3ASRModel.from_pretrained(
    "C:/Users/mumuer/qwen-asr/Qwen3-ASR-1.7B",
    forced_aligner="C:/Users/mumuer/qwen-asr/Qwen3-ForcedAligner-0.6B",
    forced_aligner_kwargs=dict(
        dtype=torch.bfloat16,
        device_map="cuda:0",
    ),
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_inference_batch_size=32,
    max_new_tokens=512,
)
```

### 不带 ForcedAligner 的最简加载
```python
asr = Qwen3ASRModel.from_pretrained(
    "C:/Users/mumuer/qwen-asr/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)
```

---

## 3. 推理调用入口

### 方法名
```python
def transcribe(
    self,
    audio: Union[AudioLike, List[AudioLike]],
    context: Union[str, List[str]] = "",
    language: Optional[Union[str, List[Optional[str]]]] = None,
    return_time_stamps: bool = False,
) -> List[ASRTranscription]:
```

### 参数详解

#### `audio` — 音频输入
类型：`AudioLike = Union[str, Tuple[np.ndarray, int]]`

| 格式 | 说明 |
|------|------|
| `str` | 本地文件路径、HTTPS URL、或 base64 data URL |
| `(np.ndarray, int)` | (waveform_float32_1d_mono, sample_rate) |
| `List[...]` | 上述任一格式的列表，支持批量 |

内部自动标准化为：
- **采样率**：16000 Hz（自动重采样）
- **声道**：单声道（多声道取均值）
- **dtype**：float32
- **范围**：[-1, 1]（自动归一化）

#### `context` — 热词/上下文 ★
类型：`Union[str, List[str]]`，默认 `""`

- 作为 chat template 中 system message 的 `content` 传入
- 单字符串会对所有样本广播；列表长度需与音频数一致
- **示例**：`"这是一段关于深度学习的讲座。专业术语：transformer, attention, backpropagation。人名：张三, 李四。"`

#### `language` — 强制语言
类型：`Optional[Union[str, List[Optional[str]]]]`，默认 `None`

- `None`：自动检测语言
- 指定时强制输出纯文本（会在 prompt 末尾追加 `"language Chinese<asr_text>"`）
- 支持的语种（完整列表见 `SUPPORTED_LANGUAGES`）：
  Chinese, English, Cantonese, Japanese, Korean, Arabic, German, French, Spanish, Portuguese, Indonesian, Italian, Russian, Thai, Vietnamese, Turkish, Hindi, Malay, Dutch, Swedish, Danish, Finnish, Polish, Czech, Filipino, Persian, Greek, Romanian, Hungarian, Macedonian
- 格式：首字母大写（如 `"Chinese"`, `"English"`）

#### `return_time_stamps` — 时间戳
类型：`bool`，默认 `False`

- 需要 `forced_aligner` 已加载，否则抛 `ValueError`
- 开启后每段会走 ForcedAligner 做字级对齐（多占 ~1.5GB 显存）

### 返回值

```python
List[ASRTranscription]  # 每个输入音频对应一个结果

@dataclass
class ASRTranscription:
    language: str              # 检测到的语言，如 "Chinese"；空字符串表示未知/静音
    text: str                  # 转录文本（所有 chunk 已合并）
    time_stamps: Optional[Any] # ForcedAlignResult 或 None
```

### 时间戳结构（ForcedAlignResult）

```python
@dataclass(frozen=True)
class ForcedAlignResult:
    items: List[ForcedAlignItem]

@dataclass(frozen=True)
class ForcedAlignItem:
    text: str         # 对齐单元（汉字逐字/英文逐词）
    start_time: float # 起始时间（秒）
    end_time: float   # 结束时间（秒）
```

- **粒度**：中文逐字，英文/空格分隔语言逐词
- **精度**：毫秒级（`round(x/1000, 3)` 秒）
- **无置信度字段**

### 内部机制
`transcribe()` 内部自动完成：
1. 音频标准化（归一化为 16kHz mono float32）
2. 长音频切分（`split_audio_into_chunks`：ASR 上限 1200s/段，ForcedAligner 上限 180s/段，基于能量边界）
3. 逐段批量推理（batch_size 由 `max_inference_batch_size` 控制）
4. 段间文本拼接 + 语言合并
5. 时间戳偏移修正 + 合并（如启用）
6. 重复检测修复（`detect_and_fix_repetitions`）

---

## 4. 关键常量和支持列表

```python
SAMPLE_RATE = 16000
MAX_ASR_INPUT_SECONDS = 1200      # ASR 单段最大秒数
MAX_FORCE_ALIGN_INPUT_SECONDS = 180  # Aligner 单段最大秒数
MIN_ASR_INPUT_SECONDS = 0.5
```

`SUPPORTED_LANGUAGES`: 30 种语言（名称为首字母大写格式）

---

## 5. 最小调用示例

```python
import torch
from qwen_asr import Qwen3ASRModel

# 加载
asr = Qwen3ASRModel.from_pretrained(
    "C:/Users/mumuer/qwen-asr/Qwen3-ASR-1.7B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
)

# 推理（单段 WAV 文件）
results = asr.transcribe(
    audio="path/to/audio.wav",
    context="专业术语：transformer, attention",
    language="Chinese",
    return_time_stamps=False,
)

print(results[0].text)       # 转录文本
print(results[0].language)   # 检测到的语言
```

---

## 6. 待确认项

- [ ] `max_inference_batch_size=-1` 时是否真的不切分，还是行为不同？源码中 `chunk_list` 在 `<=0` 时 `yield xs`（不分批），但 `split_audio_into_chunks` 仍然会按 1200s 切分长音频
- [ ] 时间戳 `start_time` / `end_time` 的单位确认：`parse_timestamp` 中除以 1000，说明原始值是毫秒，返回秒。实际验证后确认

---

*报告结束。请审阅后进入任务 2。*
