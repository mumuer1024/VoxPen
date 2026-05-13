"""
VoxPen 配置系统

基于 Pydantic v2 的类型安全配置加载与校验。
支持从 YAML 文件加载，并提供默认值。
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator


# ============================================================
# 子配置模型
# ============================================================

class ModelConfig(BaseModel):
    """模型加载相关配置"""

    asr_model_id: str = Field(
        default="Qwen/Qwen3-ASR-1.7B",
        description="HuggingFace ASR 模型 ID（注意：非 Qwen3-ASR-Flash）",
    )
    aligner_model_id: str = Field(
        default="Qwen/Qwen3-ForcedAligner-0.6B",
        description="HuggingFace ForcedAligner 模型 ID",
    )
    asr_local_path: str | None = Field(
        default=None,
        description="ASR 模型本地路径（设置后优先于 asr_model_id；留空则从 HuggingFace 下载）",
    )
    aligner_local_path: str | None = Field(
        default=None,
        description="ForcedAligner 模型本地路径（设置后优先于 aligner_model_id；留空则不加载）",
    )
    dtype: Literal["bf16", "fp16"] = Field(
        default="bf16",
        description="模型推理精度（老 Turing/Pascal 显卡请用 fp16）",
    )
    device: Literal["cuda", "cpu"] = Field(
        default="cuda",
        description="推理设备",
    )
    use_forced_aligner: bool = Field(
        default=False,
        description="是否启用 ForcedAligner（多占 ~1.5GB 显存）",
    )

    @field_validator("dtype", mode="before")
    @classmethod
    def validate_dtype(cls, v: str) -> str:
        """校验 dtype 只能是 bf16 或 fp16（兼容别名）。"""
        v = str(v).strip().lower()
        if v in ("bf16", "bfloat16"):
            return "bf16"
        if v in ("fp16", "float16", "half"):
            return "fp16"
        raise ValueError(f"不支持的 dtype: {v}，请使用 bf16 或 fp16")

    @field_validator("device", mode="before")
    @classmethod
    def validate_device(cls, v: str) -> str:
        """校验 device 只能是 cuda 或 cpu。"""
        v = str(v).strip().lower()
        if v not in ("cuda", "cpu"):
            raise ValueError(f"不支持的 device: {v}，请使用 cuda 或 cpu")
        return v


class VADConfig(BaseModel):
    """Silero VAD 分段参数

    段后处理策略：不丢任何检测到的语音段（短语气词「嗯/对/啊」承载语义）。
    流程：合并相邻 → 短段救援（强制并入最近邻居）→ 切分超长 → 兜底过滤(<min_speech_duration)。
    """

    threshold: float = Field(
        default=0.5,
        ge=0.1,
        le=0.9,
        description="Silero VAD 语音概率阈值",
    )
    min_speech_duration: float = Field(
        default=0.2,
        ge=0.05,
        le=1.0,
        description="最小语音时长（秒），低于此值视为 VAD 误触发并丢弃（唯一丢弃环节）",
    )
    min_segment_length: float = Field(
        default=1.0,
        ge=0.5,
        le=10.0,
        description="短段救援的目标长度（秒），短于此值的段会强制并入最近邻居",
    )
    merge_gap: float = Field(
        default=0.5,
        ge=0.0,
        le=5.0,
        description="相邻段间隔小于此值则合并（秒）",
    )
    max_segment_length: float = Field(
        default=30.0,
        ge=5.0,
        le=300.0,
        description="VAD 段最大秒数，超过则强制切分",
    )
    overlap: float = Field(
        default=0.3,
        ge=0.0,
        le=5.0,
        description="段边界前后各扩展 overlap/2 秒，形成人为重叠区（供 Phase 2.6 LCS 去重使用）",
    )


class PipelineConfig(BaseModel):
    """流水线执行参数"""

    queue_size: int = Field(
        default=4,
        ge=1,
        le=20,
        description="生产者-消费者队列容量",
    )
    retry_times: int = Field(
        default=3,
        ge=0,
        le=10,
        description="单段推理失败自动重试次数",
    )


class OutputConfig(BaseModel):
    """输出格式参数"""

    formats: List[Literal["txt", "srt", "md"]] = Field(
        default=["txt", "srt", "md"],
        description="启用的输出格式",
    )
    srt_max_chars_per_line: int = Field(
        default=20,
        ge=5,
        le=100,
        description="SRT 每行最大字符数",
    )
    srt_max_duration: float = Field(
        default=5.0,
        ge=1.0,
        le=30.0,
        description="SRT 每条最长秒数",
    )
    output_dir: str = Field(
        default="./output",
        description="输出根目录",
    )


class DownloadConfig(BaseModel):
    """模型下载参数"""

    source: Literal["hf", "hf-mirror", "modelscope", "local", "custom"] = Field(
        default="hf-mirror",
        description="模型下载源",
    )
    custom_endpoint: str = Field(
        default="",
        description="自定义 HF 兼容 endpoint",
    )
    cache_dir: str = Field(
        default="./models",
        description="模型缓存目录",
    )

    @field_validator("source", mode="before")
    @classmethod
    def validate_source(cls, v: str) -> str:
        """校验下载源为合法值。"""
        allowed = {"hf", "hf-mirror", "modelscope", "local", "custom"}
        v = str(v).strip().lower()
        if v not in allowed:
            raise ValueError(
                f"不支持的下载源: {v}，请使用: {', '.join(sorted(allowed))}"
            )
        return v


# ============================================================
# 顶层聚合配置
# ============================================================

class VoxPenConfig(BaseModel):
    """VoxPen 应用总配置"""

    model: ModelConfig = Field(default_factory=ModelConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)


# ============================================================
# 工厂函数
# ============================================================

def load_config(config_path: Optional[str | Path] = None) -> VoxPenConfig:
    """
    加载配置文件。

    优先级：
    1. 传入的 config_path
    2. 项目根目录 config/default.yaml

    Args:
        config_path: 配置文件路径（可选）

    Returns:
        VoxPenConfig: 校验后的配置对象

    Raises:
        FileNotFoundError: 配置文件不存在
        yaml.YAMLError: YAML 格式错误
        pydantic.ValidationError: 配置校验失败
    """
    if config_path is None:
        config_path = Path(__file__).resolve().parent.parent / "config" / "default.yaml"

    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    if raw is None:
        raw = {}

    return VoxPenConfig.model_validate(raw)


def get_default_config() -> VoxPenConfig:
    """返回全默认值的配置对象（不读取文件）。"""
    return VoxPenConfig()
