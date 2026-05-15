"""
模型下载模块

基于 PRD v1.4 §6，支持 5 种下载源：
- hf: HuggingFace 官方
- hf-mirror: hf-mirror.com 镜像（国内推荐）
- modelscope: 魔搭社区
- local: 本地已有模型
- custom: 自定义 HF 兼容 endpoint

提供：下载、缓存检测、连通性测试、轻量文件校验。
"""

from __future__ import annotations

import socket
import time
from pathlib import Path
from typing import Callable, Literal, Optional
from urllib.parse import urlparse

from voxpen.config import DownloadConfig
from voxpen.utils.logger import get_logger

logger = get_logger("asr.downloader")

# ── 类型与常量 ────────────────────────────────────────────

DownloadSource = Literal["hf", "hf-mirror", "modelscope", "local", "custom"]
DownloadProgressCallback = Callable[[str, int, int], None]
# 签名: (file_name, files_done, files_total) — v1 粗粒度文件级

HF_OFFICIAL_ENDPOINT = "https://huggingface.co"
HF_MIRROR_ENDPOINT = "https://hf-mirror.com"
MODELSCOPE_ENDPOINT = "https://www.modelscope.cn"

CONNECTIVITY_TIMEOUT_SEC = 5

# ── 异常 ──────────────────────────────────────────────────


class DownloadError(RuntimeError):
    """下载失败的统一异常，附诊断信息。"""


class ModelNotFoundError(DownloadError):
    """模型路径无效或下载结果不完整。"""

# ── 主函数 ────────────────────────────────────────────────


def download_model(
    model_id: str,
    config: DownloadConfig,
    progress_callback: DownloadProgressCallback | None = None,
) -> Path:
    """
    下载模型权重到本地，返回本地路径。

    Args:
        model_id: HF 仓库 ID，如 "Qwen/Qwen3-ASR-1.7B"。
        config: DownloadConfig（含 source、custom_endpoint、cache_dir）。
        progress_callback: 文件级进度回调（v1 粗粒度，仅触发开始/结束两次）。

    Returns:
        模型在本地的绝对路径。

    Raises:
        DownloadError: 下载失败（网络/认证等）。
        ModelNotFoundError: source=local 时路径不存在；或下载后校验失败。
        ValueError: source 取值非法、custom 时 endpoint 为空。
    """
    model_stem = model_id.split("/")[-1]
    local_dir = (Path(config.cache_dir) / model_stem).resolve()

    # ── source=local: 直接返回本地路径 ──
    if config.source == "local":
        if not verify_model_files(local_dir):
            raise ModelNotFoundError(f"本地模型不存在或不完整: {local_dir}")
        logger.info(f"使用本地模型: {local_dir}")
        return local_dir

    # ── 确定 endpoint ──
    source: DownloadSource = config.source
    if source == "hf":
        endpoint = HF_OFFICIAL_ENDPOINT
    elif source == "hf-mirror":
        endpoint = HF_MIRROR_ENDPOINT
    elif source == "custom":
        if not config.custom_endpoint or not config.custom_endpoint.strip():
            raise ValueError("source=custom 但 custom_endpoint 未配置")
        endpoint = config.custom_endpoint.strip()
    elif source == "modelscope":
        endpoint = MODELSCOPE_ENDPOINT
    else:
        raise ValueError(f"不支持的下载源: {source!r}")

    # ── 触发进度回调（开始） ──
    if progress_callback:
        progress_callback(model_stem, 0, 1)

    t0 = time.perf_counter()
    logger.info(f"开始下载模型: {model_id} (source={source}, endpoint={endpoint})")

    try:
        if source == "modelscope":
            _download_from_modelscope(model_id, local_dir)
        else:
            _download_from_hf(model_id, local_dir, endpoint)

    except (DownloadError, ModelNotFoundError, ValueError):
        raise
    except Exception as e:
        logger.error(f"模型下载失败: {e}")
        raise DownloadError(
            f"模型下载失败 (source={source}, endpoint={endpoint}): {e}"
        ) from e

    elapsed = time.perf_counter() - t0
    logger.info(f"模型下载完成: {local_dir} ({elapsed:.1f}s)")

    # ── 校验 ──
    if not verify_model_files(local_dir):
        raise ModelNotFoundError(f"下载后校验失败，模型目录不完整: {local_dir}")

    # ── 触发进度回调（结束） ──
    if progress_callback:
        progress_callback(model_stem, 1, 1)

    return local_dir


def _download_from_hf(
    model_id: str,
    local_dir: Path,
    endpoint: str,
) -> None:
    """通过 huggingface_hub 下载。"""
    from huggingface_hub import snapshot_download

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        endpoint=endpoint,
        resume_download=True,
    )


def _download_from_modelscope(
    model_id: str,
    local_dir: Path,
) -> None:
    """通过 modelscope 下载。"""
    from modelscope import snapshot_download as ms_snapshot_download

    local_dir.parent.mkdir(parents=True, exist_ok=True)
    ms_snapshot_download(
        model_id=model_id,
        local_dir=str(local_dir),
    )

# ── 缓存检测 ──────────────────────────────────────────────


def is_model_downloaded(
    model_id: str,
    cache_dir: str | Path,
) -> bool:
    """
    检查模型是否已下载且完整。

    等价于：能否在 cache_dir/{model_stem} 找到一个有效模型目录。
    有效 = 存在 config.json 且至少一个 *.safetensors 文件。

    Args:
        model_id: 模型 ID 或本地路径。
        cache_dir: 缓存目录。

    Returns:
        True 表示模型已就位，可直接加载。
    """
    model_stem = model_id.split("/")[-1]
    local_dir = Path(cache_dir) / model_stem
    return verify_model_files(local_dir)

# ── 文件校验 ──────────────────────────────────────────────


def verify_model_files(model_dir: str | Path) -> bool:
    """
    轻量校验：模型目录下是否同时存在 config.json 和至少一个 *.safetensors。

    Args:
        model_dir: 模型目录路径。

    Returns:
        True 表示通过校验。不抛错。
    """
    model_dir = Path(model_dir)
    if not model_dir.is_dir():
        return False

    config_path = model_dir / "config.json"
    if not config_path.is_file():
        return False

    safetensors_files = list(model_dir.glob("*.safetensors"))
    if not safetensors_files:
        return False

    return True

# ── 连通性测试 ───────────────────────────────────────────


def check_connectivity(
    source: DownloadSource,
    custom_endpoint: str = "",
    timeout: int = CONNECTIVITY_TIMEOUT_SEC,
) -> bool:
    """
    测试下载源连通性。用 socket 连接对应 endpoint 的 443 端口。

    Args:
        source: 下载源标识。
        custom_endpoint: source=custom 时使用。
        timeout: 超时秒数，默认 5。

    Returns:
        True 表示连通，False 表示不可达。
    """
    if source == "local":
        logger.info("连通性测试: source=local，跳过（无需联网）")
        return True

    # 解析 hostname
    if source == "hf":
        hostname = "huggingface.co"
    elif source == "hf-mirror":
        hostname = "hf-mirror.com"
    elif source == "modelscope":
        hostname = "www.modelscope.cn"
    elif source == "custom":
        if not custom_endpoint or not custom_endpoint.strip():
            logger.warning("连通性测试: source=custom 但 custom_endpoint 为空")
            return False
        try:
            parsed = urlparse(custom_endpoint.strip())
            hostname = parsed.hostname or custom_endpoint.strip()
        except Exception:
            hostname = custom_endpoint.strip()
    else:
        logger.warning(f"连通性测试: 不支持的 source={source!r}")
        return False

    try:
        sock = socket.create_connection((hostname, 443), timeout=timeout)
        sock.close()
        logger.info(f"连通性测试: {hostname}:443 可达 (source={source})")
        return True
    except Exception as e:
        logger.warning(f"连通性测试: {hostname}:443 不可达 ({e})")
        return False
