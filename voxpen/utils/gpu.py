"""
GPU 工具

显存监测、CUDA 可用性检测。
依赖 PyTorch（torch.cuda）。
"""

from __future__ import annotations

from typing import Dict, Optional


def check_cuda_available() -> bool:
    """
    检测 CUDA 是否可用。

    Returns:
        True 如果 PyTorch 能找到 CUDA 设备
    """
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def get_gpu_count() -> int:
    """
    获取可用 GPU 数量。

    Returns:
        GPU 数量
    """
    try:
        import torch
        return torch.cuda.device_count()
    except ImportError:
        return 0


def get_gpu_info() -> Dict[str, any]:
    """
    获取 GPU 详细信息。

    Returns:
        {
            "cuda_available": bool,
            "gpu_count": int,
            "devices": [
                {
                    "index": 0,
                    "name": str,
                    "total_vram_gb": float,
                    "free_vram_gb": float,
                    "used_vram_gb": float,
                },
                ...
            ],
        }
    """
    info: Dict[str, any] = {
        "cuda_available": False,
        "gpu_count": 0,
        "devices": [],
    }

    try:
        import torch
    except ImportError:
        return info

    info["cuda_available"] = torch.cuda.is_available()
    if not info["cuda_available"]:
        return info

    gpu_count = torch.cuda.device_count()
    info["gpu_count"] = gpu_count

    for i in range(gpu_count):
        device_info: Dict[str, any] = {
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "total_vram_gb": 0.0,
            "free_vram_gb": 0.0,
            "used_vram_gb": 0.0,
        }

        # 尝试通过 PyTorch API 获取显存信息（PyTorch >= 2.0）
        try:
            mem_info = torch.cuda.mem_get_info(i)  # (free, total) in bytes
            free_bytes, total_bytes = mem_info
            device_info["total_vram_gb"] = round(total_bytes / (1024**3), 2)
            device_info["free_vram_gb"] = round(free_bytes / (1024**3), 2)
            device_info["used_vram_gb"] = round(
                (total_bytes - free_bytes) / (1024**3), 2
            )
        except (AttributeError, RuntimeError):
            # PyTorch < 2.0 或无法获取显存信息
            try:
                total_bytes = torch.cuda.get_device_properties(i).total_memory
                device_info["total_vram_gb"] = round(total_bytes / (1024**3), 2)
                # 无法获取 free/used 信息
            except Exception:
                pass

        info["devices"].append(device_info)

    return info


def get_free_vram_gb(device_index: int = 0) -> float:
    """
    获取指定 GPU 的剩余显存（GB）。

    Args:
        device_index: GPU 索引（默认 0）

    Returns:
        剩余显存 GB，获取失败返回 0.0
    """
    try:
        import torch
        mem_info = torch.cuda.mem_get_info(device_index)
        free_bytes, _ = mem_info
        return round(free_bytes / (1024**3), 2)
    except Exception:
        return 0.0


def get_total_vram_gb(device_index: int = 0) -> float:
    """
    获取指定 GPU 的总显存（GB）。

    Args:
        device_index: GPU 索引（默认 0）

    Returns:
        总显存 GB，获取失败返回 0.0
    """
    try:
        import torch
        mem_info = torch.cuda.mem_get_info(device_index)
        _, total_bytes = mem_info
        return round(total_bytes / (1024**3), 2)
    except Exception:
        return 0.0


def suggest_max_segment_length(
    target_free_vram_gb: float = 4.0,
    device_index: int = 0,
) -> float:
    """
    根据当前剩余显存建议 max_segment_length。

    启发式规则：
    - 剩余 >= 8GB  → 30 秒（默认）
    - 剩余 >= 6GB  → 20 秒
    - 剩余 >= 4GB  → 10 秒
    - 剩余 <  4GB  → 5 秒

    Args:
        target_free_vram_gb: 目标预留显存
        device_index: GPU 索引

    Returns:
        建议的 max_segment_length（秒）
    """
    free = get_free_vram_gb(device_index)
    if free <= 0:
        return 30.0  # 无法检测，用默认值

    if free >= 8.0:
        return 30.0
    elif free >= 6.0:
        return 20.0
    elif free >= 4.0:
        return 10.0
    else:
        return 5.0
