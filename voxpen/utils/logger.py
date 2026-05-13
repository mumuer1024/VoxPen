"""
VoxPen 日志系统

提供结构化日志，支持：
- 控制台输出（INFO 级别以上）
- 按任务独立日志文件（DEBUG 级别，包含详细上下文）
- 统一的日志格式
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


# 全局日志格式
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_root_logger(
    level: int = logging.INFO,
    log_dir: Optional[str | Path] = None,
) -> logging.Logger:
    """
    初始化根日志器（控制台输出）。

    Args:
        level: 控制台日志级别
        log_dir: 全局日志目录（可选，未提供时仅输出到控制台）

    Returns:
        root logger
    """
    root = logging.getLogger("voxpen")
    root.setLevel(logging.DEBUG)  # 根级别设为 DEBUG，由 handler 控制实际输出级别

    # 避免重复添加 handler
    if not root.handlers:
        # 控制台 handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level)
        console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        root.addHandler(console)

        # 全局日志文件（如果指定了目录）
        if log_dir:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(
                log_dir / "voxpen.log", encoding="utf-8"
            )
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
            root.addHandler(file_handler)

    return root


def get_task_logger(
    task_name: str,
    log_dir: str | Path = "./output",
) -> logging.Logger:
    """
    为单个转录任务创建独立日志器。

    日志输出到 <log_dir>/<task_name>/task.log，
    同时继承根日志器的控制台输出。

    Args:
        task_name: 任务名称（通常为 原文件名_时间戳）
        log_dir: 日志根目录

    Returns:
        任务专属 logger
    """
    task_log_dir = Path(log_dir) / task_name
    task_log_dir.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(f"voxpen.task.{task_name}")
    logger.setLevel(logging.DEBUG)

    # 文件 handler（每个任务一个）
    file_handler = logging.FileHandler(
        task_log_dir / "task.log", encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(file_handler)

    # 不向父 logger 传播（避免重复输出）
    logger.propagate = False

    return logger


def get_logger(name: str) -> logging.Logger:
    """
    获取子模块 logger（继承 voxpen 根 logger 配置）。

    自动确保根日志器已初始化（幂等调用），
    无需手动调用 setup_root_logger()。

    Args:
        name: logger 名称（如 "voxpen.asr.transcriber"）

    Returns:
        对应 logger
    """
    # 自动确保根日志器已初始化（幂等调用，不会重复添加 handler）
    setup_root_logger()
    return logging.getLogger(f"voxpen.{name}")
