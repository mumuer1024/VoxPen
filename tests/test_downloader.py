"""
Model downloader 单元测试。

mock huggingface_hub / modelscope / socket，
用 tmp_path 做真实文件系统测试。
"""

from __future__ import annotations

import socket
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from voxpen.asr.downloader import (
    CONNECTIVITY_TIMEOUT_SEC,
    HF_MIRROR_ENDPOINT,
    HF_OFFICIAL_ENDPOINT,
    MODELSCOPE_ENDPOINT,
    DownloadError,
    ModelNotFoundError,
    download_model,
    is_model_downloaded,
    check_connectivity,
    verify_model_files,
)
from voxpen.config import DownloadConfig


# ── helpers ────────────────────────────────────────────────


def _make_download_config(
    source: str = "hf-mirror",
    custom_endpoint: str = "",
    cache_dir: str = "./models",
) -> DownloadConfig:
    return DownloadConfig(source=source, custom_endpoint=custom_endpoint, cache_dir=cache_dir)


def _create_fake_model_dir(base: Path, model_id: str) -> Path:
    """创建假的模型目录（含 config.json + dummy.safetensors）。"""
    model_dir = base / model_id.split("/")[-1]
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_text("dummy")
    return model_dir


# ── DownloadModel 测试 ────────────────────────────────────


class TestDownloadModel:
    """download_model 测试。"""

    def test_download_local_existing(self, tmp_path):
        """source=local，目录存在且完整 → 返回路径。"""
        model_dir = _create_fake_model_dir(tmp_path, "Qwen/Qwen3-ASR-1.7B")
        cfg = _make_download_config(source="local", cache_dir=str(tmp_path))
        result = download_model("Qwen/Qwen3-ASR-1.7B", cfg)
        assert result.resolve() == model_dir.resolve()

    def test_download_local_missing(self, tmp_path):
        """source=local，目录不存在 → ModelNotFoundError。"""
        cfg = _make_download_config(source="local", cache_dir=str(tmp_path))
        with pytest.raises(ModelNotFoundError, match="本地模型不存在"):
            download_model("Qwen/Nonexistent-Model", cfg)

    def test_download_hf_calls_snapshot_with_official_endpoint(self, mocker, tmp_path):
        """source=hf → 验证 snapshot_download 参数。"""
        mock_snapshot = mocker.patch("huggingface_hub.snapshot_download")
        _create_fake_model_dir(tmp_path, "Qwen/Qwen3-ASR-1.7B")
        cfg = _make_download_config(source="hf", cache_dir=str(tmp_path))

        download_model("Qwen/Qwen3-ASR-1.7B", cfg)

        call_kwargs = mock_snapshot.call_args.kwargs
        assert call_kwargs["repo_id"] == "Qwen/Qwen3-ASR-1.7B"
        assert call_kwargs["endpoint"] == HF_OFFICIAL_ENDPOINT
        assert call_kwargs["resume_download"] is True

    def test_download_hf_mirror_uses_mirror_endpoint(self, mocker, tmp_path):
        """source=hf-mirror → 验证 endpoint。"""
        mock_snapshot = mocker.patch("huggingface_hub.snapshot_download")
        _create_fake_model_dir(tmp_path, "Qwen/Qwen3-ASR-1.7B")
        cfg = _make_download_config(source="hf-mirror", cache_dir=str(tmp_path))

        download_model("Qwen/Qwen3-ASR-1.7B", cfg)

        assert mock_snapshot.call_args.kwargs["endpoint"] == HF_MIRROR_ENDPOINT

    def test_download_modelscope_calls_ms_snapshot(self, mocker, tmp_path):
        """source=modelscope → 验证 modelscope.snapshot_download 调用。"""
        mock_snapshot = mocker.patch("modelscope.snapshot_download")
        _create_fake_model_dir(tmp_path, "Qwen/Qwen3-ASR-1.7B")
        cfg = _make_download_config(source="modelscope", cache_dir=str(tmp_path))

        download_model("Qwen/Qwen3-ASR-1.7B", cfg)

        assert mock_snapshot.call_args.kwargs["model_id"] == "Qwen/Qwen3-ASR-1.7B"

    def test_download_custom_with_empty_endpoint_raises(self, tmp_path):
        """source=custom + 空 endpoint → ValueError。"""
        cfg = _make_download_config(source="custom", custom_endpoint="", cache_dir=str(tmp_path))
        with pytest.raises(ValueError, match="custom_endpoint"):
            download_model("Qwen/Qwen3-ASR-1.7B", cfg)

    def test_download_custom_uses_custom_endpoint(self, mocker, tmp_path):
        """source=custom → 验证使用了自定义 endpoint。"""
        mock_snapshot = mocker.patch("huggingface_hub.snapshot_download")
        _create_fake_model_dir(tmp_path, "Qwen/Qwen3-ASR-1.7B")
        cfg = _make_download_config(source="custom", custom_endpoint="https://my.endpoint.com", cache_dir=str(tmp_path))

        download_model("Qwen/Qwen3-ASR-1.7B", cfg)

        assert mock_snapshot.call_args.kwargs["endpoint"] == "https://my.endpoint.com"

    def test_download_network_error_wrapped(self, mocker, tmp_path):
        """snapshot_download 抛 ConnectionError → 包成 DownloadError。"""
        mock_snapshot = mocker.patch("huggingface_hub.snapshot_download", side_effect=ConnectionError("boom"))
        _create_fake_model_dir(tmp_path, "Qwen/Qwen3-ASR-1.7B")
        cfg = _make_download_config(source="hf", cache_dir=str(tmp_path))

        with pytest.raises(DownloadError, match="boom"):
            download_model("Qwen/Qwen3-ASR-1.7B", cfg)

    def test_download_verification_failed(self, mocker, tmp_path):
        """snapshot_download 正常但本地无 config.json → ModelNotFoundError。"""
        mock_snapshot = mocker.patch("huggingface_hub.snapshot_download")
        # 不创建 fake 模型目录 → 下载"完成"后校验失败
        cfg = _make_download_config(source="hf", cache_dir=str(tmp_path))

        with pytest.raises(ModelNotFoundError, match="校验失败"):
            download_model("Qwen/Qwen3-ASR-1.7B", cfg)

    def test_download_progress_callback_invoked(self, mocker, tmp_path):
        """验证 callback 被调用两次。"""
        mocker.patch("huggingface_hub.snapshot_download")
        _create_fake_model_dir(tmp_path, "Qwen/Qwen3-ASR-1.7B")
        cfg = _make_download_config(source="hf", cache_dir=str(tmp_path))

        calls = []

        def cb(name, done, total):
            calls.append((name, done, total))

        download_model("Qwen/Qwen3-ASR-1.7B", cfg, progress_callback=cb)

        assert len(calls) == 2
        assert calls[0] == ("Qwen3-ASR-1.7B", 0, 1)
        assert calls[1] == ("Qwen3-ASR-1.7B", 1, 1)


# ── IsModelDownloaded 测试 ────────────────────────────────


class TestIsModelDownloaded:
    """is_model_downloaded 测试。"""

    def test_is_downloaded_true(self, tmp_path):
        _create_fake_model_dir(tmp_path, "Qwen/Qwen3-ASR-1.7B")
        assert is_model_downloaded("Qwen/Qwen3-ASR-1.7B", tmp_path) is True

    def test_is_downloaded_false_no_dir(self, tmp_path):
        assert is_model_downloaded("Qwen/Nonexistent", tmp_path) is False

    def test_is_downloaded_false_missing_config(self, tmp_path):
        model_dir = tmp_path / "Qwen3-ASR-1.7B"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_text("dummy")
        assert is_model_downloaded("Qwen/Qwen3-ASR-1.7B", tmp_path) is False

    def test_is_downloaded_false_missing_safetensors(self, tmp_path):
        model_dir = tmp_path / "Qwen3-ASR-1.7B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        assert is_model_downloaded("Qwen/Qwen3-ASR-1.7B", tmp_path) is False


# ── VerifyModelFiles 测试 ─────────────────────────────────


class TestVerifyModelFiles:
    """verify_model_files 测试。"""

    def test_verify_pass(self, tmp_path):
        model_dir = _create_fake_model_dir(tmp_path, "Qwen/Qwen3-ASR-1.7B")
        assert verify_model_files(model_dir) is True

    def test_verify_missing_config_json(self, tmp_path):
        model_dir = tmp_path / "Qwen3-ASR-1.7B"
        model_dir.mkdir()
        (model_dir / "model.safetensors").write_text("dummy")
        assert verify_model_files(model_dir) is False

    def test_verify_no_safetensors(self, tmp_path):
        model_dir = tmp_path / "Qwen3-ASR-1.7B"
        model_dir.mkdir()
        (model_dir / "config.json").write_text("{}")
        (model_dir / "pytorch_model.bin").write_text("dummy")
        assert verify_model_files(model_dir) is False

    def test_verify_nonexistent_dir(self, tmp_path):
        assert verify_model_files(tmp_path / "no_such_dir") is False


# ── TestConnectivity 测试 ─────────────────────────────────


class TestConnectivity:
    """test_connectivity 测试。"""

    def test_connectivity_local_always_true(self, mocker):
        """source=local → True，不调 socket。"""
        mock_sock = mocker.patch("socket.create_connection")
        assert check_connectivity("local") is True
        mock_sock.assert_not_called()

    def test_connectivity_hf_mirror_ok(self, mocker):
        """mock socket 成功 → True。"""
        mocker.patch("socket.create_connection")
        assert check_connectivity("hf-mirror") is True

    def test_connectivity_timeout_returns_false(self, mocker):
        """socket 超时 → False（不抛错）。"""
        mocker.patch("socket.create_connection", side_effect=socket.timeout("timeout"))
        assert check_connectivity("hf") is False

    def test_connectivity_custom_empty_endpoint_returns_false(self):
        """source=custom + 空 endpoint → False。"""
        assert check_connectivity("custom", custom_endpoint="") is False

    def test_connectivity_custom_with_endpoint(self, mocker):
        """source=custom → 验证 hostname 提取正确。"""
        mock_sock = mocker.patch("socket.create_connection")
        assert check_connectivity("custom", custom_endpoint="https://my.endpoint.com") is True
        # 验证 hostname 提取为 my.endpoint.com
        call_args = mock_sock.call_args[0]
        assert call_args[0] == ("my.endpoint.com", 443)
