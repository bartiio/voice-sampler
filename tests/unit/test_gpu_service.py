"""
Unit tests for GPUService hardware detection, CUDA querying, and simulation modes.
"""

import os
from unittest.mock import MagicMock, patch
import pytest

from backend.app.services.gpu_service import GPUService, detect_system_gpu


@pytest.fixture
def service():
    s = GPUService()
    return s


def test_gpu_info_no_cuda_fallback(service):
    with patch("backend.app.services.gpu_service.TORCH_AVAILABLE", False):
        with patch.object(
            service,
            "detect_windows_hardware_gpu",
            return_value=("AMD Radeon 740M Graphics", 512),
        ):
            info = service.get_gpu_info(force_refresh=True)
            assert info["cuda_available"] is False
            assert info["gpu_name"] == "AMD Radeon 740M Graphics"
            assert info["vram_total_mb"] == 512
            assert info["vram_allocated_mb"] == 0
            assert info["cuda_version"] is None
            assert info["simulation_mode"] is True
            assert "No CUDA GPU detected" in info["message"]


def test_gpu_info_with_cuda_mocked(service):
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.6.0"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"
    mock_torch.cuda.mem_get_info.return_value = (
        20 * 1024 * 1024 * 1024,
        24 * 1024 * 1024 * 1024,
    )
    mock_torch.cuda.memory_allocated.return_value = 1024 * 1024 * 1024
    mock_torch.version.cuda = "12.6"

    with patch("backend.app.services.gpu_service.TORCH_AVAILABLE", True):
        with patch("backend.app.services.gpu_service.torch", mock_torch):
            with patch.dict(os.environ, {}, clear=True):
                info = service.get_gpu_info(force_refresh=True)
                assert info["cuda_available"] is True
                assert info["gpu_name"] == "NVIDIA GeForce RTX 4090"
                assert info["vram_total_mb"] == 24576
                assert info["vram_allocated_mb"] == 1024
                assert info["vram_free_mb"] == 20480
                assert info["cuda_version"] == "12.6"
                assert info["simulation_mode"] is False
                assert info["message"] is None


def test_dev_mode_override_flag(service):
    mock_torch = MagicMock()
    mock_torch.__version__ = "2.6.0"
    mock_torch.cuda.is_available.return_value = True
    mock_torch.cuda.device_count.return_value = 1
    mock_torch.cuda.get_device_name.return_value = "NVIDIA GeForce RTX 4090"
    mock_torch.cuda.mem_get_info.return_value = (1000, 2000)
    mock_torch.cuda.memory_allocated.return_value = 100
    mock_torch.version.cuda = "12.6"

    with patch("backend.app.services.gpu_service.TORCH_AVAILABLE", True):
        with patch("backend.app.services.gpu_service.torch", mock_torch):
            with patch.dict(os.environ, {"VOICE_SAMPLER_DEV_MODE": "1"}):
                info = service.get_gpu_info(force_refresh=True)
                assert info["cuda_available"] is True
                assert info["simulation_mode"] is True
                assert "simulation mode is forced" in info["message"]


def test_detect_system_gpu_helper():
    info = detect_system_gpu()
    assert isinstance(info, dict)
    assert "cuda_available" in info
    assert "gpu_name" in info
    assert "simulation_mode" in info
