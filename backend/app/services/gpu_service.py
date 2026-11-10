"""
backend/app/services/gpu_service.py

Hardware GPU and CUDA detection service for Voice Sampler.
Queries PyTorch CUDA capabilities with Windows WMI / CimInstance fallback
and manages simulation/dev mode flags.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import subprocess
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("voice_sampler.gpu_service")

# Safe PyTorch import
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore
    TORCH_AVAILABLE = False


class GPUService:
    """Service for querying GPU hardware, CUDA capability, and dev simulation flags."""

    _instance: Optional[GPUService] = None

    def __init__(self) -> None:
        self._cached_hardware_gpu: Optional[Tuple[Optional[str], Optional[int]]] = None
        self._cached_cuda_static: Optional[Dict[str, Any]] = None

    @classmethod
    def get_instance(cls) -> GPUService:
        if cls._instance is None:
            cls._instance = GPUService()
        return cls._instance

    def is_dev_mode_enabled(self) -> Optional[bool]:
        """
        Check if dev/simulation mode is explicitly enabled or disabled via env vars.
        Returns:
            True if explicitly enabled, False if explicitly disabled, None if unset.
        """
        # Primary flag: VOICE_SAMPLER_DEV_MODE
        env_val = os.getenv("VOICE_SAMPLER_DEV_MODE", "").strip().lower()
        if env_val in ("1", "true", "yes", "on"):
            return True
        if env_val in ("0", "false", "no", "off"):
            return False

        # Secondary flag: SIMULATE_TRAINING
        sim_val = os.getenv("SIMULATE_TRAINING", "").strip().lower()
        if sim_val in ("1", "true", "yes", "on"):
            return True
        if sim_val in ("0", "false", "no", "off"):
            return False

        # Tertiary flag: VOICE_SAMPLER_SIMULATION_MODE
        mode_val = os.getenv("VOICE_SAMPLER_SIMULATION_MODE", "").strip().lower()
        if mode_val in ("1", "true", "yes", "on"):
            return True
        if mode_val in ("0", "false", "no", "off"):
            return False

        return None

    def detect_windows_hardware_gpu(self) -> Tuple[Optional[str], Optional[int]]:
        """
        Detect host GPU name and total VRAM (in MB) using Windows CIM/WMI.
        Returns (gpu_name, vram_mb).
        """
        if platform.system() != "Windows":
            return None, None

        if self._cached_hardware_gpu is not None:
            return self._cached_hardware_gpu

        try:
            cmd = [
                "powershell",
                "-NoProfile",
                "-Command",
                "Get-CimInstance Win32_VideoController | Select-Object Name, AdapterRAM | ConvertTo-Json",
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=4)
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                chosen = None

                if isinstance(data, dict):
                    chosen = data
                elif isinstance(data, list) and len(data) > 0:
                    # Filter out generic/basic display drivers if a named adapter exists
                    filtered = [
                        d for d in data
                        if isinstance(d, dict) and "basic display" not in str(d.get("Name", "")).lower()
                    ]
                    chosen = filtered[0] if filtered else data[0]

                if chosen and isinstance(chosen, dict):
                    name = chosen.get("Name")
                    raw_ram = chosen.get("AdapterRAM")
                    vram_mb = None

                    if raw_ram is not None:
                        try:
                            ram_int = int(raw_ram)
                            # Handle signed 32-bit overflow on large VRAM adapters
                            if ram_int < 0:
                                ram_int = ram_int & 0xFFFFFFFF
                            if ram_int > 0:
                                vram_mb = round(ram_int / (1024 * 1024))
                        except (ValueError, TypeError):
                            vram_mb = None

                    if name:
                        self._cached_hardware_gpu = (str(name).strip(), vram_mb)
                        return self._cached_hardware_gpu

        except subprocess.TimeoutExpired:
            logger.warning("Windows CIM GPU detection timed out after 4 seconds.")
        except Exception as e:
            logger.debug("Windows CIM GPU detection failed: %s", e)

        self._cached_hardware_gpu = (None, None)
        return self._cached_hardware_gpu

    def get_gpu_info(self, force_refresh: bool = False) -> Dict[str, Any]:
        """
        Detect full GPU status and system capabilities.
        Matches exact schema required by R5 and PROJECT.md.
        """
        if force_refresh:
            self._cached_hardware_gpu = None
            self._cached_cuda_static = None

        cuda_available = False
        gpu_name: Optional[str] = None
        vram_total_mb: Optional[int] = None
        vram_allocated_mb: Optional[int] = 0
        vram_free_mb: Optional[int] = None
        cuda_version: Optional[str] = None
        torch_version: Optional[str] = None
        hardware_gpu_name: Optional[str] = None

        if TORCH_AVAILABLE and torch is not None:
            raw_version = getattr(torch, "__version__", None)
            torch_version = str(raw_version) if raw_version is not None else None
            try:
                cuda_available = bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
            except Exception as e:
                logger.warning("Exception calling torch.cuda.is_available(): %s", e)
                cuda_available = False

        if cuda_available and torch is not None:
            try:
                device_idx = 0
                gpu_name = torch.cuda.get_device_name(device_idx)
                cuda_version = getattr(torch.version, "cuda", None)

                # PyTorch 2.x memory info
                try:
                    free_bytes, total_bytes = torch.cuda.mem_get_info(device_idx)
                    vram_total_mb = round(total_bytes / (1024 * 1024))
                    vram_free_mb = round(free_bytes / (1024 * 1024))
                except Exception:
                    props = torch.cuda.get_device_properties(device_idx)
                    vram_total_mb = round(props.total_memory / (1024 * 1024))
                    vram_free_mb = None

                # Currently allocated VRAM
                try:
                    allocated_bytes = torch.cuda.memory_allocated(device_idx)
                    vram_allocated_mb = round(allocated_bytes / (1024 * 1024))
                except Exception:
                    vram_allocated_mb = 0

            except Exception as e:
                logger.error("Error inspecting CUDA device: %s", e)
                cuda_available = False

        # Fallback when CUDA is absent
        if not cuda_available:
            hw_name, hw_ram = self.detect_windows_hardware_gpu()
            hardware_gpu_name = hw_name
            gpu_name = hw_name or "None (CPU Only)"
            if hw_ram is not None:
                vram_total_mb = hw_ram
            elif hw_name and "Radeon 740M" in hw_name:
                vram_total_mb = 512
            else:
                vram_total_mb = 0
            vram_allocated_mb = 0
            vram_free_mb = vram_total_mb
            cuda_version = None

        # Resolve Dev / Simulation Mode
        dev_flag = self.is_dev_mode_enabled()
        if dev_flag is not None:
            simulation_mode = dev_flag
        else:
            simulation_mode = not cuda_available

        # Construct status / warning message
        if not cuda_available:
            message = "No CUDA GPU detected. GPU is required for real training. Running in simulation mode."
        elif simulation_mode:
            message = "CUDA GPU is available, but simulation mode is forced via environment configuration."
        else:
            message = None

        return {
            "cuda_available": cuda_available,
            "gpu_name": gpu_name,
            "vram_total_mb": vram_total_mb,
            "vram_allocated_mb": vram_allocated_mb,
            "cuda_version": cuda_version,
            "simulation_mode": simulation_mode,
            "message": message,
            # Backward & cross-spec compatibility aliases
            "device_name": gpu_name,
            "hardware_gpu_name": hardware_gpu_name,
            "vram_free_mb": vram_free_mb,
            "warning_message": message,
            "torch_version": torch_version,
        }


# Global singleton helpers
def get_gpu_service() -> GPUService:
    """Return GPUService singleton instance."""
    return GPUService.get_instance()


def detect_system_gpu(force_refresh: bool = False) -> Dict[str, Any]:
    """Helper to detect and return system GPU info."""
    return get_gpu_service().get_gpu_info(force_refresh=force_refresh)
