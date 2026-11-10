"""
backend/app/routers/settings.py

Settings and System Hardware API endpoints for Voice Sampler.
Implements GET /api/settings/gpu conforming to R5 and PROJECT.md specifications.
"""

from __future__ import annotations

from typing import Any, Dict, Optional
from fastapi import APIRouter, Query, status
from pydantic import BaseModel, Field

from backend.app.config import get_settings
from backend.app.services.gpu_service import get_gpu_service

router = APIRouter()


class GPUInfoResponse(BaseModel):
    """Schema for GPU diagnostic information and simulation status."""
    cuda_available: bool = Field(
        ...,
        description="Whether PyTorch detected a CUDA-capable NVIDIA GPU."
    )
    gpu_name: Optional[str] = Field(
        None,
        description="Name of the active GPU (NVIDIA CUDA device name or host hardware GPU name)."
    )
    vram_total_mb: Optional[int] = Field(
        None,
        description="Total Video RAM in megabytes."
    )
    vram_allocated_mb: Optional[int] = Field(
        0,
        description="Video RAM currently allocated by PyTorch in megabytes."
    )
    cuda_version: Optional[str] = Field(
        None,
        description="CUDA runtime version reported by PyTorch."
    )
    simulation_mode: bool = Field(
        ...,
        description="Whether simulation mode is active for training and inference pipelines."
    )
    message: Optional[str] = Field(
        None,
        description="Informative message or warning when CUDA is absent or simulation mode is active."
    )

    # Compatibility fields
    device_name: Optional[str] = Field(None, description="Synonym for gpu_name.")
    hardware_gpu_name: Optional[str] = Field(None, description="Host hardware GPU name detected via OS.")
    vram_free_mb: Optional[int] = Field(None, description="Available free Video RAM in megabytes.")
    warning_message: Optional[str] = Field(None, description="Synonym for message.")
    torch_version: Optional[str] = Field(None, description="Installed PyTorch version.")


class AppSettingsResponse(BaseModel):
    """Schema for general system settings and paths."""
    app_name: str
    version: str
    dev_mode: bool
    simulation_mode: bool
    data_dir: str
    db_path: str
    gpu: GPUInfoResponse


@router.get(
    "/gpu",
    response_model=GPUInfoResponse,
    status_code=status.HTTP_200_OK,
    summary="GPU Hardware & CUDA Diagnostics",
    description="Returns GPU model, VRAM statistics, CUDA availability, and dev simulation flags.",
)
async def get_gpu_settings(
    refresh: bool = Query(
        False,
        description="Force hardware re-scan instead of using cached device information."
    )
) -> GPUInfoResponse:
    """
    Retrieve GPU and CUDA system status.
    Used by the Settings view to display GPU information or warning banners.
    """
    service = get_gpu_service()
    data = service.get_gpu_info(force_refresh=refresh)
    return GPUInfoResponse(**data)


@router.get(
    "",
    response_model=AppSettingsResponse,
    status_code=status.HTTP_200_OK,
    summary="General Application Settings",
    description="Returns general backend configuration, storage paths, and GPU summary.",
)
async def get_general_settings() -> AppSettingsResponse:
    """Return general application settings and configuration."""
    settings = get_settings()
    service = get_gpu_service()
    gpu_data = service.get_gpu_info()
    gpu_response = GPUInfoResponse(**gpu_data)

    return AppSettingsResponse(
        app_name=settings.app_name,
        version=settings.app_version,
        dev_mode=settings.dev_mode,
        simulation_mode=gpu_data.get("simulation_mode", True),
        data_dir=str(settings.data_dir),
        db_path=str(settings.db_path),
        gpu=gpu_response,
    )
