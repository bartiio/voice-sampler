"""
backend/app/routers/train.py
REST API endpoints for ML model training lifecycle management.
Supports F5-TTS and RVC v2 training initiation, cancellation, and status tracking.
Part of Voice Sampler Milestone 3 (M3).
"""

from __future__ import annotations

import logging
import re
import threading
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from backend.app.db import repository
from backend.app.services.f5tts_service import get_f5tts_service
from backend.app.services.rvc_service import get_rvc_service

logger = logging.getLogger("voice_sampler.routers.train")
router = APIRouter()

# Global coordinator lock ensuring atomic training initiation and stop across engines
_training_coordinator_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Request and Response Schemas
# ---------------------------------------------------------------------------

class TrainF5TTSRequest(BaseModel):
    """Payload for initiating F5-TTS fine-tuning."""
    model_name: str = Field(..., description="Unique model name")
    epochs: int = Field(default=10, description="Training epochs (>= 1)")
    batch_size: int = Field(default=2, description="Batch size (>= 1)")
    learning_rate: float = Field(default=0.0001, description="Learning rate (> 0)")
    sample_ids: Optional[List[str]] = Field(None, description="Optional list of sample IDs to train on")


class TrainRVCRequest(BaseModel):
    """Payload for initiating RVC v2 training."""
    model_name: str = Field(..., description="Unique model name")
    epochs: int = Field(default=20, description="Training epochs (>= 1)")
    batch_size: int = Field(default=4, description="Batch size (>= 1)")
    sample_rate: int = Field(default=40000, description="Audio sample rate (32000, 40000, or 48000)")
    sample_ids: Optional[List[str]] = Field(None, description="Optional list of sample IDs to train on")


class TrainStartResponse(BaseModel):
    """Response returned upon successfully queueing a training job."""
    run_id: str
    status: str = "started"
    engine: str
    model_name: str


class StopTrainingResponse(BaseModel):
    """Response returned upon requesting training run cancellation."""
    status: str
    run_id: Optional[str] = None


class TrainingStatusResponse(BaseModel):
    """Response returned by GET /api/train/status."""
    active: bool
    run: Optional[Dict[str, Any]] = None


# ---------------------------------------------------------------------------
# Router Endpoints
# ---------------------------------------------------------------------------

@router.post(
    "/f5-tts",
    response_model=TrainStartResponse,
    status_code=status.HTTP_200_OK,
    summary="Start F5-TTS model fine-tuning",
)
def start_f5tts_training(req: TrainF5TTSRequest) -> Dict[str, Any]:
    """
    Validates hyperparameters, verifies sample existence, registers run,
    and initiates F5-TTS dataset preparation and fine-tuning.
    """
    # 1. Validate model_name
    model_name = (req.model_name or "").strip()
    if not model_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model name cannot be empty",
        )
    if not re.match(r"^[a-zA-Z0-9_\-]+$", model_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model name '{model_name}'. Only alphanumeric characters, dashes, and underscores allowed.",
        )

    # 2. Validate numeric hyperparameters
    if req.epochs < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Epochs must be >= 1",
        )
    if req.batch_size < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size must be >= 1",
        )
    if req.learning_rate <= 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Learning rate must be > 0",
        )

    # 3. Validate sample_ids in SQLite
    if req.sample_ids is not None:
        if len(req.sample_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="sample_ids list cannot be empty",
            )
        for sid in req.sample_ids:
            if not repository.get_sample(sid):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Sample id {sid} not found",
                )

    # 4. Check for active run conflict across all engines under coordinator lock
    with _training_coordinator_lock:
        f5_svc = get_f5tts_service()
        rvc_svc = get_rvc_service()
        if f5_svc.is_training_active() or rvc_svc.is_training_active():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A training run is already in progress",
            )

        try:
            result = f5_svc.start_training(
                model_name=model_name,
                epochs=req.epochs,
                batch_size=req.batch_size,
                learning_rate=req.learning_rate,
                sample_ids=req.sample_ids,
            )
            return result
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
        except RuntimeError as re_err:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(re_err))
        except Exception as exc:
            logger.error(f"Failed to start F5-TTS training: {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal error starting F5-TTS training: {exc}",
            )


@router.post(
    "/rvc",
    response_model=TrainStartResponse,
    status_code=status.HTTP_200_OK,
    summary="Start RVC v2 model training",
)
def start_rvc_training(req: TrainRVCRequest) -> Dict[str, Any]:
    """
    Validates hyperparameters, verifies sample existence, registers run,
    and initiates the RVC v2 5-stage training pipeline.
    """
    # 1. Validate model_name
    model_name = (req.model_name or "").strip()
    if not model_name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model name cannot be empty",
        )
    if not re.match(r"^[a-zA-Z0-9_\-]+$", model_name):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid model name '{model_name}'. Only alphanumeric characters, dashes, and underscores allowed.",
        )

    # 2. Validate numeric hyperparameters
    if req.epochs < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Epochs must be >= 1",
        )
    if req.batch_size < 1:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Batch size must be >= 1",
        )
    if req.sample_rate not in (32000, 40000, 48000):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Sample rate must be 32k, 40k, or 48k",
        )

    # 3. Validate sample_ids in SQLite
    if req.sample_ids is not None:
        if len(req.sample_ids) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="sample_ids list cannot be empty",
            )
        for sid in req.sample_ids:
            if not repository.get_sample(sid):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Sample id {sid} not found",
                )

    # 4. Check for active run conflict across all engines under coordinator lock
    with _training_coordinator_lock:
        f5_svc = get_f5tts_service()
        rvc_svc = get_rvc_service()
        if f5_svc.is_training_active() or rvc_svc.is_training_active():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="A training run is already in progress",
            )

        try:
            result = rvc_svc.start_training(
                model_name=model_name,
                epochs=req.epochs,
                batch_size=req.batch_size,
                sample_rate=req.sample_rate,
                sample_ids=req.sample_ids,
            )
            return result
        except ValueError as ve:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(ve))
        except RuntimeError as re_err:
            raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(re_err))
        except Exception as exc:
            logger.error(f"Failed to start RVC training: {exc}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"Internal error starting RVC training: {exc}",
            )


@router.post(
    "/stop",
    response_model=StopTrainingResponse,
    status_code=status.HTTP_200_OK,
    summary="Stop active training run",
)
def stop_training() -> Dict[str, Any]:
    """
    Cooperatively stops the active training run if one is executing.
    Returns status 'stopped' with run_id, or 'no_active_run' if idle.
    """
    with _training_coordinator_lock:
        f5_svc = get_f5tts_service()
        rvc_svc = get_rvc_service()

        stopped_id: Optional[str] = None

        if f5_svc.is_training_active():
            stopped_id = f5_svc.get_active_run_id()
            f5_svc.stop_training()
        elif rvc_svc.is_training_active():
            stopped_id = rvc_svc.get_active_run_id()
            rvc_svc.cancel_training()
        else:
            # Check SQLite if a run is marked queued or running
            active_db_run = repository.get_active_training_run()
            if active_db_run:
                stopped_id = active_db_run["id"]
                repository.complete_training_run(
                    stopped_id,
                    status="stopped",
                    error_message="Training stopped by user request",
                )

        if stopped_id:
            return {"status": "stopped", "run_id": stopped_id}
        return {"status": "no_active_run"}


@router.get(
    "/status",
    response_model=TrainingStatusResponse,
    status_code=status.HTTP_200_OK,
    summary="Query active training status",
)
def get_training_status() -> Dict[str, Any]:
    """
    Returns details on active training run or indicates idle status.
    """
    f5_svc = get_f5tts_service()
    rvc_svc = get_rvc_service()

    is_active = f5_svc.is_training_active() or rvc_svc.is_training_active()
    active_run = repository.get_active_training_run()

    return {
        "active": is_active or (active_run is not None),
        "run": active_run,
    }
