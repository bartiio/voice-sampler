"""
backend/app/routers/generate.py
REST API endpoints for Voice Generation (F5-TTS and RVC v2 STS), history tracking,
and metadata management.
Part of Voice Sampler Milestone 4 (M4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import unicodedata

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.routers.audio import audio_router
from backend.app.services.f5tts_service import (
    ModelNotFoundError as F5TTSModelNotFoundError,
    SampleNotFoundError as F5TTSSampleNotFoundError,
    get_f5tts_service,
)
from backend.app.services.rvc_service import (
    ModelNotFoundError as RVCModelNotFoundError,
    SampleNotFoundError as RVCSampleNotFoundError,
    get_rvc_service,
)

logger = logging.getLogger("voice_sampler.routers.generate")
router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class GenerateTTSRequest(BaseModel):
    text: str = Field(..., description="Text prompt to synthesize into speech")
    model_name: str = Field(..., description="Name of trained F5-TTS model")
    reference_sample_id: Optional[str] = Field(None, description="Optional UUID of reference audio sample")
    speed: float = Field(default=1.0, description="Speech rate multiplier (default 1.0)")


class GenerateSTSRequest(BaseModel):
    input_sample_id: str = Field(..., description="UUID of source audio sample to convert")
    model_name: str = Field(..., description="Name of trained RVC v2 model")
    pitch_shift: int = Field(default=0, description="Pitch transposition in semitones (-24 to +24)")
    index_rate: float = Field(default=0.75, description="Feature retrieval blending ratio (0.0 to 1.0)")


class GenerateResponse(BaseModel):
    generation_id: str = Field(..., description="Unique UUID of generation")
    audio_url: str = Field(..., description="Relative streaming URL: /api/audio/generated/{filename}")
    filename: str = Field(..., description="Filename on disk, e.g. tts_{generation_id[:8]}.wav")
    duration: Optional[float] = Field(None, description="Audio duration in seconds")
    sample_rate: Optional[int] = Field(None, description="Audio sample rate in Hz")


class GenerationMetadataResponse(BaseModel):
    id: str
    type: str
    model_id: str
    input_text: Optional[str] = None
    input_audio_path: Optional[str] = None
    reference_sample_id: Optional[str] = None
    output_audio_path: str
    output_filename: str
    duration_seconds: float
    sample_rate: int
    channels: int
    parameters: Dict[str, Any]
    created_at: str


class GenerationListResponse(BaseModel):
    generations: List[GenerationMetadataResponse] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Generation Endpoints (Mounted at /api/generate)
# ---------------------------------------------------------------------------

@router.post("/tts", response_model=GenerateResponse, status_code=status.HTTP_200_OK)
def generate_tts(req: GenerateTTSRequest):
    """
    Synthesize speech from text using trained F5-TTS model (or high-fidelity simulation).
    Validates non-empty text, model existence, and reference sample.
    """
    # Explicit validation for strict HTTP 400 compliance
    if not req.text or not req.text.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Text cannot be empty",
        )

    if not req.model_name or not req.model_name.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model name cannot be empty",
        )

    f5_service = get_f5tts_service()
    try:
        result = f5_service.generate_tts(
            text=req.text,
            model_name=req.model_name,
            reference_sample_id=req.reference_sample_id,
            speed=req.speed,
        )
        return {
            "generation_id": result["generation_id"],
            "audio_url": result["audio_url"],
            "filename": result["filename"],
            "duration": result.get("duration"),
            "sample_rate": result.get("sample_rate", 24000),
        }
    except (F5TTSModelNotFoundError, F5TTSSampleNotFoundError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


@router.post("/sts", response_model=GenerateResponse, status_code=status.HTTP_200_OK)
def generate_sts(req: GenerateSTSRequest):
    """
    Convert speech voice using trained RVC v2 model (or high-fidelity simulation).
    Validates input sample existence, model existence, and parameter boundaries.
    """
    # Explicit boundary validation for strict HTTP 400 compliance
    if not req.input_sample_id or not str(req.input_sample_id).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Input sample ID cannot be empty",
        )

    if not req.model_name or not str(req.model_name).strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Model name cannot be empty",
        )

    if req.pitch_shift < -24 or req.pitch_shift > 24:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Pitch shift must be between -24 and +24 semitones",
        )

    if req.index_rate < 0.0 or req.index_rate > 1.0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Index rate must be between 0.0 and 1.0",
        )

    rvc_service = get_rvc_service()
    try:
        result = rvc_service.convert_voice(
            input_sample_id=req.input_sample_id,
            model_name=req.model_name,
            pitch_shift=req.pitch_shift,
            index_rate=req.index_rate,
        )
        return {
            "generation_id": result["generation_id"],
            "audio_url": result["audio_url"],
            "filename": result["filename"],
            "duration": result.get("duration"),
            "sample_rate": result.get("sample_rate", 40000),
        }
    except (RVCModelNotFoundError, RVCSampleNotFoundError) as e:
        raise HTTPException(status_code=e.status_code, detail=e.detail)
    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))


# ---------------------------------------------------------------------------
# History & Metadata Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=GenerationListResponse)
@router.get("/history", response_model=GenerationListResponse)
def list_generation_history(
    type: Optional[str] = Query(None, description="Filter by 'tts' or 'sts'"),
    limit: int = Query(50, ge=1, le=200),
):
    """List recent speech generation records ordered by created_at DESC."""
    rows = repository.list_generations(gen_type=type, limit=limit)
    return {"generations": rows}


@router.get("/{generation_id}", response_model=GenerationMetadataResponse)
def get_generation_metadata(generation_id: str):
    """Retrieve metadata of a specific generation by UUID."""
    record = repository.get_generation(generation_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation record not found")
    return record


@router.delete("/{generation_id}", status_code=status.HTTP_200_OK)
def delete_generation_record(generation_id: str):
    """Delete generation record and its generated audio file from disk."""
    record = repository.get_generation(generation_id)
    if not record:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Generation record not found")

    settings = get_settings()
    filename = record.get("output_filename")
    candidate_paths = []
    if record.get("output_audio_path"):
        p = Path(record["output_audio_path"])
        candidate_paths.append(p)
        if not p.is_absolute():
            candidate_paths.append((settings.project_root / p).resolve())
            candidate_paths.append((settings.data_dir / p.name).resolve())
    if filename:
        candidate_paths.append((settings.generated_dir / filename).resolve())
        candidate_paths.append((settings.generated_tts_dir / filename).resolve())
        candidate_paths.append((settings.generated_sts_dir / filename).resolve())

    for cp in candidate_paths:
        try:
            if cp.is_file():
                cp.unlink()
        except Exception as e:
            logger.warning(f"Failed to unlink generated file {cp}: {e}")

    deleted = repository.delete_generation(generation_id)
    return {"status": "deleted", "id": generation_id}
