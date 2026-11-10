"""
backend/app/routers/samples.py
REST API endpoints for Sample upload, microphone recording, audio streaming, and CRUD.
Part of Voice Sampler Milestone 2 (M2).
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
import unicodedata
import uuid

from fastapi import APIRouter, File, HTTPException, Query, Request, Response, UploadFile, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.services.audio_service import get_audio_service
from backend.app.services.notification_service import get_notification_manager
from backend.app.services.whisper_service import get_whisper_service
from backend.app.utils.audio_stream import iter_file_range, stream_audio_file

logger = logging.getLogger("voice_sampler.routers.samples")
router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic Schemas
# ---------------------------------------------------------------------------

class UpdateTranscriptRequest(BaseModel):
    transcript: str = Field(..., description="Updated transcript text")
    status: Optional[str] = Field(None, description="Optional status override")

    @field_validator("transcript")
    @classmethod
    def normalize_transcript_nfc(cls, v: str) -> str:
        if isinstance(v, str):
            return unicodedata.normalize("NFC", v)
        return v


class SampleResponse(BaseModel):
    """Schema representing an individual sample with dual-aliased duration and status."""
    id: str = Field(..., description="Unique sample UUID")
    name: str = Field(default="", description="Display name for the sample")
    filename: str = Field(..., description="Original or recorded audio filename")
    file_path: Optional[str] = Field(None, description="Path to raw audio file")
    preprocessed_path: Optional[str] = Field(None, description="Path to preprocessed 24kHz audio")
    duration: float = Field(..., description="Audio duration in seconds (contract alias)")
    duration_seconds: float = Field(..., description="Audio duration in seconds (DB alias)")
    transcript: str = Field(default="", description="Speech-to-text transcript or manual edit")
    status: str = Field(..., description="Lifecycle status alias: pending, transcribing, transcribed, manual, error")
    transcript_status: str = Field(..., description="Lifecycle status matching SQLite column")
    sample_rate: int = Field(default=44100, description="Audio sample rate in Hz")
    channels: int = Field(default=1, description="Number of channels")
    created_at: str = Field(..., description="ISO 8601 UTC creation timestamp")
    updated_at: Optional[str] = Field(None, description="ISO 8601 UTC update timestamp")


class SampleListResponse(BaseModel):
    samples: List[SampleResponse] = Field(default_factory=list, description="List of recorded/uploaded samples")


class DeleteSampleResponse(BaseModel):
    status: str = Field(default="deleted", description="Confirmation status flag")
    id: str = Field(..., description="ID of the deleted sample")
    message: Optional[str] = Field(None, description="Confirmation message")


class BulkDeleteResponse(BaseModel):
    status: str = Field(default="deleted", description="Confirmation status flag")
    deleted_count: int = Field(..., description="Number of samples deleted")
    message: str = Field(..., description="Confirmation message")


def _format_sample(row: Dict[str, Any]) -> Dict[str, Any]:
    """Format SQLite row into dual-aliased dictionary conforming to SampleResponse."""
    duration = float(row.get("duration_seconds", row.get("duration", 0.0)))
    st = str(row.get("transcript_status", row.get("status", "pending")))
    return {
        "id": str(row["id"]),
        "name": str(row.get("name") or row.get("filename") or ""),
        "filename": str(row.get("filename") or ""),
        "file_path": str(row.get("file_path") or ""),
        "preprocessed_path": row.get("preprocessed_path"),
        "duration": duration,
        "duration_seconds": duration,
        "transcript": str(row.get("transcript") or ""),
        "status": st,
        "transcript_status": st,
        "sample_rate": int(row.get("sample_rate", 44100)),
        "channels": int(row.get("channels", 1)),
        "created_at": str(row.get("created_at") or ""),
        "updated_at": str(row.get("updated_at") or "") if row.get("updated_at") else None,
    }

# ---------------------------------------------------------------------------
# Sample Ingestion Endpoints (Declared before /{sample_id} to prevent path shadowing)
# ---------------------------------------------------------------------------

@router.post("/upload", response_model=SampleResponse)
async def upload_sample(file: UploadFile = File(...)):
    """
    Upload an audio file (WAV, MP3, FLAC, OGG).
    Normalizes audio, stores PCM 16-bit WAV, registers SQLite record,
    and initiates asynchronous Whisper transcription with Polish priority.
    """
    content = await file.read()
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty audio file provided")

    sample_id = str(uuid.uuid4())
    filename = Path(file.filename or "sample.wav").name

    audio_service = get_audio_service()
    try:
        audio_info = audio_service.process_and_save_audio(content, sample_id, filename)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to process uploaded audio: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid or corrupted audio file: {e}")

    sample_row = repository.create_sample(
        sample_id=sample_id,
        name=filename,
        filename=filename,
        file_path=str(audio_info["file_path"]),
        duration_seconds=audio_info["duration"],
        sample_rate=audio_info["sample_rate"],
        channels=audio_info["channels"],
        transcript="",
        transcript_status="pending",
    )

    # 1. Immediately notify WebSocket clients of sample creation (preserves lifecycle order)
    formatted_initial = _format_sample(sample_row)
    get_notification_manager().notify_sample_created(formatted_initial)

    # 2. Enqueue background Polish Whisper transcription
    whisper_svc = get_whisper_service()
    fut = whisper_svc.enqueue_transcription(sample_id=sample_id, audio_path=audio_info["file_path"])

    # 3. Asynchronously await completion without blocking the ASGI event loop thread
    try:
        await asyncio.wait_for(asyncio.wrap_future(fut), timeout=1.0)
        updated = repository.get_sample(sample_id)
        if updated:
            sample_row = updated
    except (asyncio.TimeoutError, Exception):
        pass

    return _format_sample(sample_row)


@router.post("/record", response_model=SampleResponse)
async def record_sample(file: UploadFile = File(...)):
    """
    Ingest recorded audio from browser MediaRecorder (WAV, WebM/Opus, OGG).
    Decodes containers, normalizes to 16-bit PCM WAV, registers sample,
    and triggers Polish Whisper transcription.
    """
    content = await file.read()
    if not content or len(content) == 0:
        raise HTTPException(status_code=400, detail="Empty recording buffer")

    sample_id = str(uuid.uuid4())
    raw_name = file.filename or f"recording_{sample_id[:8]}.wav"
    filename = Path(raw_name).name

    audio_service = get_audio_service()
    try:
        audio_info = audio_service.process_and_save_audio(content, sample_id, filename)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Failed to process recorded audio: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Invalid recording audio data: {e}")

    sample_row = repository.create_sample(
        sample_id=sample_id,
        name=filename,
        filename=filename,
        file_path=str(audio_info["file_path"]),
        duration_seconds=audio_info["duration"],
        sample_rate=audio_info["sample_rate"],
        channels=audio_info["channels"],
        transcript="",
        transcript_status="pending",
    )

    # 1. Immediately notify WebSocket clients of sample creation
    formatted_initial = _format_sample(sample_row)
    get_notification_manager().notify_sample_created(formatted_initial)

    # 2. Enqueue background Polish Whisper transcription
    whisper_svc = get_whisper_service()
    fut = whisper_svc.enqueue_transcription(sample_id=sample_id, audio_path=audio_info["file_path"])

    # 3. Asynchronously await completion without blocking the ASGI event loop thread
    try:
        await asyncio.wait_for(asyncio.wrap_future(fut), timeout=1.0)
        updated = repository.get_sample(sample_id)
        if updated:
            sample_row = updated
    except (asyncio.TimeoutError, Exception):
        pass

    return _format_sample(sample_row)


# ---------------------------------------------------------------------------
# Sample CRUD Endpoints
# ---------------------------------------------------------------------------

@router.get("", response_model=SampleListResponse)
@router.get("/", response_model=SampleListResponse, include_in_schema=False)
def list_samples(status: Optional[str] = Query(None, description="Optional status filter")):
    """List all voice samples ordered by creation date descending."""
    rows = repository.list_samples(status=status)
    return {"samples": [_format_sample(r) for r in rows]}


@router.get("/{sample_id}", response_model=SampleResponse)
def get_sample(sample_id: str):
    """Retrieve details of an individual sample by UUID."""
    sample = repository.get_sample(sample_id)
    if not sample:
        raise HTTPException(status_code=404, detail="Sample not found")
    return _format_sample(sample)


@router.put("/{sample_id}", response_model=SampleResponse)
def update_sample(sample_id: str, req: UpdateTranscriptRequest):
    """
    Update the transcript of a sample.
    Preserves Polish diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż) in canonical NFC form
    and broadcasts update notification.
    """
    existing = repository.get_sample(sample_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Sample not found")

    normalized_transcript = unicodedata.normalize("NFC", req.transcript)
    target_status = req.status if req.status is not None else "manual"
    updated = repository.update_sample(
        sample_id,
        transcript=normalized_transcript,
        transcript_status=target_status,
    )
    if not updated:
        raise HTTPException(status_code=404, detail="Sample not found")

    formatted = _format_sample(updated)
    get_notification_manager().notify_sample_updated(sample_id, normalized_transcript, target_status)
    return formatted


@router.post("/{sample_id}/transcribe", response_model=SampleResponse)
def retranscribe_sample(sample_id: str):
    """
    Trigger or re-run automatic Whisper transcription on an existing sample.
    Resets status to 'pending' and enqueues transcription in the background pool.
    """
    existing = repository.get_sample(sample_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Sample not found")

    file_path = existing.get("file_path")
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=400, detail="Audio file not found on disk")

    updated = repository.update_sample(sample_id, transcript_status="pending")
    whisper_service = get_whisper_service()
    whisper_service.enqueue_transcription(sample_id, audio_path=file_path)

    return _format_sample(updated or existing)


@router.delete("", response_model=BulkDeleteResponse)
def bulk_delete_samples(
    only_missing: bool = Query(False, description="If True, only deletes samples whose files do not exist")
):
    """
    Bulk delete samples. If only_missing is True, only deletes samples without physical audio files.
    Otherwise, removes all samples and unlinks any remaining audio files.
    """
    audio_service = get_audio_service()
    if not only_missing:
        all_samples = repository.get_all_samples()
        for s in all_samples:
            sample_id = s.get("id")
            if sample_id:
                audio_service.delete_audio_file(sample_id)
            if s.get("file_path"):
                try:
                    fp = Path(s["file_path"])
                    if fp.exists():
                        fp.unlink()
                except Exception:
                    pass

    deleted_count = repository.delete_all_samples(only_missing_files=only_missing)
    return {
        "status": "deleted",
        "deleted_count": deleted_count,
        "message": f"Successfully deleted {deleted_count} sample(s)",
    }


@router.delete("/{sample_id}", response_model=DeleteSampleResponse)
def delete_sample(sample_id: str):
    """
    Delete a sample: unlinks raw and preprocessed audio files and removes SQLite record.
    """
    existing = repository.get_sample(sample_id)
    if not existing:
        raise HTTPException(status_code=404, detail="Sample not found")

    audio_service = get_audio_service()
    audio_service.delete_audio_file(sample_id)

    if existing.get("file_path"):
        try:
            fp = Path(existing["file_path"])
            if fp.exists():
                fp.unlink()
        except Exception as e:
            logger.warning(f"Error unlinking audio path {existing.get('file_path')}: {e}")

    deleted = repository.delete_sample(sample_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Sample not found")

    get_notification_manager().notify_sample_deleted(sample_id)
    return {"status": "deleted", "id": sample_id, "message": f"Sample {sample_id} deleted"}


# ---------------------------------------------------------------------------
# Audio Streaming Endpoint (RFC 7233 / HTTP 206 Partial Content)
# ---------------------------------------------------------------------------

@router.get("/{sample_id}/audio")
def get_sample_audio(sample_id: str, request: Request):
    """
    Stream audio file with full byte-range support (RFC 7233).
    Supports HTTP 200 (full audio) and HTTP 206 (partial range),
    with strict HTTP 416 validation for out-of-bounds or reversed ranges,
    and DB existence verification.
    """
    sample = repository.get_sample(sample_id)
    if sample is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Sample {sample_id} not found",
        )

    audio_service = get_audio_service()
    file_path: Optional[Path] = None

    if sample.get("file_path"):
        fp = Path(sample["file_path"])
        if fp.exists() and fp.is_file():
            file_path = fp

    if file_path is None:
        file_path = audio_service.get_audio_path(sample_id)

    if file_path is None or not file_path.exists():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found",
        )

    return stream_audio_file(file_path=file_path, request=request, media_type="audio/wav")

