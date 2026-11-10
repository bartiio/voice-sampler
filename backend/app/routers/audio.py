"""
backend/app/routers/audio.py
REST API endpoints for generated audio playback streaming and download.
Supports RFC 7233 byte-range requests for audio scrubbers, Content-Disposition download,
and strict directory traversal defenses.
Part of Voice Sampler Milestone 4 (M4).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Request, status

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.utils.audio_stream import stream_audio_file

logger = logging.getLogger("voice_sampler.routers.audio")
router = APIRouter()
audio_router = router  # Alias for module re-exports


def resolve_generated_audio_file(filename: str) -> Path:
    """
    Validate and resolve generated audio file path safely.
    Guards strictly against directory traversal (.., /, \\).
    """
    if (
        not filename
        or not filename.strip()
        or Path(filename).name != filename
        or ".." in filename
        or "/" in filename
        or "\\" in filename
    ):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid filename format",
        )

    settings = get_settings()

    # 1. Lookup in SQLite repository
    gen_record = repository.get_generation_by_filename(filename)
    if gen_record and gen_record.get("output_audio_path"):
        out_p = Path(gen_record["output_audio_path"])
        if not out_p.is_absolute():
            out_p = (settings.project_root / out_p).resolve()
        if out_p.is_file():
            return out_p

    # 2. Check canonical directory paths
    candidates = [
        (settings.generated_dir / filename).resolve(),
        (settings.generated_tts_dir / filename).resolve(),
        (settings.generated_sts_dir / filename).resolve(),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Audio file not found",
    )


@router.get("/generated/{filename}")
def stream_generated_audio(filename: str, request: Request):
    """
    Stream generated audio for in-app playback with RFC 7233 Range support.
    """
    file_path = resolve_generated_audio_file(filename)
    return stream_audio_file(file_path=file_path, request=request, media_type="audio/wav")


@router.get("/download/{filename}")
def download_generated_audio(filename: str, request: Request):
    """
    Download generated audio file as an attachment.
    """
    file_path = resolve_generated_audio_file(filename)
    return stream_audio_file(
        file_path=file_path,
        request=request,
        media_type="audio/wav",
        download_filename=filename,
    )
