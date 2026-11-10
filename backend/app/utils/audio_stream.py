"""
backend/app/utils/audio_stream.py
Reusable RFC 7233 byte-range audio streaming and download response generator.
Shared by sample audio streaming and generated audio streaming/download.
Part of Voice Sampler Milestone 4 (M4).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Generator, Optional

from fastapi import HTTPException, Request, status
from fastapi.responses import StreamingResponse


def iter_file_range(file_path: Path, start: int, length: int, chunk_size: int = 65536) -> Generator[bytes, None, None]:
    """Yield chunks of a file within a specified byte range."""
    with open(file_path, "rb") as f:
        f.seek(start)
        remaining = length
        while remaining > 0:
            read_size = min(remaining, chunk_size)
            chunk = f.read(read_size)
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


def stream_audio_file(
    file_path: Path,
    request: Request,
    media_type: str = "audio/wav",
    download_filename: Optional[str] = None,
) -> StreamingResponse:
    """
    Stream an audio file with RFC 7233 byte-range request support.
    Supports HTTP 200 (full audio), HTTP 206 (partial range),
    and strict HTTP 416 validation.
    Optionally sets Content-Disposition attachment if download_filename is provided.
    """
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Audio file not found",
        )

    total_len = os.path.getsize(file_path)
    range_header = request.headers.get("Range") or request.headers.get("range")
    if range_header:
        range_header = range_header.strip()

    disposition_header: dict[str, str] = {}
    if download_filename:
        safe_name = Path(download_filename).name.replace('"', '\\"')
        disposition_header["Content-Disposition"] = f'attachment; filename="{safe_name}"'

    if range_header and range_header.startswith("bytes="):
        content_range_416 = {"Content-Range": f"bytes */{total_len}"}
        if total_len == 0:
            raise HTTPException(
                status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
                detail="Requested Range Not Satisfiable",
                headers=content_range_416,
            )

        raw_range = range_header[6:].strip()
        parts = raw_range.split("-")
        if len(parts) != 2:
            raise HTTPException(
                status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
                detail="Invalid range format",
                headers=content_range_416,
            )

        try:
            part0 = parts[0].strip()
            part1 = parts[1].strip()

            if part0 and part1:
                start = int(part0)
                end = int(part1)
            elif part0 and not part1:
                start = int(part0)
                end = total_len - 1
            elif not part0 and part1:
                suffix = int(part1)
                if suffix <= 0:
                    raise HTTPException(
                        status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
                        detail="Requested Range Not Satisfiable",
                        headers=content_range_416,
                    )
                start = max(0, total_len - suffix)
                end = total_len - 1
            else:
                raise HTTPException(
                    status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
                    detail="Invalid range format",
                    headers=content_range_416,
                )
        except (ValueError, TypeError):
            raise HTTPException(
                status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
                detail="Invalid range format",
                headers=content_range_416,
            )

        # RFC 7233 Section 2.1: clamp upper bound extending past EOF
        if start < total_len and end >= total_len:
            end = total_len - 1

        if start >= total_len or start > end or start < 0:
            raise HTTPException(
                status_code=status.HTTP_416_RANGE_NOT_SATISFIABLE,
                detail="Requested Range Not Satisfiable",
                headers=content_range_416,
            )

        chunk_len = end - start + 1
        headers = {
            "Content-Range": f"bytes {start}-{end}/{total_len}",
            "Accept-Ranges": "bytes",
            "Content-Length": str(chunk_len),
            "Content-Type": media_type,
            **disposition_header,
        }
        return StreamingResponse(
            iter_file_range(file_path, start, chunk_len),
            status_code=status.HTTP_206_PARTIAL_CONTENT,
            headers=headers,
            media_type=media_type,
        )

    # Full representation response (HTTP 200)
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(total_len),
        "Content-Type": media_type,
        **disposition_header,
    }
    return StreamingResponse(
        iter_file_range(file_path, 0, total_len),
        status_code=status.HTTP_200_OK,
        headers=headers,
        media_type=media_type,
    )
