"""
tests/test_m2_challenger_stress.py
Adversarial stress and verification suite written by Challenger M2-3.
Focus areas:
1. Non-audio and corrupted audio ingestion rejection (HTTP 400).
2. Malformed and reversed Range headers (HTTP 416 with Content-Range).
3. Range upper-bound clamping past EOF (HTTP 206 Partial Content).
"""

import io
import math
import os
from pathlib import Path
import struct
import wave
import pytest
from fastapi.testclient import TestClient

from backend.app.main import app
from backend.app.services.audio_service import get_audio_service


def generate_wav(duration_s: float = 0.5, sr: int = 24000) -> bytes:
    num_samples = int(duration_s * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(num_samples):
            val = int(16000 * math.sin(2 * math.pi * 440.0 * (i / sr)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return buf.getvalue()


@pytest.fixture(scope="module")
def client():
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(scope="module")
def sample_id(client):
    wav = generate_wav(duration_s=1.0)
    resp = client.post(
        "/api/samples/upload",
        files={"file": ("challenger_test.wav", io.BytesIO(wav), "audio/wav")},
    )
    assert resp.status_code == 200, f"Setup upload failed: {resp.text}"
    sid = resp.json()["id"]
    yield sid
    client.delete(f"/api/samples/{sid}")


# ============================================================================
# 1. Non-audio / corrupt audio uploads and recordings -> HTTP 400
# ============================================================================

INVALID_PAYLOADS = [
    ("empty.wav", b"", "audio/wav"),
    ("corrupt_riff.wav", b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00INVALID_DATA_CORRUPT", "audio/wav"),
    ("truncated.wav", b"RIFF\x24\x00\x00\x00WAVEfmt ", "audio/wav"),
    ("random_bytes.bin", os.urandom(512), "application/octet-stream"),
    ("plain_text.txt", b"This is plain text pretending to be audio data.", "text/plain"),
    ("html_file.html", b"<!DOCTYPE html><html><body><h1>Not Audio</h1></body></html>", "text/html"),
    ("json_file.json", b'{"error": "not audio", "count": 123}', "application/json"),
    ("fake_mp3.mp3", b"ID3\x03\x00\x00\x00\x00\x00\x20CORRUPTED_NON_AUDIO_MP3_PAYLOAD", "audio/mpeg"),
    ("fake_flac.flac", b"fLaC\x00\x00\x00\x22CORRUPTED_NON_AUDIO_FLAC_PAYLOAD", "audio/flac"),
]


@pytest.mark.parametrize("filename,content,mime", INVALID_PAYLOADS)
def test_upload_corrupt_or_non_audio_rejected_400(client, filename, content, mime):
    """Confirm corrupt non-audio uploads return HTTP 400 Bad Request."""
    resp = client.post("/api/samples/upload", files={"file": (filename, io.BytesIO(content), mime)})
    assert resp.status_code == 400, f"Expected 400 for upload of {filename}, got {resp.status_code}: {resp.text}"


@pytest.mark.parametrize("filename,content,mime", INVALID_PAYLOADS)
def test_record_corrupt_or_non_audio_rejected_400(client, filename, content, mime):
    """Confirm corrupt non-audio recordings return HTTP 400 Bad Request."""
    resp = client.post("/api/samples/record", files={"file": (filename, io.BytesIO(content), mime)})
    assert resp.status_code == 400, f"Expected 400 for record of {filename}, got {resp.status_code}: {resp.text}"


# ============================================================================
# 2. Malformed / reversed Range headers -> HTTP 416 with Content-Range
# ============================================================================

MALFORMED_RANGES = [
    "bytes=foo-bar",
    "bytes=10-foo",
    "bytes=-foo",
    "bytes=-",
    "bytes=---",
    "bytes=500-100",  # Reversed
    "bytes=-0",        # Suffix of 0 bytes
    "bytes=99999999-100000000",  # Start beyond EOF
]


@pytest.mark.parametrize("range_hdr", MALFORMED_RANGES)
def test_malformed_range_returns_416_with_content_range(client, sample_id, range_hdr):
    """Confirm malformed ranges return HTTP 416 with Content-Range header."""
    resp = client.get(f"/api/samples/{sample_id}/audio", headers={"Range": range_hdr})
    assert resp.status_code == 416, f"Expected 416 for Range '{range_hdr}', got {resp.status_code}: {resp.text}"
    content_range = resp.headers.get("Content-Range") or resp.headers.get("content-range")
    assert content_range is not None, f"Expected Content-Range header on 416 response for '{range_hdr}'"
    assert content_range.startswith("bytes */"), f"Malformed Content-Range header: {content_range}"


# ============================================================================
# 3. Range upper-bound clamping past EOF -> HTTP 206 Partial Content
# ============================================================================

def test_range_clamping_returns_206_for_range_past_eof(client, sample_id):
    """
    Confirm range clamping returns HTTP 206 for ranges extending past EOF.
    RFC 7233 Section 2.1:
    If last-byte-pos is greater than or equal to current length,
    the byte range is interpreted as the remainder of the representation,
    and server replaces last-byte-pos with total_len - 1.
    """
    full_resp = client.get(f"/api/samples/{sample_id}/audio")
    total_len = len(full_resp.content)
    assert total_len > 0

    # Range 0-999999999 extending far past EOF
    resp = client.get(f"/api/samples/{sample_id}/audio", headers={"Range": "bytes=0-999999999"})
    assert resp.status_code == 206, (
        f"Expected HTTP 206 (clamped to EOF) for 'bytes=0-999999999', but got HTTP {resp.status_code}: {resp.text}"
    )
    content_range = resp.headers.get("Content-Range") or resp.headers.get("content-range")
    assert content_range == f"bytes 0-{total_len - 1}/{total_len}", (
        f"Expected 'bytes 0-{total_len - 1}/{total_len}', got '{content_range}'"
    )
    assert len(resp.content) == total_len


def test_range_clamping_with_offset_returns_206_for_range_past_eof(client, sample_id):
    """
    Range 100-999999999 extending past EOF with non-zero start.
    Should clamp end to EOF and return 206 with correct Content-Range and length.
    """
    full_resp = client.get(f"/api/samples/{sample_id}/audio")
    total_len = len(full_resp.content)
    start = 100

    resp = client.get(f"/api/samples/{sample_id}/audio", headers={"Range": f"bytes={start}-999999999"})
    assert resp.status_code == 206, (
        f"Expected HTTP 206 (clamped to EOF) for 'bytes={start}-999999999', but got HTTP {resp.status_code}: {resp.text}"
    )
    content_range = resp.headers.get("Content-Range") or resp.headers.get("content-range")
    assert content_range == f"bytes {start}-{total_len - 1}/{total_len}"
    assert len(resp.content) == total_len - start
