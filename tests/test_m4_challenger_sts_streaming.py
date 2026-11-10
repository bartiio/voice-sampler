"""
tests/test_m4_challenger_sts_streaming.py
Adversarial Challenger Test Suite for Milestone 4 (Voice Generation):
- Speech-to-Speech (STS) RVC v2 inference
- Exact boundary parameter testing on pitch shift (-24 to +24, boundary, out-of-bounds, non-integer)
- Exact boundary parameter testing on index rate (0.0 to 1.0, boundary, out-of-bounds, non-numeric)
- Audio conversion properties: 40000 Hz sample rate, PCM16 format, pitch transposition duration preservation
- RFC 7233 range streaming edge cases: single byte range, suffix range, start > end (416), start >= size (416), upper bound clamping
- Path traversal defenses: generated and download endpoints against .., %2e%2e, slashes, backslashes, absolute Windows paths
- Concurrency & multi-range stress testing
"""

from __future__ import annotations

import concurrent.futures
import io
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional
import uuid

from fastapi import HTTPException
from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.db.connection import init_storage_directories, set_db_path
from backend.app.main import app
from backend.app.routers.audio import resolve_generated_audio_file
from backend.app.services.rvc_service import get_rvc_service
from backend.app.services.simulation import get_simulation_engine
from backend.app.utils.audio_stream import stream_audio_file


# ---------------------------------------------------------------------------
# Test Environment Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_challenger_environment(tmp_path: Path):
    """Isolate storage directories and SQLite database per test."""
    test_db = tmp_path / "test_m4_challenger.db"
    set_db_path(test_db)
    init_storage_directories(tmp_path / "data")
    repository.init_db(test_db)

    settings = get_settings()
    settings.data_dir = tmp_path / "data"
    settings.models_dir = tmp_path / "data" / "models"
    settings.cache_dir = tmp_path / "data" / "cache"
    settings.generated_dir = tmp_path / "data" / "generated"
    settings.generated_tts_dir = tmp_path / "data" / "generated" / "tts"
    settings.generated_sts_dir = tmp_path / "data" / "generated" / "sts"
    settings.samples_raw_dir = tmp_path / "data" / "samples" / "raw"
    settings.ensure_directories()

    yield

    set_db_path(None)


# ---------------------------------------------------------------------------
# Seeding Helpers
# ---------------------------------------------------------------------------

def _seed_audio_sample(
    sample_id: str = "sample_sts_source",
    filename: str = "sample_sts_source.wav",
    duration_s: float = 2.0,
    sample_rate: int = 44100,
    channels: int = 1,
) -> Dict[str, Any]:
    """Generate and persist a synthetic PCM16 WAV audio sample."""
    settings = get_settings()
    file_path = settings.samples_raw_dir / filename
    num_samples = int(duration_s * sample_rate)
    t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)

    if channels == 1:
        signal = 0.5 * np.sin(2 * np.pi * 330.0 * t)
        audio = (signal * 32767.0).astype(np.int16)
    else:
        sig1 = 0.5 * np.sin(2 * np.pi * 330.0 * t)
        sig2 = 0.5 * np.cos(2 * np.pi * 440.0 * t)
        audio = np.stack([(sig1 * 32767.0).astype(np.int16), (sig2 * 32767.0).astype(np.int16)], axis=-1)

    sf.write(str(file_path), audio, sample_rate, subtype="PCM_16", format="WAV")

    return repository.create_sample(
        name=filename,
        filename=filename,
        file_path=str(file_path),
        duration_seconds=duration_s,
        sample_rate=sample_rate,
        channels=channels,
        transcript="Próbka audio do testu konwersji głosu RVC.",
        sample_id=sample_id,
        transcript_status="transcribed",
    )


def _seed_rvc_model(
    model_name: str = "rvc_challenger_voice",
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Generate and persist a mock RVC v2 model checkpoint in DB and on disk."""
    settings = get_settings()
    model_dir = settings.models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = settings.models_dir / f"{model_name}.pth"
    ckpt_path.write_bytes(b"MOCK_RVC_MODEL_CHECKPOINT_DATA")

    index_path = settings.models_dir / f"{model_name}.index"
    index_path.write_bytes(b"MOCK_RVC_FEATURE_INDEX_DATA")

    config_path = model_dir / "config.json"
    config_path.write_text('{"architecture": "NSF-VITS-v2", "sample_rate": 40000}', encoding="utf-8")

    return repository.create_model(
        name=model_name,
        engine="rvc",
        checkpoint_path=f"data/models/{model_name}.pth",
        index_path=f"data/models/{model_name}.index",
        config_path=f"data/models/{model_name}/config.json",
        parameters={"sample_ids": [sample_id] if sample_id else []},
        sample_count=1 if sample_id else 0,
    )


# ===========================================================================
# 1. Exact Boundary Testing on Pitch Shift (-24 to +24 semitones)
# ===========================================================================

class TestPitchShiftBoundaries:
    """Adversarial boundary checks for pitch shift in Speech-to-Speech (STS)."""

    @pytest.mark.parametrize("pitch_shift", [-24, 0, 24])
    def test_pitch_shift_valid_boundaries_return_200(self, pitch_shift: int):
        """Exact boundaries -24 and +24, and neutral 0 must return HTTP 200."""
        client = TestClient(app)
        sample = _seed_audio_sample("sample_ps_valid")
        _seed_rvc_model("rvc_ps_model", sample["id"])

        payload = {
            "input_sample_id": sample["id"],
            "model_name": "rvc_ps_model",
            "pitch_shift": pitch_shift,
            "index_rate": 0.75,
        }
        resp = client.post("/api/generate/sts", json=payload)
        assert resp.status_code == 200, f"Expected 200 for pitch_shift={pitch_shift}, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "generation_id" in data
        assert "filename" in data
        assert data["filename"].startswith("sts_")
        assert data["filename"].endswith(".wav")
        assert data["sample_rate"] == 40000

    @pytest.mark.parametrize("pitch_shift", [-25, 25, -100, 100, -9999, 9999])
    def test_pitch_shift_out_of_bounds_returns_400(self, pitch_shift: int):
        """Boundary violation: values < -24 or > +24 must return HTTP 400."""
        client = TestClient(app)
        sample = _seed_audio_sample("sample_ps_oob")
        _seed_rvc_model("rvc_ps_model", sample["id"])

        payload = {
            "input_sample_id": sample["id"],
            "model_name": "rvc_ps_model",
            "pitch_shift": pitch_shift,
            "index_rate": 0.75,
        }
        resp = client.post("/api/generate/sts", json=payload)
        assert resp.status_code == 400, f"Expected 400 for pitch_shift={pitch_shift}, got {resp.status_code}: {resp.text}"
        assert "Pitch shift must be between -24 and +24" in resp.json().get("detail", "")

    @pytest.mark.parametrize("invalid_pitch_shift", ["invalid_string", 2.5, -3.7, [1], {"pitch": 0}, None])
    def test_pitch_shift_non_integer_returns_422_or_400(self, invalid_pitch_shift: Any):
        """Non-integer pitch shift types (string, float with fractional part, list, null) must fail validation."""
        client = TestClient(app)
        sample = _seed_audio_sample("sample_ps_invalid")
        _seed_rvc_model("rvc_ps_model", sample["id"])

        payload = {
            "input_sample_id": sample["id"],
            "model_name": "rvc_ps_model",
            "pitch_shift": invalid_pitch_shift,
            "index_rate": 0.75,
        }
        resp = client.post("/api/generate/sts", json=payload)
        assert resp.status_code in (400, 422), f"Expected 400 or 422 for non-integer pitch_shift={invalid_pitch_shift!r}, got {resp.status_code}"


# ===========================================================================
# 2. Exact Boundary Testing on Index Rate (0.0 to 1.0 float)
# ===========================================================================

class TestIndexRateBoundaries:
    """Adversarial boundary checks for index rate (retrieval feature blend)."""

    @pytest.mark.parametrize("index_rate", [0.0, 0.5, 1.0])
    def test_index_rate_valid_boundaries_return_200(self, index_rate: float):
        """Exact boundaries 0.0 and 1.0, and midpoint 0.5 must return HTTP 200."""
        client = TestClient(app)
        sample = _seed_audio_sample("sample_ir_valid")
        _seed_rvc_model("rvc_ir_model", sample["id"])

        payload = {
            "input_sample_id": sample["id"],
            "model_name": "rvc_ir_model",
            "pitch_shift": 0,
            "index_rate": index_rate,
        }
        resp = client.post("/api/generate/sts", json=payload)
        assert resp.status_code == 200, f"Expected 200 for index_rate={index_rate}, got {resp.status_code}: {resp.text}"
        data = resp.json()
        assert "generation_id" in data
        assert data["sample_rate"] == 40000

    @pytest.mark.parametrize("index_rate", [-0.01, 1.01, -1.0, 2.0, -100.0, 100.0])
    def test_index_rate_out_of_bounds_returns_400(self, index_rate: float):
        """Boundary violation: index_rate < 0.0 or > 1.0 must return HTTP 400."""
        client = TestClient(app)
        sample = _seed_audio_sample("sample_ir_oob")
        _seed_rvc_model("rvc_ir_model", sample["id"])

        payload = {
            "input_sample_id": sample["id"],
            "model_name": "rvc_ir_model",
            "pitch_shift": 0,
            "index_rate": index_rate,
        }
        resp = client.post("/api/generate/sts", json=payload)
        assert resp.status_code == 400, f"Expected 400 for index_rate={index_rate}, got {resp.status_code}: {resp.text}"
        assert "Index rate must be between 0.0 and 1.0" in resp.json().get("detail", "")

    @pytest.mark.parametrize("invalid_index_rate", ["fast", "0.75x", [0.5], None])
    def test_index_rate_non_numeric_returns_422_or_400(self, invalid_index_rate: Any):
        """Non-numeric index_rate values must fail validation with 422 or 400."""
        client = TestClient(app)
        sample = _seed_audio_sample("sample_ir_invalid")
        _seed_rvc_model("rvc_ir_model", sample["id"])

        payload = {
            "input_sample_id": sample["id"],
            "model_name": "rvc_ir_model",
            "pitch_shift": 0,
            "index_rate": invalid_index_rate,
        }
        resp = client.post("/api/generate/sts", json=payload)
        assert resp.status_code in (400, 422), f"Expected 400 or 422 for non-numeric index_rate={invalid_index_rate!r}, got {resp.status_code}"


# ===========================================================================
# 3. Audio Conversion Properties: 40kHz, PCM16, Duration Preservation
# ===========================================================================

class TestAudioConversionProperties:
    """Empirical verification of synthesized STS audio format, sampling rate, and duration invariance."""

    def test_sts_output_audio_format_pcm16_and_40khz(self):
        """Verify generated STS audio is valid 40000 Hz, mono, 16-bit PCM WAV."""
        client = TestClient(app)
        sample = _seed_audio_sample("sample_fmt_test", duration_s=2.5, sample_rate=24000)
        _seed_rvc_model("rvc_fmt_model", sample["id"])

        payload = {
            "input_sample_id": sample["id"],
            "model_name": "rvc_fmt_model",
            "pitch_shift": 5,
            "index_rate": 0.8,
        }
        resp = client.post("/api/generate/sts", json=payload)
        assert resp.status_code == 200
        filename = resp.json()["filename"]

        settings = get_settings()
        out_file = settings.generated_dir / filename
        assert out_file.exists()

        info = sf.info(str(out_file))
        assert info.samplerate == 40000, f"Expected sample rate 40000 Hz, got {info.samplerate}"
        assert info.channels == 1, f"Expected 1 channel (mono), got {info.channels}"
        assert info.format == "WAV", f"Expected WAV format, got {info.format}"
        assert info.subtype == "PCM_16", f"Expected PCM_16 subtype, got {info.subtype}"

    @pytest.mark.parametrize("pitch_shift", [-24, -12, -7, 0, 5, 12, 24])
    @pytest.mark.parametrize("input_dur, input_sr", [(1.5, 44100), (3.0, 24000), (0.8, 16000)])
    def test_sts_pitch_transposition_preserves_duration_accurately(
        self, pitch_shift: int, input_dur: float, input_sr: int
    ):
        """
        Verify phase-vocoder pitch shift strictly preserves audio duration.
        Output duration at 40kHz must match input duration within a 0.02s threshold.
        """
        engine = get_simulation_engine()
        settings = get_settings()
        input_path = settings.samples_raw_dir / f"test_dur_{input_sr}_{int(input_dur*10)}.wav"

        t = np.linspace(0, input_dur, int(input_dur * input_sr), endpoint=False, dtype=np.float32)
        audio = (0.5 * np.sin(2 * np.pi * 300.0 * t) * 32767.0).astype(np.int16)
        sf.write(str(input_path), audio, input_sr, subtype="PCM_16", format="WAV")

        out_pcm, out_sr = engine.generate_sts_audio(
            input_audio_path=input_path,
            pitch_shift=pitch_shift,
            index_rate=0.75,
            sample_rate=40000,
        )

        assert out_sr == 40000
        out_dur = len(out_pcm) / float(out_sr)
        duration_diff = abs(out_dur - input_dur)
        assert duration_diff < 0.02, (
            f"Duration mismatch for pitch_shift={pitch_shift}, input_dur={input_dur}s: "
            f"got {out_dur:.4f}s (diff={duration_diff:.4f}s)"
        )

    def test_sts_converts_stereo_input_to_mono_40khz(self):
        """Verify stereo input audio is downmixed to mono 40kHz PCM16."""
        client = TestClient(app)
        sample = _seed_audio_sample("sample_stereo", duration_s=1.8, sample_rate=48000, channels=2)
        _seed_rvc_model("rvc_stereo_model", sample["id"])

        payload = {
            "input_sample_id": sample["id"],
            "model_name": "rvc_stereo_model",
            "pitch_shift": -3,
            "index_rate": 0.5,
        }
        resp = client.post("/api/generate/sts", json=payload)
        assert resp.status_code == 200
        filename = resp.json()["filename"]

        settings = get_settings()
        out_file = settings.generated_dir / filename
        info = sf.info(str(out_file))
        assert info.channels == 1
        assert info.samplerate == 40000
        assert info.subtype == "PCM_16"


# ===========================================================================
# 4. RFC 7233 Range Streaming Edge Cases
# ===========================================================================

class TestRFC7233RangeStreamingEdgeCases:
    """
    Adversarial verification of RFC 7233 Range header semantics:
    - single byte range (bytes=0-0 -> 206)
    - suffix range (bytes=-500 -> 206)
    - range start > end (416 Range Not Satisfiable)
    - range start >= file size (416)
    - upper bound clamping (bytes=0-99999999 -> 206 clamped to EOF)
    """

    @pytest.fixture
    def test_audio_file(self) -> Path:
        """Create a known-size PCM16 WAV file in generated_dir and return its Path."""
        settings = get_settings()
        file_path = settings.generated_dir / "sts_range_test.wav"
        sr = 40000
        duration_s = 2.0
        t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
        audio = (0.4 * np.sin(2 * np.pi * 440.0 * t) * 32767.0).astype(np.int16)
        sf.write(str(file_path), audio, sr, subtype="PCM_16", format="WAV")
        return file_path

    def test_range_single_byte_start(self, test_audio_file: Path):
        """Range: bytes=0-0 must return HTTP 206, Content-Length 1, and the exact first byte."""
        client = TestClient(app)
        total_size = test_audio_file.stat().st_size
        full_content = test_audio_file.read_bytes()

        resp = client.get(f"/api/audio/generated/{test_audio_file.name}", headers={"Range": "bytes=0-0"})
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == f"bytes 0-0/{total_size}"
        assert resp.headers.get("Content-Length") == "1"
        assert len(resp.content) == 1
        assert resp.content == full_content[0:1]

    def test_range_single_byte_end(self, test_audio_file: Path):
        """Range: bytes={total-1}-{total-1} must return HTTP 206, Content-Length 1, and the exact final byte."""
        client = TestClient(app)
        total_size = test_audio_file.stat().st_size
        full_content = test_audio_file.read_bytes()
        last_idx = total_size - 1

        resp = client.get(
            f"/api/audio/generated/{test_audio_file.name}",
            headers={"Range": f"bytes={last_idx}-{last_idx}"},
        )
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == f"bytes {last_idx}-{last_idx}/{total_size}"
        assert resp.headers.get("Content-Length") == "1"
        assert len(resp.content) == 1
        assert resp.content == full_content[-1:]

    def test_range_suffix_500_bytes(self, test_audio_file: Path):
        """Range: bytes=-500 must return HTTP 206 with the exact last 500 bytes of the file."""
        client = TestClient(app)
        total_size = test_audio_file.stat().st_size
        full_content = test_audio_file.read_bytes()
        expected_start = total_size - 500

        resp = client.get(f"/api/audio/generated/{test_audio_file.name}", headers={"Range": "bytes=-500"})
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == f"bytes {expected_start}-{total_size - 1}/{total_size}"
        assert resp.headers.get("Content-Length") == "500"
        assert len(resp.content) == 500
        assert resp.content == full_content[-500:]

    def test_range_start_greater_than_end_returns_416(self, test_audio_file: Path):
        """Range: bytes=500-100 (start > end) must return HTTP 416 with Content-Range bytes */total."""
        client = TestClient(app)
        total_size = test_audio_file.stat().st_size

        resp = client.get(f"/api/audio/generated/{test_audio_file.name}", headers={"Range": "bytes=500-100"})
        assert resp.status_code == 416
        assert resp.headers.get("Content-Range") == f"bytes */{total_size}"

    def test_range_start_greater_than_or_equal_to_size_returns_416(self, test_audio_file: Path):
        """Range starting at or after EOF must return HTTP 416."""
        client = TestClient(app)
        total_size = test_audio_file.stat().st_size

        # start exactly at EOF (0-indexed file of size L has valid bytes 0..L-1)
        resp1 = client.get(f"/api/audio/generated/{test_audio_file.name}", headers={"Range": f"bytes={total_size}-"})
        assert resp1.status_code == 416
        assert resp1.headers.get("Content-Range") == f"bytes */{total_size}"

        # start strictly past EOF
        resp2 = client.get(
            f"/api/audio/generated/{test_audio_file.name}",
            headers={"Range": f"bytes={total_size + 1000}-{total_size + 2000}"},
        )
        assert resp2.status_code == 416
        assert resp2.headers.get("Content-Range") == f"bytes */{total_size}"

    def test_range_upper_bound_clamping_to_eof(self, test_audio_file: Path):
        """
        RFC 7233 §2.1: upper bound extending past EOF (bytes=0-99999999)
        must be clamped to EOF (bytes 0-{total_size - 1}/{total_size}) and return HTTP 206.
        """
        client = TestClient(app)
        total_size = test_audio_file.stat().st_size
        full_content = test_audio_file.read_bytes()

        resp = client.get(
            f"/api/audio/generated/{test_audio_file.name}",
            headers={"Range": "bytes=0-99999999"},
        )
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == f"bytes 0-{total_size - 1}/{total_size}"
        assert resp.headers.get("Content-Length") == str(total_size)
        assert len(resp.content) == total_size
        assert resp.content == full_content

    def test_range_clamping_with_nonzero_offset(self, test_audio_file: Path):
        """Range: bytes=100-99999999 must clamp end to EOF and return (total_size - 100) bytes."""
        client = TestClient(app)
        total_size = test_audio_file.stat().st_size
        full_content = test_audio_file.read_bytes()

        resp = client.get(
            f"/api/audio/generated/{test_audio_file.name}",
            headers={"Range": "bytes=100-99999999"},
        )
        assert resp.status_code == 206
        expected_len = total_size - 100
        assert resp.headers.get("Content-Range") == f"bytes 100-{total_size - 1}/{total_size}"
        assert resp.headers.get("Content-Length") == str(expected_len)
        assert len(resp.content) == expected_len
        assert resp.content == full_content[100:]

    @pytest.mark.parametrize("invalid_range", ["bytes=abc-def", "bytes=1-2-3", "bytes=-0", "bytes=---"])
    def test_malformed_range_returns_416(self, test_audio_file: Path, invalid_range: str):
        """Malformed range headers must yield HTTP 416 Range Not Satisfiable."""
        client = TestClient(app)
        total_size = test_audio_file.stat().st_size

        resp = client.get(
            f"/api/audio/generated/{test_audio_file.name}",
            headers={"Range": invalid_range},
        )
        assert resp.status_code == 416
        assert resp.headers.get("Content-Range") == f"bytes */{total_size}"

    def test_unrecognized_unit_or_empty_falls_back_to_200(self, test_audio_file: Path):
        """Non-bytes unit (e.g. seconds=0-10) or empty header must return HTTP 200 full content."""
        client = TestClient(app)
        total_size = test_audio_file.stat().st_size

        resp = client.get(
            f"/api/audio/generated/{test_audio_file.name}",
            headers={"Range": "seconds=0-10"},
        )
        assert resp.status_code == 200
        assert resp.headers.get("Content-Length") == str(total_size)
        assert len(resp.content) == total_size

    def test_download_endpoint_supports_rfc7233_ranges(self, test_audio_file: Path):
        """Verify attachment download endpoint (/api/audio/download/{filename}) also obeys RFC 7233 ranges."""
        client = TestClient(app)
        total_size = test_audio_file.stat().st_size

        resp = client.get(
            f"/api/audio/download/{test_audio_file.name}",
            headers={"Range": "bytes=0-99"},
        )
        assert resp.status_code == 206
        assert resp.headers.get("Content-Range") == f"bytes 0-99/{total_size}"
        assert resp.headers.get("Content-Length") == "100"
        assert len(resp.content) == 100
        assert "Content-Disposition" in resp.headers
        assert f'filename="{test_audio_file.name}"' in resp.headers["Content-Disposition"]


# ===========================================================================
# 5. Path Traversal Defenses: Generated & Download Endpoints
# ===========================================================================

class TestPathTraversalDefenses:
    """
    Adversarial security audit: verify endpoints guard against directory traversal:
    - Relative parent operators: .., %2e%2e
    - Slashes and backslashes: /, \\
    - Absolute Windows paths: C:\\Windows\\..., C:/Windows/...
    - Root paths: /etc/passwd
    """

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "..",
            "%2e%2e",
            "%2e%2e%2f",
            "..%2Fsecret.wav",
            "..\\..\\windows\\win.ini",
            "sub\\file.wav",
            "C:\\Windows\\System32\\cmd.exe",
            "C:/Windows/win.ini",
            "//etc/passwd",
        ],
    )
    def test_generated_endpoint_blocks_traversal_attempts(self, malicious_path: str):
        """GET /api/audio/generated/{filename} must reject traversal with HTTP 400 or 404, never 200/500."""
        client = TestClient(app)
        resp = client.get(f"/api/audio/generated/{malicious_path}")
        assert resp.status_code in (400, 404), (
            f"Directory traversal attack '{malicious_path}' not blocked properly: "
            f"returned HTTP {resp.status_code}"
        )

    @pytest.mark.parametrize(
        "malicious_path",
        [
            "..",
            "%2e%2e",
            "%2e%2e%2f",
            "..%2Fsecret.wav",
            "..\\..\\windows\\win.ini",
            "sub\\file.wav",
            "C:\\Windows\\System32\\cmd.exe",
            "C:/Windows/win.ini",
            "//etc/passwd",
        ],
    )
    def test_download_endpoint_blocks_traversal_attempts(self, malicious_path: str):
        """GET /api/audio/download/{filename} must reject traversal with HTTP 400 or 404, never 200/500."""
        client = TestClient(app)
        resp = client.get(f"/api/audio/download/{malicious_path}")
        assert resp.status_code in (400, 404), (
            f"Directory traversal download attack '{malicious_path}' not blocked properly: "
            f"returned HTTP {resp.status_code}"
        )

    @pytest.mark.parametrize(
        "raw_traversal_input",
        [
            "",
            "   ",
            "..",
            "../secret.txt",
            "..\\boot.ini",
            "/etc/shadow",
            "\\Windows\\win.ini",
            "C:\\boot.ini",
            "C:/Windows/notepad.exe",
            "foo/bar.wav",
            "foo\\bar.wav",
        ],
    )
    def test_resolve_generated_audio_file_direct_rejections(self, raw_traversal_input: str):
        """Direct unit invocation of resolve_generated_audio_file must raise HTTP 400 for unsafe inputs."""
        with pytest.raises(HTTPException) as exc_info:
            resolve_generated_audio_file(raw_traversal_input)
        assert exc_info.value.status_code == 400
        assert exc_info.value.detail == "Invalid filename format"

    def test_nonexistent_safe_filename_returns_404(self):
        """Syntactically valid filename that does not exist on disk must return HTTP 404."""
        client = TestClient(app)
        resp = client.get("/api/audio/generated/sts_nonexistent_file_99999.wav")
        assert resp.status_code == 404
        assert "Audio file not found" in resp.json().get("detail", "")


# ===========================================================================
# 6. Concurrency & Multi-Range Streaming Stress Test
# ===========================================================================

class TestStreamingConcurrencyAndStress:
    """Stress-test concurrent byte-range streaming on the same file."""

    def test_concurrent_partial_content_reads(self):
        """Multiple concurrent client threads fetching disjoint slices of the same file simultaneously."""
        settings = get_settings()
        file_path = settings.generated_dir / "concurrent_stress.wav"
        sr = 40000
        duration_s = 3.0
        t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
        audio = (0.5 * np.sin(2 * np.pi * 500.0 * t) * 32767.0).astype(np.int16)
        sf.write(str(file_path), audio, sr, subtype="PCM_16", format="WAV")
        total_size = file_path.stat().st_size
        full_content = file_path.read_bytes()

        client = TestClient(app)

        test_ranges = [
            ("bytes=0-99", 0, 99),
            ("bytes=100-499", 100, 499),
            ("bytes=500-999", 500, 999),
            ("bytes=1000-4999", 1000, 4999),
            ("bytes=-1000", total_size - 1000, total_size - 1),
            ("bytes=0-0", 0, 0),
            (f"bytes={total_size - 1}-{total_size - 1}", total_size - 1, total_size - 1),
            ("bytes=0-99999999", 0, total_size - 1),
        ]

        def _fetch_range(spec):
            range_hdr, start, end = spec
            r = client.get(f"/api/audio/generated/{file_path.name}", headers={"Range": range_hdr})
            return spec, r

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(_fetch_range, s) for s in test_ranges * 3]
            for f in concurrent.futures.as_completed(futures):
                spec, r = f.result()
                range_hdr, start, end = spec
                assert r.status_code == 206
                expected_bytes = full_content[start : end + 1]
                assert r.content == expected_bytes
                assert r.headers.get("Content-Range") == f"bytes {start}-{end}/{total_size}"
                assert r.headers.get("Content-Length") == str(len(expected_bytes))
