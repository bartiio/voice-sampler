"""
tests/test_m4_reviewer_adversarial.py
Adversarial Stress & Robustness Test Suite for Voice Sampler Milestone 4 (M4):
- RVC STS strict parameter boundary validation (pitch_shift [-24, 24], index_rate [0.0, 1.0])
- Resource resolution robustness (404 for missing input audio sample or model)
- Path traversal & injection defenses on /api/audio/generated/{filename} and /api/audio/download/{filename}
- SQLite database persistence, generation tracking, and index querying
- DSP synthesis integrity (playable PCM16 WAV, duration preservation, no clipping)
- Concurrent generation storm (barrier race) ensuring collision-free outputs
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import threading
from typing import Any, Dict, List
import uuid

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.db.connection import init_storage_directories, set_db_path
from backend.app.main import app
from backend.app.services.simulation import get_simulation_engine


@pytest.fixture(autouse=True)
def setup_adversarial_env(tmp_path: Path):
    """Setup clean isolated environment for each adversarial test."""
    test_db = tmp_path / "test_m4_adv.db"
    set_db_path(test_db)
    init_storage_directories(tmp_path / "data")
    repository.init_db(test_db)

    settings = get_settings()
    settings.data_dir = tmp_path / "data"
    settings.samples_dir = tmp_path / "data" / "samples"
    settings.samples_raw_dir = tmp_path / "data" / "samples" / "raw"
    settings.samples_preprocessed_dir = tmp_path / "data" / "samples" / "preprocessed"
    settings.models_dir = tmp_path / "data" / "models"
    settings.generated_dir = tmp_path / "data" / "generated"
    settings.generated_tts_dir = tmp_path / "data" / "generated" / "tts"
    settings.generated_sts_dir = tmp_path / "data" / "generated" / "sts"
    settings.ensure_directories()

    yield

    # Clean up DB override
    set_db_path(None)


def _seed_test_sample(name: str = "test_voice") -> str:
    """Create and seed a valid PCM16 mono WAV sample in the database."""
    settings = get_settings()
    sr = 44100
    duration = 2.0
    t = np.linspace(0, duration, int(duration * sr), endpoint=False, dtype=np.float32)
    audio = 0.5 * np.sin(2 * np.pi * 220.0 * t)
    wav_path = settings.samples_raw_dir / f"{name}.wav"
    sf.write(str(wav_path), audio, sr, subtype="PCM_16")

    sample = repository.create_sample(
        name=name,
        filename=f"{name}.wav",
        file_path=str(wav_path),
        duration_seconds=duration,
        sample_rate=sr,
        channels=1,
        transcript="Testowy głos do konwersji RVC.",
        transcript_status="completed",
    )
    return sample["id"]


def _seed_test_rvc_model(model_name: str = "rvc_voice_1") -> str:
    """Create a simulated RVC v2 model checkpoint and register in SQLite."""
    sim = get_simulation_engine()
    artifacts = sim.create_rvc_artifacts(
        model_name=model_name,
        epochs=10,
        batch_size=4,
        sample_rate=40000,
        sample_count=2,
        final_loss=0.32,
    )
    repository.create_model(
        name=model_name,
        engine="rvc",
        checkpoint_path=artifacts["checkpoint_path"],
        index_path=artifacts["index_path"],
        config_path=artifacts["config_path"],
        parameters={"sample_rate": 40000, "epochs": 10, "version": "v2"},
        sample_count=2,
    )
    return model_name


# ==============================================================================
# 1. RVC STS Parameter Boundary Adversarial Tests
# ==============================================================================

class TestRVCSTSParameterBoundaries:
    """Stress-test RVC STS parameter validation and boundaries."""

    def test_pitch_shift_valid_boundaries(self):
        """Pitch shift of exactly -24 and +24 must succeed (HTTP 200)."""
        client = TestClient(app)
        sid = _seed_test_sample("pitch_boundary_test")
        mid = _seed_test_rvc_model("rvc_pitch_model")

        # Lower bound: -24
        resp_low = client.post("/api/generate/sts", json={
            "input_sample_id": sid,
            "model_name": mid,
            "pitch_shift": -24,
            "index_rate": 0.5,
        })
        assert resp_low.status_code == 200, f"Expected 200 for pitch_shift=-24, got {resp_low.text}"
        data_low = resp_low.json()
        assert data_low["generation_id"]
        assert data_low["audio_url"].startswith("/api/audio/generated/")

        # Upper bound: +24
        resp_high = client.post("/api/generate/sts", json={
            "input_sample_id": sid,
            "model_name": mid,
            "pitch_shift": 24,
            "index_rate": 0.5,
        })
        assert resp_high.status_code == 200, f"Expected 200 for pitch_shift=+24, got {resp_high.text}"

    @pytest.mark.parametrize("invalid_pitch", [-25, 25, -100, 100, -9999, 9999])
    def test_pitch_shift_out_of_bounds_returns_400(self, invalid_pitch: int):
        """Pitch shift outside [-24, 24] MUST return HTTP 400."""
        client = TestClient(app)
        sid = _seed_test_sample("pitch_oob_test")
        mid = _seed_test_rvc_model("rvc_pitch_oob")

        resp = client.post("/api/generate/sts", json={
            "input_sample_id": sid,
            "model_name": mid,
            "pitch_shift": invalid_pitch,
            "index_rate": 0.75,
        })
        assert resp.status_code == 400
        assert "between -24 and +24" in resp.json()["detail"]

    def test_index_rate_valid_boundaries(self):
        """Index rate of exactly 0.0 and 1.0 must succeed (HTTP 200)."""
        client = TestClient(app)
        sid = _seed_test_sample("idx_boundary_test")
        mid = _seed_test_rvc_model("rvc_idx_model")

        # Lower bound: 0.0
        resp_zero = client.post("/api/generate/sts", json={
            "input_sample_id": sid,
            "model_name": mid,
            "pitch_shift": 0,
            "index_rate": 0.0,
        })
        assert resp_zero.status_code == 200

        # Upper bound: 1.0
        resp_one = client.post("/api/generate/sts", json={
            "input_sample_id": sid,
            "model_name": mid,
            "pitch_shift": 0,
            "index_rate": 1.0,
        })
        assert resp_one.status_code == 200

    @pytest.mark.parametrize("invalid_rate", [-0.01, 1.01, -1.0, 2.0, -100.0, 100.0])
    def test_index_rate_out_of_bounds_returns_400(self, invalid_rate: float):
        """Index rate outside [0.0, 1.0] MUST return HTTP 400."""
        client = TestClient(app)
        sid = _seed_test_sample("idx_oob_test")
        mid = _seed_test_rvc_model("rvc_idx_oob")

        resp = client.post("/api/generate/sts", json={
            "input_sample_id": sid,
            "model_name": mid,
            "pitch_shift": 0,
            "index_rate": invalid_rate,
        })
        assert resp.status_code == 400
        assert "between 0.0 and 1.0" in resp.json()["detail"]

    @pytest.mark.parametrize("empty_val", ["", "   ", "\t\n"])
    def test_empty_input_sample_id_returns_400(self, empty_val: str):
        """Empty or whitespace input_sample_id must return HTTP 400."""
        client = TestClient(app)
        mid = _seed_test_rvc_model("rvc_empty_test")

        resp = client.post("/api/generate/sts", json={
            "input_sample_id": empty_val,
            "model_name": mid,
            "pitch_shift": 0,
            "index_rate": 0.75,
        })
        assert resp.status_code == 400

    @pytest.mark.parametrize("empty_val", ["", "   ", "\t\n"])
    def test_empty_model_name_returns_400(self, empty_val: str):
        """Empty or whitespace model_name must return HTTP 400."""
        client = TestClient(app)
        sid = _seed_test_sample("sample_empty_model")

        resp = client.post("/api/generate/sts", json={
            "input_sample_id": sid,
            "model_name": empty_val,
            "pitch_shift": 0,
            "index_rate": 0.75,
        })
        assert resp.status_code == 400


# ==============================================================================
# 2. Resource Resolution & Missing Resources (HTTP 404)
# ==============================================================================

class TestMissingResourceHandling:
    """Ensure missing samples and models return HTTP 404."""

    def test_nonexistent_input_sample_returns_404(self):
        """Request with unknown sample ID must return HTTP 404."""
        client = TestClient(app)
        mid = _seed_test_rvc_model("rvc_missing_sample_test")

        resp = client.post("/api/generate/sts", json={
            "input_sample_id": "nonexistent-sample-uuid-1234",
            "model_name": mid,
            "pitch_shift": 0,
            "index_rate": 0.75,
        })
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    def test_nonexistent_model_returns_404(self):
        """Request with unknown model name must return HTTP 404."""
        client = TestClient(app)
        sid = _seed_test_sample("sample_for_missing_model")

        resp = client.post("/api/generate/sts", json={
            "input_sample_id": sid,
            "model_name": "unknown_rvc_model_999",
            "pitch_shift": 0,
            "index_rate": 0.75,
        })
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ==============================================================================
# 3. Path Traversal & Audio Endpoint Security
# ==============================================================================

class TestPathTraversalDefenses:
    """Adversarially probe /api/audio/ endpoints for directory traversal."""

    @pytest.mark.parametrize(
        "payload",
        [
            "..",
            "../",
            "..\\",
            "../../etc/passwd",
            "..\\..\\windows\\win.ini",
            "....//....//etc/shadow",
            "subfolder/test.wav",
            "C:\\boot.ini",
            "/etc/hosts",
            "%2e%2e%2fpasswd",
        ],
    )
    def test_stream_generated_blocks_path_traversal(self, payload: str):
        """Directory traversal payloads on /api/audio/generated/{filename} must return 400 or 404, never 200 or 500."""
        client = TestClient(app)
        resp = client.get(f"/api/audio/generated/{payload}")
        assert resp.status_code in (400, 404), f"Payload '{payload}' returned unexpected {resp.status_code}"

    @pytest.mark.parametrize(
        "payload",
        [
            "..",
            "../",
            "..\\",
            "../../etc/passwd",
            "..\\..\\windows\\win.ini",
            "subfolder/test.wav",
            "/etc/passwd",
        ],
    )
    def test_download_blocks_path_traversal(self, payload: str):
        """Directory traversal payloads on /api/audio/download/{filename} must return 400 or 404, never 200 or 500."""
        client = TestClient(app)
        resp = client.get(f"/api/audio/download/{payload}")
        assert resp.status_code in (400, 404), f"Payload '{payload}' returned unexpected {resp.status_code}"

    def test_stream_nonexistent_audio_returns_404(self):
        """Request for nonexistent safe filename must return HTTP 404, not 500."""
        client = TestClient(app)
        resp = client.get("/api/audio/generated/sts_00000000.wav")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()


# ==============================================================================
# 4. DSP & Synthesis Quality Verification
# ==============================================================================

class TestDSPSynthesisIntegrity:
    """Verify digital signal processing quality and playable audio attributes."""

    def test_sts_audio_format_and_duration_preservation(self):
        """Verify STS audio output is valid 40kHz PCM16 WAV and preserves original duration."""
        client = TestClient(app)
        sid = _seed_test_sample("dsp_source_sample")
        mid = _seed_test_rvc_model("dsp_rvc_model")

        resp = client.post("/api/generate/sts", json={
            "input_sample_id": sid,
            "model_name": mid,
            "pitch_shift": 12,  # 1 octave up
            "index_rate": 0.8,
        })
        assert resp.status_code == 200
        data = resp.json()
        filename = data["filename"]

        settings = get_settings()
        out_file = settings.generated_dir / filename
        assert out_file.is_file(), "Generated audio file must exist on disk"

        audio_data, sr = sf.read(str(out_file), dtype="int16")
        assert sr == 40000, f"RVC output sample rate must be 40000, got {sr}"
        assert audio_data.ndim == 1, "Output must be mono"
        assert len(audio_data) > 0

        # Duration preservation (input duration was 2.0s)
        duration_s = len(audio_data) / float(sr)
        assert abs(duration_s - 2.0) < 0.25, f"Expected ~2.0s duration, got {duration_s}s"

        # Check amplitude is not silent and not severely clipped
        float_audio = audio_data.astype(np.float32) / 32767.0
        peak = float(np.max(np.abs(float_audio)))
        rms = float(np.sqrt(np.mean(float_audio ** 2)))
        assert peak > 0.1, f"Audio signal should not be silent, peak={peak}"
        assert peak <= 1.0, f"Audio should not clip beyond 1.0, peak={peak}"
        assert rms > 0.05, f"Audio RMS should be significant, rms={rms}"


# ==============================================================================
# 5. Database Persistence & Filename Index Querying
# ==============================================================================

class TestDatabasePersistenceAndIndex:
    """Verify generation recording, indexing, history retrieval, and cleanup."""

    def test_generation_persists_and_indexes_correctly(self):
        """Verify generation metadata is recorded, indexable by filename, and queryable."""
        client = TestClient(app)
        sid = _seed_test_sample("db_test_sample")
        mid = _seed_test_rvc_model("db_rvc_model")

        resp = client.post("/api/generate/sts", json={
            "input_sample_id": sid,
            "model_name": mid,
            "pitch_shift": -5,
            "index_rate": 0.65,
        })
        assert resp.status_code == 200
        gen_id = resp.json()["generation_id"]
        filename = resp.json()["filename"]

        # 1. Query by ID
        rec = repository.get_generation(gen_id)
        assert rec is not None
        assert rec["type"] == "sts"
        assert rec["output_filename"] == filename
        assert rec["parameters"]["pitch_shift"] == -5
        assert rec["parameters"]["index_rate"] == 0.65

        # 2. Query by filename (using new repository method and idx_generations_filename)
        by_fn = repository.get_generation_by_filename(filename)
        assert by_fn is not None
        assert by_fn["id"] == gen_id

        # 3. Query via REST history
        hist_resp = client.get("/api/generate/history?type=sts")
        assert hist_resp.status_code == 200
        items = hist_resp.json()["generations"]
        assert any(item["id"] == gen_id for item in items)

        # 4. Delete generation
        del_resp = client.delete(f"/api/generate/{gen_id}")
        assert del_resp.status_code == 200
        assert repository.get_generation(gen_id) is None
        assert repository.get_generation_by_filename(filename) is None

        settings = get_settings()
        assert not (settings.generated_dir / filename).is_file()


# ==============================================================================
# 6. Concurrency Barrier Race on Generation
# ==============================================================================

class TestGenerationConcurrency:
    """Stress-test concurrent generation requests with simultaneous barrier release."""

    def test_concurrent_sts_generations_have_unique_files(self):
        """10 threads requesting STS generation concurrently must all succeed with unique files."""
        client = TestClient(app)
        sid = _seed_test_sample("concurrent_sample")
        mid = _seed_test_rvc_model("concurrent_model")

        num_threads = 10
        barrier = threading.Barrier(num_threads)
        results: List[Dict[str, Any]] = []
        lock = threading.Lock()

        def _worker(thread_idx: int):
            t_client = TestClient(app)
            barrier.wait()
            resp = t_client.post("/api/generate/sts", json={
                "input_sample_id": sid,
                "model_name": mid,
                "pitch_shift": (thread_idx % 5) - 2,
                "index_rate": 0.5 + (thread_idx * 0.04),
            })
            with lock:
                results.append({
                    "thread_idx": thread_idx,
                    "status_code": resp.status_code,
                    "data": resp.json() if resp.status_code == 200 else resp.text,
                })

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert len(results) == num_threads
        for r in results:
            assert r["status_code"] == 200, f"Thread {r['thread_idx']} failed: {r['data']}"

        # All generation IDs and filenames must be unique
        gen_ids = [r["data"]["generation_id"] for r in results]
        filenames = [r["data"]["filename"] for r in results]
        assert len(set(gen_ids)) == num_threads, "Generation IDs must be unique"
        assert len(set(filenames)) == num_threads, "Filenames must be unique"
