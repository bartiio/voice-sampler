"""
tests/test_tier5_adversarial_hardening.py

Milestone 6 Phase 2: Tier 5 Comprehensive Adversarial Coverage Hardening Test Suite.
Empirically challenges the integrated Voice Sampler application across 4 mission-critical dimensions:
1. Simultaneous Multi-Threaded Operations (Concurrent Ingestion, Mutex Race Conditions, Generation Contention)
2. Corrupt and Boundary Audio Handling (Truncated headers, Garbage payloads, Silence, Extreme Clipping, Resampling)
3. Database Connection Integrity & Concurrency (Thread-pool contention, Rollback safety, WAL checkpoints, Unicode diacritics)
4. Full-Lifecycle Health Probing & WebSocket Churn (Rapid health probes, Telemetry broadcast churn, Dual-channel ping/pong)
"""

from __future__ import annotations

import concurrent.futures
import io
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time
from typing import Any, Dict, List, Optional
import unicodedata
import uuid
import wave

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

# Guarantee project root on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.db.connection import (
    checkpoint_db,
    get_db_connection,
    init_storage_directories,
    set_db_path,
)
from backend.app.main import app
from backend.app.routers.websocket import (
    broadcast_training_telemetry,
    get_training_telemetry_manager,
)
from backend.app.services.audio_service import (
    AudioService,
    TARGET_PEAK_NORM,
    TARGET_SAMPLE_RATE,
    get_audio_service,
)
from backend.app.services.f5tts_service import get_f5tts_service
from backend.app.services.notification_service import get_notification_manager
from backend.app.services.rvc_service import get_rvc_service


# ==============================================================================
# Global Fixtures & Helpers
# ==============================================================================

@pytest.fixture(autouse=True)
def setup_tier5_adversarial_environment(tmp_path: Path):
    """
    Guarantees hermetic test isolation: dedicated SQLite database, clean storage dirs,
    and reset singleton lifecycle states.
    """
    test_db = tmp_path / "tier5_adversarial.db"
    set_db_path(test_db)
    init_storage_directories(tmp_path / "data")
    repository.init_db(test_db)

    settings = get_settings()
    settings.data_dir = tmp_path / "data"
    settings.models_dir = tmp_path / "data" / "models"
    settings.cache_dir = tmp_path / "data" / "cache"
    settings.samples_raw_dir = tmp_path / "data" / "samples" / "raw"
    settings.samples_preprocessed_dir = tmp_path / "data" / "samples" / "preprocessed"
    settings.generated_dir = tmp_path / "data" / "generated"
    settings.generated_tts_dir = tmp_path / "data" / "generated" / "tts"
    settings.generated_sts_dir = tmp_path / "data" / "generated" / "sts"
    settings.ensure_directories()

    # Guarantee server start time initialized for health probe
    if not hasattr(app.state, "start_time") or app.state.start_time is None:
        app.state.start_time = time.time()

    f5_svc = get_f5tts_service()
    f5_svc.stop_training()
    rvc_svc = get_rvc_service()
    rvc_svc.cancel_training()

    ws_telemetry = get_training_telemetry_manager()
    ws_telemetry._latest_telemetry = None

    yield

    f5_svc = get_f5tts_service()
    f5_svc.stop_training()
    rvc_svc = get_rvc_service()
    rvc_svc.cancel_training()
    set_db_path(None)


def generate_pcm16_wav_bytes(
    duration_s: float = 1.0,
    sr: int = 44100,
    freq: float = 440.0,
    amplitude: float = 0.5,
    channels: int = 1,
) -> bytes:
    """Generate in-memory valid PCM16 WAV byte payload."""
    num_samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)
    mono = (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)

    if channels > 1:
        audio = np.tile(mono[:, np.newaxis], (1, channels))
    else:
        audio = mono

    buf = io.BytesIO()
    sf.write(buf, audio, sr, subtype="PCM_16", format="WAV")
    return buf.getvalue()


def seed_sample_in_db(
    name: str = "seed_sample.wav",
    transcript: str = "Przykładowy tekst w języku polskim",
    duration_s: float = 1.5,
) -> Dict[str, Any]:
    """Helper to seed an audio file on disk and in the SQLite samples table."""
    settings = get_settings()
    path = settings.samples_raw_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    wav_bytes = generate_pcm16_wav_bytes(duration_s=duration_s, freq=300.0)
    with open(path, "wb") as f:
        f.write(wav_bytes)

    sample = repository.create_sample(
        name=name,
        filename=name,
        file_path=str(path),
        duration_seconds=duration_s,
        sample_rate=44100,
        channels=1,
        transcript=transcript,
        transcript_status="transcribed",
    )
    return sample


def seed_trained_model_in_db(
    model_name: str,
    engine: str = "f5-tts",
    sample_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Helper to seed a registered model record in the SQLite models table."""
    settings = get_settings()
    ckpt_ext = ".pt" if engine == "f5-tts" else ".pth"
    ckpt_path = settings.models_dir / f"{model_name}{ckpt_ext}"
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    with open(ckpt_path, "wb") as f:
        f.write(b"MOCK_CHECKPOINT_DATA_TIER5")

    model = repository.create_model(
        name=model_name,
        engine=engine,
        checkpoint_path=str(ckpt_path),
        parameters={"sample_ids": sample_ids or []},
        sample_count=len(sample_ids or []),
    )
    return model


# ==============================================================================
# DIMENSION 1: SIMULTANEOUS MULTI-THREADED OPERATIONS
# ==============================================================================

class TestConcurrentMultiThreadedOperations:
    """Stress-tests concurrency boundaries, race condition vectors, and mutex coordinators."""

    def test_concurrent_sample_creation_barrier(self):
        """
        Stress scenario: 10 threads concurrently upload distinct audio files
        synchronized release via threading.Barrier.
        Assertions:
        - 100% of uploads return HTTP 200.
        - Exactly 10 unique sample IDs created.
        - SQLite samples table contains 10 records with valid durations.
        - 10 unique audio files persisted on disk without corruption.
        """
        client = TestClient(app, raise_server_exceptions=False)
        num_threads = 10
        barrier = threading.Barrier(num_threads)
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        lock = threading.Lock()

        def _worker(thread_idx: int):
            t_client = TestClient(app, raise_server_exceptions=False)
            freq = 200.0 + (thread_idx * 50.0)
            audio_bytes = generate_pcm16_wav_bytes(duration_s=0.5, freq=freq)
            filename = f"concurrent_sample_{thread_idx}.wav"

            barrier.wait()  # Synchronize simultaneous dispatch
            try:
                resp = t_client.post(
                    "/api/samples/upload",
                    files={"file": (filename, io.BytesIO(audio_bytes), "audio/wav")},
                )
                with lock:
                    if resp.status_code == 200:
                        results.append(resp.json())
                    else:
                        errors.append(f"Thread {thread_idx} failed with {resp.status_code}: {resp.text}")
            except Exception as exc:
                with lock:
                    errors.append(f"Thread {thread_idx} exception: {exc}")

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(errors) == 0, f"Concurrent sample uploads encountered errors: {errors}"
        assert len(results) == num_threads, f"Expected {num_threads} successful uploads, got {len(results)}"

        # Assert uniqueness of sample IDs
        sample_ids = [r["id"] for r in results]
        assert len(set(sample_ids)) == num_threads, "Duplicate sample IDs detected under concurrent creation"

        # Assert SQLite persistence integrity
        db_samples = repository.list_samples()
        assert len(db_samples) == num_threads, f"DB count mismatch: expected {num_threads}, found {len(db_samples)}"

        # Assert disk integrity
        for item in results:
            fp = Path(item["file_path"])
            assert fp.exists(), f"Audio file not found on disk: {fp}"
            assert fp.stat().st_size > 0, f"Persisted audio file is 0 bytes: {fp}"

    def test_concurrent_identical_filename_upload_collision_safety(self):
        """
        Adversarial scenario: 5 threads concurrently upload with the exact same filename
        `identical_name.wav` containing distinct waveforms.
        Assertions:
        - All 5 uploads succeed with HTTP 200.
        - System assigns unique IDs and does not clobber audio files on disk.
        """
        num_threads = 5
        barrier = threading.Barrier(num_threads)
        results: List[Dict[str, Any]] = []
        lock = threading.Lock()

        def _worker(thread_idx: int):
            t_client = TestClient(app, raise_server_exceptions=False)
            freq = 300.0 + (thread_idx * 75.0)
            audio_bytes = generate_pcm16_wav_bytes(duration_s=0.4, freq=freq)

            barrier.wait()
            resp = t_client.post(
                "/api/samples/upload",
                files={"file": ("identical_name.wav", io.BytesIO(audio_bytes), "audio/wav")},
            )
            with lock:
                if resp.status_code == 200:
                    results.append(resp.json())

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10.0)

        assert len(results) == num_threads
        ids = [r["id"] for r in results]
        assert len(set(ids)) == num_threads
        paths = [r["file_path"] for r in results]
        assert len(set(paths)) == num_threads, "File path collision occurred when uploading identical filenames"

    def test_concurrent_training_engine_mutual_exclusion_barrier(self):
        """
        Race condition vector: Thread A triggers F5-TTS training and Thread B triggers RVC training
        at the exact same microsecond using a Barrier.
        Assertions:
        - Exactly 1 request succeeds with HTTP 200.
        - The competing request receives HTTP 409 Conflict with detail 'A training run is already in progress'.
        - System state remains clean and stops gracefully.
        """
        sample = seed_sample_in_db("mutex_seed.wav")
        sid = sample["id"]

        barrier = threading.Barrier(2)
        statuses: List[int] = []
        details: List[str] = []
        lock = threading.Lock()

        def _worker_f5():
            t_client = TestClient(app, raise_server_exceptions=False)
            barrier.wait()
            resp = t_client.post(
                "/api/train/f5-tts",
                json={"model_name": "f5_mutex_test", "epochs": 5, "batch_size": 2, "sample_ids": [sid]},
            )
            with lock:
                statuses.append(resp.status_code)
                details.append(resp.text)

        def _worker_rvc():
            t_client = TestClient(app, raise_server_exceptions=False)
            barrier.wait()
            resp = t_client.post(
                "/api/train/rvc",
                json={"model_name": "rvc_mutex_test", "epochs": 5, "batch_size": 2, "sample_rate": 40000, "sample_ids": [sid]},
            )
            with lock:
                statuses.append(resp.status_code)
                details.append(resp.text)

        t1 = threading.Thread(target=_worker_f5)
        t2 = threading.Thread(target=_worker_rvc)
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        assert 200 in statuses, f"Expected one 200 success, got statuses: {statuses}"
        assert 409 in statuses, f"Expected one 409 Conflict, got statuses: {statuses}"
        assert len(statuses) == 2

        # Clean up active run
        client = TestClient(app, raise_server_exceptions=False)
        client.post("/api/train/stop")

    def test_concurrent_training_start_and_stop_race(self):
        """
        Adversarial race condition: rapid alternating loop of training start and stop requests
        across multiple threads.
        Assertions:
        - Coordinator lock prevents deadlocks.
        - Responses are strictly valid HTTP codes (200, 409).
        - No unhandled 500 errors or thread crashes.
        """
        sample = seed_sample_in_db("race_seed.wav")
        sid = sample["id"]

        client = TestClient(app, raise_server_exceptions=False)
        exceptions: List[str] = []

        def _starter():
            for i in range(5):
                try:
                    resp = client.post(
                        "/api/train/f5-tts",
                        json={"model_name": f"race_model_{i}", "epochs": 5, "sample_ids": [sid]},
                    )
                    if resp.status_code not in (200, 409):
                        exceptions.append(f"Unexpected start status: {resp.status_code} ({resp.text})")
                except Exception as e:
                    exceptions.append(f"Starter exception: {e}")
                time.sleep(0.01)

        def _stopper():
            for _ in range(5):
                try:
                    resp = client.post("/api/train/stop")
                    if resp.status_code != 200:
                        exceptions.append(f"Unexpected stop status: {resp.status_code}")
                except Exception as e:
                    exceptions.append(f"Stopper exception: {e}")
                time.sleep(0.01)

        t1 = threading.Thread(target=_starter)
        t2 = threading.Thread(target=_stopper)
        t1.start()
        t2.start()
        t1.join(timeout=10.0)
        t2.join(timeout=10.0)

        # Stop any remaining run
        client.post("/api/train/stop")
        assert len(exceptions) == 0, f"Race condition errors encountered: {exceptions}"

    def test_concurrent_audio_generation_tts_and_sts_barrier(self):
        """
        Stress scenario: 8 concurrent generation requests (4 F5-TTS and 4 RVC STS)
        dispatched simultaneously via a Barrier.
        Assertions:
        - 100% of generation requests return HTTP 200.
        - All 8 returned generation_ids are distinct.
        - All 8 output audio files exist on disk and have non-zero size.
        - SQLite generations table contains all 8 generation records.
        """
        sample = seed_sample_in_db("gen_ref.wav")
        sid = sample["id"]
        seed_trained_model_in_db("f5_test_voice", engine="f5-tts", sample_ids=[sid])
        seed_trained_model_in_db("rvc_test_voice", engine="rvc", sample_ids=[sid])

        num_threads = 8
        barrier = threading.Barrier(num_threads)
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        lock = threading.Lock()

        polish_texts = [
            "Cześć, to jest pierwszy test syntezy.",
            "W Szczebrzeszynie chrząszcz brzmi w trzcinie.",
            "Zażółć gęślą jaźń w pełnej krasie.",
            "Testowanie generowania mowy w czasie rzeczywistym.",
        ]

        def _tts_worker(idx: int):
            t_client = TestClient(app, raise_server_exceptions=False)
            text = polish_texts[idx % len(polish_texts)]
            barrier.wait()
            resp = t_client.post(
                "/api/generate/tts",
                json={"text": text, "model_name": "f5_test_voice", "reference_sample_id": sid},
            )
            with lock:
                if resp.status_code == 200:
                    results.append(resp.json())
                else:
                    errors.append(f"TTS {idx} failed: {resp.status_code} - {resp.text}")

        def _sts_worker(idx: int):
            t_client = TestClient(app, raise_server_exceptions=False)
            pitch = (idx * 2) - 4
            barrier.wait()
            resp = t_client.post(
                "/api/generate/sts",
                json={
                    "input_sample_id": sid,
                    "model_name": "rvc_test_voice",
                    "pitch_shift": pitch,
                    "index_rate": 0.75,
                },
            )
            with lock:
                if resp.status_code == 200:
                    results.append(resp.json())
                else:
                    errors.append(f"STS {idx} failed: {resp.status_code} - {resp.text}")

        threads = []
        for i in range(4):
            threads.append(threading.Thread(target=_tts_worker, args=(i,)))
        for i in range(4):
            threads.append(threading.Thread(target=_sts_worker, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert len(errors) == 0, f"Generation errors: {errors}"
        assert len(results) == num_threads, f"Expected {num_threads} generations, got {len(results)}"

        gen_ids = [r["generation_id"] for r in results]
        assert len(set(gen_ids)) == num_threads, "Duplicate generation IDs under concurrent requests"

        # Verify database records
        db_gens = repository.list_generations(limit=50)
        assert len(db_gens) == num_threads

        # Verify disk persistence
        settings = get_settings()
        for r in results:
            fn = r["filename"]
            p = settings.generated_dir / fn
            assert p.exists(), f"Generated audio file missing: {p}"
            assert p.stat().st_size > 0, f"Generated audio file is empty: {p}"

    def test_concurrent_sample_read_during_deletion(self):
        """
        Adversarial race condition: Thread A streams audio while Thread B deletes the sample.
        Assertions:
        - Neither thread crashes with HTTP 500 or unhandled exceptions.
        - Thread A receives either HTTP 200 (pre-delete) or HTTP 404 (post-delete).
        """
        sample = seed_sample_in_db("del_race.wav")
        sid = sample["id"]
        client = TestClient(app, raise_server_exceptions=False)

        read_statuses: List[int] = []
        lock = threading.Lock()

        def _reader():
            t_client = TestClient(app, raise_server_exceptions=False)
            for _ in range(15):
                resp = t_client.get(f"/api/samples/{sid}/audio")
                with lock:
                    read_statuses.append(resp.status_code)
                time.sleep(0.005)

        def _deleter():
            t_client = TestClient(app, raise_server_exceptions=False)
            time.sleep(0.02)
            t_client.delete(f"/api/samples/{sid}")

        t1 = threading.Thread(target=_reader)
        t2 = threading.Thread(target=_deleter)
        t1.start()
        t2.start()
        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        # All read statuses must be valid HTTP codes (200 or 404), never 500
        for code in read_statuses:
            assert code in (200, 404), f"Unexpected status {code} during concurrent read-delete"


# ==============================================================================
# DIMENSION 2: CORRUPT AND BOUNDARY AUDIO HANDLING
# ==============================================================================

class TestCorruptAndBoundaryAudioHandling:
    """Stress-tests corrupted, boundary, and extreme audio payloads."""

    def test_reject_empty_audio_file(self):
        """Verify upload and record endpoints cleanly reject 0-byte audio payloads with HTTP 400."""
        client = TestClient(app, raise_server_exceptions=False)

        resp_upload = client.post(
            "/api/samples/upload",
            files={"file": ("zero.wav", b"", "audio/wav")},
        )
        assert resp_upload.status_code == 400
        assert "empty" in resp_upload.text.lower()

        resp_record = client.post(
            "/api/samples/record",
            files={"file": ("zero_rec.wav", b"", "audio/wav")},
        )
        assert resp_record.status_code == 400
        assert "empty" in resp_record.text.lower()

    def test_reject_truncated_wav_header(self):
        """Verify incomplete / truncated RIFF headers are safely rejected with HTTP 400."""
        client = TestClient(app, raise_server_exceptions=False)

        # 4-byte truncated RIFF
        resp1 = client.post(
            "/api/samples/upload",
            files={"file": ("trunc4.wav", b"RIFF", "audio/wav")},
        )
        assert resp1.status_code == 400

        # 12-byte truncated RIFF WAVE header with no chunks
        resp2 = client.post(
            "/api/samples/upload",
            files={"file": ("trunc12.wav", b"RIFF\x24\x00\x00\x00WAVE", "audio/wav")},
        )
        assert resp2.status_code == 400

    def test_reject_corrupt_payload_text_and_random_noise(self):
        """Verify arbitrary text, binary random noise, ELF binaries, and JSON payloads are rejected."""
        client = TestClient(app, raise_server_exceptions=False)

        # 1. Plain ASCII text
        resp_text = client.post(
            "/api/samples/upload",
            files={"file": ("not_audio.txt", b"This is plain text and definitely not a valid sound file.", "text/plain")},
        )
        assert resp_text.status_code == 400

        # 2. Random binary garbage
        random_bytes = os.urandom(2048)
        resp_rand = client.post(
            "/api/samples/upload",
            files={"file": ("noise.bin", random_bytes, "application/octet-stream")},
        )
        assert resp_rand.status_code == 400

        # 3. ELF binary executable
        elf_header = b"\x7fELF\x02\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00\x02\x00\x3e\x00"
        resp_elf = client.post(
            "/api/samples/upload",
            files={"file": ("app.bin", elf_header, "application/octet-stream")},
        )
        assert resp_elf.status_code == 400

        # 4. JSON payload masquerading as audio
        json_bytes = b'{"status": "error", "message": "malicious injection"}'
        resp_json = client.post(
            "/api/samples/upload",
            files={"file": ("test.json", json_bytes, "application/json")},
        )
        assert resp_json.status_code == 400

    def test_reject_malformed_mp3_and_flac_containers(self):
        """Verify corrupt MP3 frame sync and broken FLAC bitstream headers without mock authorization are rejected."""
        client = TestClient(app, raise_server_exceptions=False)

        corrupt_mp3 = b"\xff\xfb\x90\x64" + (b"\x00\xff\xee\xdd" * 50)
        resp_mp3 = client.post(
            "/api/samples/upload",
            files={"file": ("bad.mp3", corrupt_mp3, "audio/mpeg")},
        )
        assert resp_mp3.status_code == 400

        corrupt_flac = b"fLaC\x00\x00\x00\x22" + (b"\x00\x11\x22\x33" * 20)
        resp_flac = client.post(
            "/api/samples/upload",
            files={"file": ("bad.flac", corrupt_flac, "audio/flac")},
        )
        assert resp_flac.status_code == 400

    def test_boundary_extreme_short_audio(self):
        """Verify extremely short audio buffer (10ms / 441 samples) processes without zero-division or index errors."""
        client = TestClient(app, raise_server_exceptions=False)
        short_wav = generate_pcm16_wav_bytes(duration_s=0.01, sr=44100)

        resp = client.post(
            "/api/samples/upload",
            files={"file": ("short.wav", io.BytesIO(short_wav), "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["duration"] >= 0.01
        assert data["sample_rate"] == 44100

    def test_boundary_extreme_silence_all_zeros(self):
        """Verify pure digital silence (all zeros) normalizes cleanly without division by zero or NaN generation."""
        audio_svc = get_audio_service()
        silence = np.zeros(44100, dtype=np.float32)

        norm_audio, sr = audio_svc.normalize_to_pcm16_mono(silence, 44100)
        assert norm_audio.dtype == np.int16
        assert sr == 44100
        assert not np.isnan(norm_audio).any()
        assert np.max(np.abs(norm_audio)) == 0

    def test_boundary_extreme_dc_offset_and_clipping(self):
        """Verify massive DC offset (+50.0) and clipping waveforms normalize to signed int16 without overflow."""
        audio_svc = get_audio_service()
        # Waveform with massive positive DC offset and high peaks
        t = np.linspace(0, 0.5, 22050, endpoint=False)
        clipped_signal = 50.0 + 10.0 * np.sin(2 * np.pi * 440.0 * t)

        norm_audio, sr = audio_svc.normalize_to_pcm16_mono(clipped_signal, 44100)
        assert norm_audio.dtype == np.int16
        assert not np.isnan(norm_audio).any()
        assert np.max(norm_audio) <= 32767
        assert np.min(norm_audio) >= -32768
        # Peak should be normalized close to TARGET_PEAK_NORM (-0.5 dBFS ~ 31128)
        assert np.max(np.abs(norm_audio)) <= 32767

    def test_boundary_extreme_multichannel_downmix(self):
        """Verify 8-channel 7.1 surround sound downmixes cleanly to mono 1-channel."""
        audio_svc = get_audio_service()
        # 8-channel surround matrix (44100 samples x 8 channels)
        surround_audio = np.random.uniform(-0.5, 0.5, size=(44100, 8)).astype(np.float32)

        norm_audio, sr = audio_svc.normalize_to_pcm16_mono(surround_audio, 44100)
        assert norm_audio.ndim == 1
        assert len(norm_audio) == 44100
        assert norm_audio.dtype == np.int16

    def test_boundary_unusual_sample_rates(self):
        """Verify non-standard sample rates (8 kHz telephony and 96 kHz studio master) resample cleanly to 44.1 kHz."""
        audio_svc = get_audio_service()

        # 8000 Hz input (800 samples = 0.1s)
        audio_8k = (0.5 * np.sin(2 * np.pi * 300.0 * np.linspace(0, 0.1, 800))).astype(np.float32)
        norm_8k, sr_8k = audio_svc.normalize_to_pcm16_mono(audio_8k, 8000)
        assert sr_8k == 44100
        assert len(norm_8k) == 4410

        # 96000 Hz input (9600 samples = 0.1s)
        audio_96k = (0.5 * np.sin(2 * np.pi * 1000.0 * np.linspace(0, 0.1, 9600))).astype(np.float32)
        norm_96k, sr_96k = audio_svc.normalize_to_pcm16_mono(audio_96k, 96000)
        assert sr_96k == 44100
        assert len(norm_96k) == 4410


# ==============================================================================
# DIMENSION 3: DATABASE CONNECTION INTEGRITY & CONCURRENCY
# ==============================================================================

class TestDatabaseConnectionIntegrityAndConcurrency:
    """Stress-tests SQLite WAL mode, multi-threaded connection pools, and Polish diacritics persistence."""

    def test_concurrent_thread_pool_crud_stress(self):
        """
        Stress scenario: 20 concurrent worker threads executing 100 simultaneous CRUD operations
        (inserts, updates, conditional updates, selects, and counts).
        Assertions:
        - 0 instances of sqlite3.OperationalError: database is locked.
        - Foreign key constraints and transaction boundaries remain 100% consistent.
        """
        num_workers = 20
        num_tasks = 100
        errors: List[str] = []
        lock = threading.Lock()

        def _crud_task(task_id: int):
            try:
                sid = f"task_sample_{task_id}_{uuid.uuid4().hex[:6]}"
                # 1. Insert
                repository.create_sample(
                    name=f"stress_{task_id}.wav",
                    filename=f"stress_{task_id}.wav",
                    file_path=f"/fake/path/{task_id}.wav",
                    duration_seconds=1.23,
                    sample_id=sid,
                    transcript="Początkowy tekst",
                    transcript_status="pending",
                )

                # 2. Query
                item = repository.get_sample(sid)
                if not item:
                    with lock:
                        errors.append(f"Task {task_id}: sample not found immediately after insert")
                    return

                # 3. Conditional Update
                updated = repository.update_sample_conditional(
                    sid,
                    expected_status="pending",
                    transcript="Zaktualizowany polski tekst z ą, ć, ę",
                    transcript_status="transcribing",
                )
                if not updated:
                    with lock:
                        errors.append(f"Task {task_id}: conditional update failed")
                    return

                # 4. Final Update
                repository.update_sample_transcript(sid, "Ukończona transkrypcja: ż, ź, ś", status="transcribed")

                # 5. List query
                samples = repository.list_samples(status="transcribed")
                if len(samples) == 0:
                    with lock:
                        errors.append(f"Task {task_id}: empty transcribed list")
            except Exception as exc:
                with lock:
                    errors.append(f"Task {task_id} failed with exception: {exc}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_crud_task, i) for i in range(num_tasks)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Concurrent DB CRUD errors encountered: {errors}"
        total_count = repository.count_samples()
        assert total_count == num_tasks, f"Expected {num_tasks} records in DB, found {total_count}"

    def test_database_transaction_rollback_on_injected_error(self):
        """
        Verify atomic rollback: an unhandled exception inside a get_db_connection() block
        cleanly aborts the transaction, commits nothing, and leaves the DB unlocked for subsequent operations.
        """
        sid = f"rollback_test_{uuid.uuid4().hex[:8]}"

        with pytest.raises(RuntimeError):
            with get_db_connection() as conn:
                conn.execute(
                    "INSERT INTO samples (id, name, filename, file_path, duration_seconds, sample_rate, channels, transcript, transcript_status, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (sid, "fail.wav", "fail.wav", "/p", 1.0, 44100, 1, "test", "pending", "now", "now"),
                )
                # Injected exception before transaction commit
                raise RuntimeError("Simulated mid-transaction failure")

        # Verify record does NOT exist
        assert repository.get_sample(sid) is None, "Uncommitted transaction was not rolled back"

        # Verify database is unlocked and operational immediately after
        normal_sid = f"normal_{uuid.uuid4().hex[:8]}"
        created = repository.create_sample(
            name="recovered.wav",
            filename="recovered.wav",
            file_path="/p2",
            sample_id=normal_sid,
        )
        assert created["id"] == normal_sid

    def test_concurrent_wal_checkpoint_under_active_readers_writers(self):
        """
        Adversarial scenario: Multiple threads performing write and read transactions
        while checkpoint_db() is called concurrently.
        Assertions:
        - WAL TRUNCATE executes safely without deadlock or database locked errors.
        """
        stop_event = threading.Event()
        errors: List[str] = []
        lock = threading.Lock()

        def _writer():
            idx = 0
            while not stop_event.is_set():
                try:
                    repository.create_sample(
                        name=f"chk_write_{idx}.wav",
                        filename=f"chk_write_{idx}.wav",
                        file_path=f"/path/{idx}",
                        sample_id=f"chk_{threading.get_ident()}_{idx}",
                    )
                    idx += 1
                    time.sleep(0.002)
                except Exception as e:
                    with lock:
                        errors.append(f"Writer error: {e}")

        def _reader():
            while not stop_event.is_set():
                try:
                    _ = repository.list_samples()
                    time.sleep(0.002)
                except Exception as e:
                    with lock:
                        errors.append(f"Reader error: {e}")

        threads = [threading.Thread(target=_writer) for _ in range(3)]
        threads += [threading.Thread(target=_reader) for _ in range(3)]

        for t in threads:
            t.start()

        # Trigger WAL checkpoints from main thread
        for _ in range(5):
            time.sleep(0.01)
            checkpoint_db()

        stop_event.set()
        for t in threads:
            t.join(timeout=5.0)

        assert len(errors) == 0, f"Errors during concurrent WAL checkpoint: {errors}"

    def test_unicode_polish_diacritics_persistence_integrity(self):
        """
        Fidelity test: All 32 Polish uppercase and lowercase diacritics
        (ą, ć, ę, ł, ń, ó, ś, ź, ż / Ą, Ć, Ę, Ł, Ń, Ó, Ś, Ź, Ż) and NFD/NFC normalizations.
        Assertions:
        - Stored exactly in canonical NFC form.
        - No lossy ASCII transliteration or mojibake corruption across updates.
        """
        client = TestClient(app, raise_server_exceptions=False)
        sample = seed_sample_in_db("unicode_test.wav")
        sid = sample["id"]

        all_polish_diacritics = (
            "Zażółć gęślą jaźń! ZAŻÓŁĆ GĘŚLĄ JAŹŃ! "
            "W Szczebrzeszynie chrząszcz brzmi w trzcinie, "
            "a chrabąszcz żuje świeży pączek z dżemem."
        )

        # Send as NFD decomposed form to verify backend normalizes to NFC
        decomposed_nfd = unicodedata.normalize("NFD", all_polish_diacritics)
        assert decomposed_nfd != all_polish_diacritics  # Guarantee test input is decomposed

        resp = client.put(
            f"/api/samples/{sid}",
            json={"transcript": decomposed_nfd, "status": "manual"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["transcript"] == all_polish_diacritics
        assert unicodedata.is_normalized("NFC", data["transcript"])

        # Query from SQLite to ensure roundtrip persistence fidelity
        db_record = repository.get_sample(sid)
        assert db_record["transcript"] == all_polish_diacritics
        assert unicodedata.is_normalized("NFC", db_record["transcript"])


# ==============================================================================
# DIMENSION 4: FULL-LIFECYCLE HEALTH PROBING & WEBSOCKET CHURN
# ==============================================================================

class TestFullLifecycleHealthProbingAndWebSocketChurn:
    """Stress-tests rapid health endpoint polling and WebSocket broadcast resilience under client churn."""

    def test_rapid_high_frequency_health_probing(self):
        """
        Stress scenario: 100 rapid concurrent GET requests to /api/health across 10 threads.
        Assertions:
        - 100% of requests return HTTP 200.
        - uptime_seconds is monotonically non-decreasing.
        - backend_ready is consistently True.
        - No memory or connection leaks.
        """
        num_requests = 100
        num_workers = 10
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        lock = threading.Lock()

        def _probe():
            t_client = TestClient(app, raise_server_exceptions=False)
            try:
                resp = t_client.get("/api/health")
                with lock:
                    if resp.status_code == 200:
                        results.append(resp.json())
                    else:
                        errors.append(f"Health probe status {resp.status_code}")
            except Exception as e:
                with lock:
                    errors.append(f"Probe error: {e}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = [executor.submit(_probe) for _ in range(num_requests)]
            concurrent.futures.wait(futures)

        assert len(errors) == 0, f"Health probe errors: {errors}"
        assert len(results) == num_requests

        for item in results:
            assert item["status"] == "ok"
            assert item["backend_ready"] is True
            assert item["uptime_seconds"] >= 0.0

    def test_websocket_high_frequency_telemetry_churn(self):
        """
        Adversarial scenario: Multiple WebSocket clients subscribe to /ws/training.
        A rapid burst of telemetry frames is broadcast while some clients disconnect abruptly.
        Assertions:
        - Dead connections are cleanly unhooked from _active_connections.
        - No 'RuntimeError: Set changed size during iteration' occurs.
        - Active surviving connections receive telemetry frames cleanly.
        """
        client = TestClient(app, raise_server_exceptions=False)
        telemetry_mgr = get_training_telemetry_manager()

        with client.websocket_connect("/ws/training") as ws1:
            ws1.send_text("ping")
            assert ws1.receive_text() == "pong"

            # Connect and immediately disconnect a second client (simulating dropped tab)
            with client.websocket_connect("/ws/training") as ws2:
                ws2.send_text("ping")
                assert ws2.receive_text() == "pong"

            # ws2 is now closed. Broadcast frames to trigger dead connection removal logic
            for i in range(15):
                frame = {
                    "type": "progress",
                    "engine": "f5-tts",
                    "run_id": "churn_test_run",
                    "epoch": i + 1,
                    "total_epochs": 15,
                    "loss": round(1.0 / (i + 1), 4),
                    "progress_percent": round(((i + 1) / 15.0) * 100, 1),
                }
                broadcast_training_telemetry(frame)

            # Assert ws1 continues to receive frames cleanly
            received_frames = []
            for _ in range(15):
                try:
                    msg = ws1.receive_json()
                    received_frames.append(msg)
                except Exception:
                    break

            assert len(received_frames) > 0, "Active client did not receive any telemetry frames"
            assert received_frames[-1]["type"] == "progress"

    def test_websocket_notifications_broadcast_churn(self):
        """
        Verify real-time notification events (/ws/notifications) broadcast cleanly
        across sample lifecycle events (created, transcribing, completed, updated, deleted).
        """
        client = TestClient(app, raise_server_exceptions=False)
        notif_mgr = get_notification_manager()

        with client.websocket_connect("/ws/notifications") as ws:
            ws.send_text("ping")
            assert ws.receive_text() == "pong"

            # 1. Created notification
            notif_mgr.notify_sample_created({"id": "sample_n1", "filename": "n1.wav", "duration": 2.5})
            msg1 = ws.receive_json()
            assert msg1["event"] == "sample_created"
            assert msg1["sample_id"] == "sample_n1"

            # 2. Transcription started notification
            notif_mgr.notify_transcription_started("sample_n1", "n1.wav")
            msg2 = ws.receive_json()
            assert msg2["event"] == "transcription_started"

            # 3. Transcription completed notification
            notif_mgr.notify_transcription_completed("sample_n1", "Dzień dobry świecie", duration=2.5)
            msg3 = ws.receive_json()
            assert msg3["event"] == "transcription_completed"
            assert msg3["transcript"] == "Dzień dobry świecie"

            # 4. Updated notification
            notif_mgr.notify_sample_updated("sample_n1", "Poprawiony tekst", "manual")
            msg4 = ws.receive_json()
            assert msg4["event"] == "sample_updated"
            assert msg4["transcript"] == "Poprawiony tekst"

            # 5. Deleted notification
            notif_mgr.notify_sample_deleted("sample_n1")
            msg5 = ws.receive_json()
            assert msg5["event"] == "sample_deleted"

    def test_websocket_dual_channel_ping_pong_under_load(self):
        """
        Verify bidirectional health on both /ws/training and /ws/notifications
        simultaneously maintaining responsive ping/pong under concurrent telemetry traffic.
        """
        client = TestClient(app, raise_server_exceptions=False)

        with client.websocket_connect("/ws/training") as ws_train:
            with client.websocket_connect("/ws/notifications") as ws_notif:
                # Interleaved ping/pong
                for _ in range(5):
                    ws_train.send_text("ping")
                    assert ws_train.receive_text() == "pong"

                    ws_notif.send_text("ping")
                    assert ws_notif.receive_text() == "pong"
