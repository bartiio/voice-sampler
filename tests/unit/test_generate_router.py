"""
tests/unit/test_generate_router.py
Comprehensive unit tests for Voice Generation API Routers:
- POST /api/generate/tts (F5-TTS speech synthesis)
- POST /api/generate/sts (RVC v2 voice conversion)
- GET /api/generate/history and GET /api/generate (generation history listing)
- GET /api/generate/{id} (metadata retrieval)
- DELETE /api/generate/{id} (generation record & audio deletion)
- GET /api/audio/generated/{filename} (RFC 7233 range-enabled audio streaming)
- GET /api/audio/download/{filename} (attachment download with traversal defenses)

Part of Voice Sampler Milestone 4 (M4).
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Any, Dict
import uuid

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.db.connection import init_storage_directories, set_db_path
from backend.app.main import app


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path):
    """Isolate storage directories and SQLite database per test."""
    test_db = tmp_path / "test_generate_router.db"
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


def _seed_audio_sample(
    sample_id: str = "sample_test_1",
    filename: str = "sample_test_1.wav",
    transcript: str = "Głos testowy do syntezy",
    duration_s: float = 2.0,
    sample_rate: int = 24000,
) -> Dict[str, Any]:
    """Create a sample on disk and register it in SQLite."""
    settings = get_settings()
    file_path = settings.samples_raw_dir / filename
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False, dtype=np.float32)
    audio = (0.5 * np.sin(2 * np.pi * 220.0 * t) * 32767.0).astype(np.int16)
    sf.write(str(file_path), audio, sample_rate, subtype="PCM_16", format="WAV")

    return repository.create_sample(
        name=filename,
        filename=filename,
        file_path=str(file_path),
        duration_seconds=duration_s,
        sample_rate=sample_rate,
        channels=1,
        transcript=transcript,
        sample_id=sample_id,
        transcript_status="transcribed",
    )


def _seed_f5tts_model(
    model_name: str = "f5_test_voice",
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create simulated F5-TTS model checkpoint on disk and register in DB."""
    settings = get_settings()
    model_dir = settings.models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = settings.models_dir / f"{model_name}_f5tts.pt"
    ckpt_path.write_bytes(b"F5TTS_DUMMY_CKPT_BYTES")
    vocab_path = model_dir / "vocab.txt"
    vocab_path.write_text("a\nb\nc\n", encoding="utf-8")
    config_path = model_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    return repository.create_model(
        name=model_name,
        engine="f5-tts",
        checkpoint_path=f"data/models/{model_name}_f5tts.pt",
        vocab_path=f"data/models/{model_name}/vocab.txt",
        config_path=f"data/models/{model_name}/config.json",
        parameters={"sample_ids": [sample_id] if sample_id else []},
        sample_count=1 if sample_id else 0,
    )


def _seed_rvc_model(
    model_name: str = "rvc_test_voice",
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Create simulated RVC v2 model checkpoint on disk and register in DB."""
    settings = get_settings()
    model_dir = settings.models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = settings.models_dir / f"{model_name}.pth"
    ckpt_path.write_bytes(b"RVC_DUMMY_CKPT_BYTES")
    index_path = settings.models_dir / f"{model_name}.index"
    index_path.write_bytes(b"RVC_DUMMY_INDEX_BYTES")
    config_path = model_dir / "config.json"
    config_path.write_text("{}", encoding="utf-8")

    return repository.create_model(
        name=model_name,
        engine="rvc",
        checkpoint_path=f"data/models/{model_name}.pth",
        index_path=f"data/models/{model_name}.index",
        config_path=f"data/models/{model_name}/config.json",
        parameters={"sample_ids": [sample_id] if sample_id else []},
        sample_count=1 if sample_id else 0,
    )


# ---------------------------------------------------------------------------
# TTS Inference Tests
# ---------------------------------------------------------------------------

def test_generate_tts_success():
    client = TestClient(app)
    sample = _seed_audio_sample("sample_1")
    _seed_f5tts_model("voice_1", "sample_1")

    payload = {
        "text": "Witaj świecie, to jest test syntezy mowy.",
        "model_name": "voice_1",
        "reference_sample_id": "sample_1",
        "speed": 1.0,
    }
    resp = client.post("/api/generate/tts", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "generation_id" in data
    assert "audio_url" in data
    assert "filename" in data
    assert data["filename"].startswith("tts_")
    assert data["filename"].endswith(".wav")
    assert data["audio_url"] == f"/api/audio/generated/{data['filename']}"

    # Verify audio file exists on disk and is valid WAV
    settings = get_settings()
    out_file = settings.generated_dir / data["filename"]
    assert out_file.is_file()
    audio, sr = sf.read(str(out_file))
    assert sr == 24000
    assert len(audio) > 1000


def test_generate_tts_rejects_empty_and_whitespace_text():
    client = TestClient(app)
    _seed_f5tts_model("voice_empty")

    resp1 = client.post("/api/generate/tts", json={"text": "", "model_name": "voice_empty"})
    assert resp1.status_code == 400
    assert "Text cannot be empty" in resp1.json()["detail"]

    resp2 = client.post("/api/generate/tts", json={"text": "   \t\n   ", "model_name": "voice_empty"})
    assert resp2.status_code == 400
    assert "Text cannot be empty" in resp2.json()["detail"]


def test_generate_tts_rejects_nonexistent_model():
    client = TestClient(app)
    _seed_audio_sample("sample_2")

    resp = client.post(
        "/api/generate/tts",
        json={"text": "Test", "model_name": "nonexistent_model", "reference_sample_id": "sample_2"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_generate_tts_rejects_nonexistent_reference_sample():
    client = TestClient(app)
    _seed_f5tts_model("voice_ref_test")

    resp = client.post(
        "/api/generate/tts",
        json={"text": "Test", "model_name": "voice_ref_test", "reference_sample_id": "ghost_sample_uuid"},
    )
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"]


def test_generate_tts_supports_all_polish_diacritics():
    client = TestClient(app)
    _seed_f5tts_model("voice_diacritics")

    # Polish pangram with all 32 diacritics lower and upper
    pangram = "Pchnąć w tę łódź jeża lub ośm skrzyń fig (Zażółć gęślą jaźń)."
    resp = client.post("/api/generate/tts", json={"text": pangram, "model_name": "voice_diacritics"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["duration"] is not None
    assert data["duration"] > 1.0


def test_generate_tts_persists_in_sqlite():
    client = TestClient(app)
    _seed_f5tts_model("voice_sql")

    resp = client.post(
        "/api/generate/tts",
        json={"text": "Sprawdzamy bazę SQLite", "model_name": "voice_sql"},
    )
    assert resp.status_code == 200
    gen_id = resp.json()["generation_id"]

    rec = repository.get_generation(gen_id)
    assert rec is not None
    assert rec["type"] == "tts"
    assert rec["model_id"] == "voice_sql"
    assert rec["input_text"] == "Sprawdzamy bazę SQLite"
    assert rec["sample_rate"] == 24000


# ---------------------------------------------------------------------------
# STS Conversion Tests
# ---------------------------------------------------------------------------

def test_generate_sts_success():
    client = TestClient(app)
    sample = _seed_audio_sample("sts_sample_1")
    _seed_rvc_model("rvc_voice_1", "sts_sample_1")

    payload = {
        "input_sample_id": "sts_sample_1",
        "model_name": "rvc_voice_1",
        "pitch_shift": 0,
        "index_rate": 0.75,
    }
    resp = client.post("/api/generate/sts", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "generation_id" in data
    assert "filename" in data
    assert data["filename"].startswith("sts_")
    assert data["filename"].endswith(".wav")
    assert data["audio_url"] == f"/api/audio/generated/{data['filename']}"

    # Verify output audio file exists on disk and is 40kHz WAV
    settings = get_settings()
    out_file = settings.generated_dir / data["filename"]
    assert out_file.is_file()
    audio, sr = sf.read(str(out_file))
    assert sr == 40000
    assert len(audio) > 1000


def test_generate_sts_pitch_shift_boundaries():
    client = TestClient(app)
    _seed_audio_sample("sts_sample_bounds")
    _seed_rvc_model("rvc_voice_bounds")

    # Valid boundary limits: -24 and +24
    for pitch in [-24, 0, 12, 24]:
        resp = client.post(
            "/api/generate/sts",
            json={
                "input_sample_id": "sts_sample_bounds",
                "model_name": "rvc_voice_bounds",
                "pitch_shift": pitch,
                "index_rate": 0.5,
            },
        )
        assert resp.status_code == 200

    # Invalid boundary limits: -25 and +25
    for pitch in [-25, 25, 30]:
        resp = client.post(
            "/api/generate/sts",
            json={
                "input_sample_id": "sts_sample_bounds",
                "model_name": "rvc_voice_bounds",
                "pitch_shift": pitch,
                "index_rate": 0.5,
            },
        )
        assert resp.status_code == 400
        assert "Pitch shift must be between -24 and +24" in resp.json()["detail"]


def test_generate_sts_index_rate_boundaries():
    client = TestClient(app)
    _seed_audio_sample("sts_sample_idx")
    _seed_rvc_model("rvc_voice_idx")

    # Valid index rates: 0.0, 0.5, 1.0
    for rate in [0.0, 0.5, 1.0]:
        resp = client.post(
            "/api/generate/sts",
            json={
                "input_sample_id": "sts_sample_idx",
                "model_name": "rvc_voice_idx",
                "pitch_shift": 0,
                "index_rate": rate,
            },
        )
        assert resp.status_code == 200

    # Invalid index rates: -0.01 and 1.05
    for rate in [-0.01, 1.05]:
        resp = client.post(
            "/api/generate/sts",
            json={
                "input_sample_id": "sts_sample_idx",
                "model_name": "rvc_voice_idx",
                "pitch_shift": 0,
                "index_rate": rate,
            },
        )
        assert resp.status_code == 400
        assert "Index rate must be between 0.0 and 1.0" in resp.json()["detail"]


def test_generate_sts_rejects_nonexistent_resources():
    client = TestClient(app)
    _seed_audio_sample("real_sample")
    _seed_rvc_model("real_model")

    # Nonexistent sample -> 404
    resp1 = client.post(
        "/api/generate/sts",
        json={"input_sample_id": "ghost_sample", "model_name": "real_model"},
    )
    assert resp1.status_code == 404

    # Nonexistent model -> 404
    resp2 = client.post(
        "/api/generate/sts",
        json={"input_sample_id": "real_sample", "model_name": "ghost_model"},
    )
    assert resp2.status_code == 404


# ---------------------------------------------------------------------------
# Audio Playback & Streaming Tests
# ---------------------------------------------------------------------------

def test_stream_generated_audio_full_200():
    client = TestClient(app)
    _seed_f5tts_model("voice_stream")
    gen_resp = client.post(
        "/api/generate/tts",
        json={"text": "Test strumieniowania dźwięku", "model_name": "voice_stream"},
    )
    filename = gen_resp.json()["filename"]

    stream_resp = client.get(f"/api/audio/generated/{filename}")
    assert stream_resp.status_code == 200
    assert stream_resp.headers.get("Content-Type") == "audio/wav"
    assert stream_resp.headers.get("Accept-Ranges") == "bytes"
    assert int(stream_resp.headers.get("Content-Length", 0)) > 1000
    assert len(stream_resp.content) > 1000


def test_stream_generated_audio_byte_range_206():
    client = TestClient(app)
    _seed_f5tts_model("voice_range")
    gen_resp = client.post(
        "/api/generate/tts",
        json={"text": "Test odtwarzacza i nagłówka Range", "model_name": "voice_range"},
    )
    filename = gen_resp.json()["filename"]

    # Request first 101 bytes
    resp = client.get(f"/api/audio/generated/{filename}", headers={"Range": "bytes=0-100"})
    assert resp.status_code == 206
    assert len(resp.content) == 101
    assert resp.headers.get("Content-Range", "").startswith("bytes 0-100/")
    assert resp.headers.get("Content-Length") == "101"


def test_stream_generated_audio_invalid_range_416():
    client = TestClient(app)
    _seed_f5tts_model("voice_range_416")
    gen_resp = client.post(
        "/api/generate/tts",
        json={"text": "Test błędu 416", "model_name": "voice_range_416"},
    )
    filename = gen_resp.json()["filename"]

    resp = client.get(f"/api/audio/generated/{filename}", headers={"Range": "bytes=9999999-99999999"})
    assert resp.status_code == 416
    assert "Content-Range" in resp.headers


def test_stream_generated_audio_path_traversal_defense():
    client = TestClient(app)
    # Defense against directory traversal
    for bad_name in ["../etc/passwd", "..\\boot.ini", "sub/file.wav"]:
        resp = client.get(f"/api/audio/generated/{bad_name}")
        assert resp.status_code in [400, 404]


def test_stream_nonexistent_generated_audio_404():
    client = TestClient(app)
    resp = client.get("/api/audio/generated/nonexistent_voice_output.wav")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Audio Download Tests
# ---------------------------------------------------------------------------

def test_download_generated_audio_attachment():
    client = TestClient(app)
    _seed_f5tts_model("voice_dl")
    gen_resp = client.post(
        "/api/generate/tts",
        json={"text": "Test pobierania pliku dźwiękowego", "model_name": "voice_dl"},
    )
    filename = gen_resp.json()["filename"]

    dl_resp = client.get(f"/api/audio/download/{filename}")
    assert dl_resp.status_code == 200
    disposition = dl_resp.headers.get("Content-Disposition", "")
    assert "attachment" in disposition
    assert filename in disposition
    assert int(dl_resp.headers.get("Content-Length", 0)) == len(dl_resp.content)


def test_download_nonexistent_audio_404():
    client = TestClient(app)
    resp = client.get("/api/audio/download/missing_file.wav")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# History & Metadata Management Tests
# ---------------------------------------------------------------------------

def test_list_and_get_generation_metadata():
    client = TestClient(app)
    _seed_f5tts_model("voice_hist")

    resp1 = client.post("/api/generate/tts", json={"text": "Pierwsza generacja", "model_name": "voice_hist"})
    resp2 = client.post("/api/generate/tts", json={"text": "Druga generacja", "model_name": "voice_hist"})

    gen1_id = resp1.json()["generation_id"]
    gen2_id = resp2.json()["generation_id"]

    # History listing
    hist_resp = client.get("/api/generate/history")
    assert hist_resp.status_code == 200
    history = hist_resp.json()["generations"]
    ids = [item["id"] for item in history]
    assert gen1_id in ids
    assert gen2_id in ids

    # Single metadata lookup
    meta_resp = client.get(f"/api/generate/{gen1_id}")
    assert meta_resp.status_code == 200
    data = meta_resp.json()
    assert data["id"] == gen1_id
    assert data["type"] == "tts"
    assert data["model_id"] == "voice_hist"


def test_delete_generation_record_and_file():
    client = TestClient(app)
    _seed_f5tts_model("voice_del")

    gen_resp = client.post("/api/generate/tts", json={"text": "Do usunięcia", "model_name": "voice_del"})
    gen_id = gen_resp.json()["generation_id"]
    filename = gen_resp.json()["filename"]

    settings = get_settings()
    out_file = settings.generated_dir / filename
    assert out_file.is_file()

    # Delete generation
    del_resp = client.delete(f"/api/generate/{gen_id}")
    assert del_resp.status_code == 200
    assert del_resp.json()["status"] == "deleted"

    # Verify file unlinked from disk
    assert not out_file.exists()

    # Verify removed from DB
    assert repository.get_generation(gen_id) is None
    assert client.get(f"/api/generate/{gen_id}").status_code == 404
