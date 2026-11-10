"""
Unit tests for WhisperService (backend/app/services/whisper_service.py).
Tests Polish diacritic preservation, simulation engine, fallback logic, and background thread execution.
"""

from pathlib import Path
import pytest

from backend.app.db.repository import create_sample, get_sample
from backend.app.services.whisper_service import (
    SimulationWhisperEngine,
    WhisperService,
    get_whisper_service,
)


def test_whisper_service_singleton():
    svc1 = get_whisper_service()
    svc2 = get_whisper_service()
    assert svc1 is svc2


def test_simulation_whisper_engine_contextual_phrases(tmp_path):
    engine = SimulationWhisperEngine()

    # 1. Microphone recording
    mic_file = tmp_path / "recording_123.wav"
    mic_file.write_bytes(b"dummy_audio_bytes" * 1000)
    text_mic = engine.transcribe(mic_file)
    assert "polsk" in text_mic.lower()
    assert "zażółć gęślą jaźń" in text_mic.lower()

    # 2. Tongue twister
    tongue_file = tmp_path / "szczebrzeszyn.wav"
    tongue_file.write_bytes(b"dummy_audio_bytes" * 1000)
    text_tongue = engine.transcribe(tongue_file)
    assert "chrząszcz" in text_tongue

    # 3. Polish diacritics check (all 9 characters)
    diacritics = ["ą", "ć", "ę", "ł", "ń", "ó", "ś", "ź", "ż"]
    assert any(c in text_mic for c in diacritics)


def test_simulation_whisper_empty_file_raises_error(tmp_path):
    engine = SimulationWhisperEngine()
    empty_file = tmp_path / "empty.wav"
    empty_file.write_bytes(b"")
    with pytest.raises(ValueError, match="Empty or corrupted audio file"):
        engine.transcribe(empty_file)


def test_simulation_whisper_nonexistent_file_raises_error():
    engine = SimulationWhisperEngine()
    with pytest.raises(FileNotFoundError):
        engine.transcribe(Path("non_existent_audio_path_xyz.wav"))


def test_whisper_service_enqueue_and_status_progression(tmp_path):
    svc = get_whisper_service()

    # Create dummy audio file
    dummy_wav = tmp_path / "mic_sample.wav"
    dummy_wav.write_bytes(b"dummy_audio_content" * 1000)

    # Register sample in DB
    sample = create_sample(
        name="test_mic",
        filename="mic_sample.wav",
        file_path=str(dummy_wav),
        duration_seconds=2.0,
        transcript="",
        transcript_status="pending",
    )
    sid = sample["id"]

    # Enqueue transcription
    future = svc.enqueue_transcription(sid, audio_path=dummy_wav)
    result_text = future.result(timeout=5.0)

    assert len(result_text) > 0
    assert "zażółć gęślą jaźń" in result_text.lower()

    # Verify updated in DB
    updated = get_sample(sid)
    assert updated["transcript_status"] == "transcribed"
    assert updated["transcript"] == result_text


def test_whisper_preserves_manual_edit_during_transcription(tmp_path):
    """Verify that background Whisper worker does not overwrite user manual edits."""
    import time
    import threading
    from backend.app.db.repository import update_sample

    svc = get_whisper_service()
    engine = svc._engine
    orig_transcribe = engine.transcribe

    started_evt = threading.Event()

    def slow_transcribe(audio_path, language="pl"):
        started_evt.set()
        time.sleep(0.3)
        return "Whisper generated text that must NOT overwrite"

    engine.transcribe = slow_transcribe

    dummy_wav = tmp_path / "manual_protect.wav"
    dummy_wav.write_bytes(b"dummy_audio_content" * 1000)

    sample = create_sample(
        name="manual_protect",
        filename="manual_protect.wav",
        file_path=str(dummy_wav),
        duration_seconds=1.0,
        transcript="",
        transcript_status="pending",
    )
    sid = sample["id"]

    try:
        fut = svc.enqueue_transcription(sid, audio_path=dummy_wav)
        assert started_evt.wait(timeout=3.0)

        # Simulate user manual edit while transcribing
        user_text = "Ręcznie wprowadzona transkrypcja użytkownika"
        update_sample(sid, transcript=user_text, transcript_status="manual")

        fut.result(timeout=3.0)

        # Check DB: user edit must be preserved!
        final_sample = get_sample(sid)
        assert final_sample["transcript"] == user_text
        assert final_sample["transcript_status"] == "manual"
    finally:
        engine.transcribe = orig_transcribe


def test_whisper_discards_deleted_sample_silently(tmp_path):
    """Verify that if sample is deleted before or during transcription, worker gracefully discards."""
    import time
    import threading
    from backend.app.db.repository import delete_sample

    svc = get_whisper_service()
    engine = svc._engine
    orig_transcribe = engine.transcribe

    started_evt = threading.Event()

    def slow_transcribe(audio_path, language="pl"):
        started_evt.set()
        time.sleep(0.3)
        return "Whisper output"

    engine.transcribe = slow_transcribe

    dummy_wav = tmp_path / "del_sample.wav"
    dummy_wav.write_bytes(b"dummy_audio_content" * 1000)

    sample = create_sample(
        name="del_sample",
        filename="del_sample.wav",
        file_path=str(dummy_wav),
        duration_seconds=1.0,
        transcript="",
        transcript_status="pending",
    )
    sid = sample["id"]

    try:
        fut = svc.enqueue_transcription(sid, audio_path=dummy_wav)
        assert started_evt.wait(timeout=3.0)

        # Delete sample while transcribing
        delete_sample(sid)

        fut.result(timeout=3.0)

        # Check DB: sample is still deleted, not resurrected
        assert get_sample(sid) is None
    finally:
        engine.transcribe = orig_transcribe
