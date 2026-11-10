"""
Unit tests for AudioService (backend/app/services/audio_service.py).
Tests multi-tier decoding, normalization, duration calculation, and safe filesystem operations.
"""

import io
import math
import struct
import wave
import numpy as np
import pytest
import soundfile as sf

from backend.app.services.audio_service import AudioService, get_audio_service


def _generate_test_wav(duration_s=1.0, sr=24000, freq=440.0):
    num_samples = int(duration_s * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        frames = bytearray()
        for i in range(num_samples):
            val = int(16000 * math.sin(2 * math.pi * freq * (i / sr)))
            frames.extend(struct.pack("<h", val))
        wf.writeframes(frames)
    return buf.getvalue()


def test_audio_service_singleton():
    svc1 = get_audio_service()
    svc2 = get_audio_service()
    assert svc1 is svc2


def test_decode_empty_buffer_raises_value_error():
    svc = get_audio_service()
    with pytest.raises(ValueError, match="Empty audio buffer"):
        svc.decode_audio_bytes(b"")


def test_decode_valid_wav():
    svc = get_audio_service()
    wav_bytes = _generate_test_wav(duration_s=1.0, sr=24000)
    audio, sr = svc.decode_audio_bytes(wav_bytes)
    assert sr == 24000
    assert len(audio) == 24000
    assert isinstance(audio, np.ndarray)


def test_normalize_to_pcm16_mono():
    svc = get_audio_service()
    # Create stereo float array with sample rate 24000
    stereo = np.ones((24000, 2), dtype=np.float32) * 0.5
    int16_audio, final_sr = svc.normalize_to_pcm16_mono(stereo, 24000)
    assert final_sr == 44100
    assert int16_audio.ndim == 1
    assert int16_audio.dtype == np.int16
    assert len(int16_audio) == 44100


def test_process_and_save_audio(tmp_path, monkeypatch):
    svc = AudioService()
    monkeypatch.setattr(svc.settings, "samples_raw_dir", tmp_path / "raw")
    monkeypatch.setattr(svc.settings, "samples_preprocessed_dir", tmp_path / "pre")

    wav_bytes = _generate_test_wav(duration_s=1.5, sr=24000)
    sample_id = "test-sample-123"
    result = svc.process_and_save_audio(wav_bytes, sample_id, filename="test.wav")

    assert result["sample_id"] == sample_id
    assert result["sample_rate"] == 44100
    assert result["channels"] == 1
    assert abs(result["duration"] - 1.5) < 0.1
    assert result["file_path"].exists()

    # Verify saved file can be read back and has valid PCM16 format
    data, sr = sf.read(str(result["file_path"]))
    assert sr == 44100
    assert len(data) > 0

    # Path retrieval and deletion
    resolved = svc.get_audio_path(sample_id)
    assert resolved == result["file_path"]

    deleted = svc.delete_audio_file(sample_id)
    assert deleted is True
    assert not result["file_path"].exists()


def test_decode_corrupted_riff_raises_value_error():
    svc = get_audio_service()
    with pytest.raises(ValueError, match="Unable to decode audio content"):
        svc.decode_audio_bytes(b"RIFF\x00\x00\x00\x00WAVEfmt \x10\x00\x00\x00INVALID_DATA_CORRUPT")


def test_decode_plain_text_raises_value_error():
    svc = get_audio_service()
    with pytest.raises(ValueError, match="Unable to decode audio content"):
        svc.decode_audio_bytes(b"This is just a text string, not an audio file.")


def test_decode_random_binary_raises_value_error():
    svc = get_audio_service()
    with pytest.raises(ValueError, match="Unable to decode audio content"):
        svc.decode_audio_bytes(b"\xde\xad\xbe\xef" * 64)


def test_decode_recognized_mock_payload_in_test_mode():
    from tests.e2e.harness import SyntheticAudioGenerator
    svc = get_audio_service()

    # Mock MP3 dummy container
    audio_mp3, sr_mp3 = svc.decode_audio_bytes(SyntheticAudioGenerator.generate_mock_mp3())
    assert sr_mp3 == 44100
    assert len(audio_mp3) > 0

    # Mock FLAC dummy container
    audio_flac, sr_flac = svc.decode_audio_bytes(SyntheticAudioGenerator.generate_mock_flac())
    assert sr_flac == 44100
    assert len(audio_flac) > 0

    # Explicit MOCK_AUDIO_PAYLOAD
    audio_marker, sr_marker = svc.decode_audio_bytes(b"MOCK_AUDIO_PAYLOAD:1.0")
    assert sr_marker == 44100
    assert len(audio_marker) > 0


def test_synthetic_fallback_duration_bounding():
    """Verify that huge mock payloads are capped to 30.0s to avoid DoS memory expansion."""
    from tests.e2e.harness import SyntheticAudioGenerator
    svc = get_audio_service()
    # 5MB mock payload
    big_mock = SyntheticAudioGenerator.generate_mock_mp3(size_bytes=5 * 1024 * 1024)
    audio, sr = svc.decode_audio_bytes(big_mock)
    assert sr == 44100
    duration_s = len(audio) / sr
    assert duration_s <= 30.01
