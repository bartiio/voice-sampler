"""
tests/test_m2_challenger_m2_6_verification.py
Empirical Challenger M2-6 Verification and Stress Suite:
1. TOCTOU Race Condition Verification: Whisper Step 2 vs Manual Transcript Edit
2. RFC 7233 Range Clamping and Bounds Matrix
3. Audio Ingestion Multi-Channel and Extreme Sample Rate Matrix
4. Polish Diacritics NFC Canonical Normalization & Persistence
5. Concurrent Deletions and Filesystem Unlinking Integrity
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
import math
import os
from pathlib import Path
import struct
import sys
import threading
import time
from typing import Any, Dict, List
import unicodedata
import uuid
import wave

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.app.config import get_settings
from backend.app.db.repository import (
    create_sample,
    delete_sample,
    get_sample,
    list_samples,
    update_sample,
    update_sample_conditional,
    update_sample_status_conditional,
)
from backend.app.main import app
from backend.app.services.audio_service import get_audio_service
from backend.app.services.whisper_service import get_whisper_service


def _make_wav_bytes(
    duration_s: float = 0.5,
    sr: int = 44100,
    channels: int = 1,
    freq: float = 440.0,
) -> bytes:
    """Generate in-memory WAV bytes with custom duration, sample rate, and channels."""
    num_samples = int(duration_s * sr)
    t = np.linspace(0, duration_s, num_samples, endpoint=False, dtype=np.float32)
    signal = 0.5 * np.sin(2 * np.pi * freq * t)

    if channels == 1:
        audio_data = signal
    else:
        # Replicate signal across multiple channels
        audio_data = np.column_stack([signal] * channels)

    buf = io.BytesIO()
    sf.write(buf, audio_data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


# ==============================================================================
# TEST SUITE 1: TOCTOU RACE CONDITION INVESTIGATION
# ==============================================================================

class TestWhisperManualEditRaceConditions:
    """Empirical investigation into manual edit clobbering during Whisper background execution."""

    def test_step2_atomic_conditional_update_prevents_clobbering_manual_status(self):
        """
        VERIFICATION OF RACE CONDITION REMEDIATION:
        When a user edits a transcript while the Whisper job is queued,
        Step 2 of _transcription_worker_task executes:
            update_sample_conditional(sample_id, expected_status="pending", transcript_status="transcribing")
        Because the user already updated transcript_status to 'manual',
        the conditional transition must reject (return None) and NEVER clobber 'manual' to 'transcribing'.
        """
        sid = f"race_fix_{uuid.uuid4().hex[:8]}"
        wav_bytes = _make_wav_bytes(duration_s=0.2)
        audio_info = get_audio_service().process_and_save_audio(wav_bytes, sid, f"{sid}.wav")

        try:
            # 1. Create sample in pending state
            create_sample(
                name=f"{sid}.wav",
                filename=f"{sid}.wav",
                file_path=str(audio_info["file_path"]),
                duration_seconds=audio_info["duration"],
                sample_id=sid,
                transcript="",
                transcript_status="pending",
            )

            # 2. Simulate User Manual Edit before worker reaches Step 2
            user_text = "To jest ważny tekst wprowadzony ręcznie przez użytkownika (Zażółć gęślą jaźń)"
            client = TestClient(app)
            put_resp = client.put(f"/api/samples/{sid}", json={"transcript": user_text})
            assert put_resp.status_code == 200
            assert put_resp.json()["status"] == "manual"

            db_rec_before = get_sample(sid)
            assert db_rec_before is not None
            assert db_rec_before["transcript"] == user_text
            assert db_rec_before["transcript_status"] == "manual"

            # 3. Worker Step 2 execution uses atomic conditional update:
            updated = update_sample_conditional(sid, expected_status="pending", transcript_status="transcribing")
            assert updated is None, "Conditional transition must return None when status is 'manual'"

            db_rec_after_step2 = get_sample(sid)
            assert db_rec_after_step2 is not None
            # Verify that Step 2 did NOT erase 'manual' status
            assert db_rec_after_step2["transcript_status"] == "manual"
            assert db_rec_after_step2["transcript"] == user_text

            # 4. Now simulate Step 5 conditional update if worker erroneously continued:
            whisper_output = "WHISPER_AUTOTRANSCRIBED_TEXT_OVERWRITING_USER"
            sample_rec = update_sample_conditional(
                sid,
                expected_status="transcribing",
                transcript=whisper_output,
                transcript_status="transcribed",
            )
            assert sample_rec is None, "Step 5 conditional update must reject write when status is 'manual'"

            db_rec_final = get_sample(sid)
            assert db_rec_final is not None
            assert db_rec_final["transcript"] == user_text
            assert db_rec_final["transcript_status"] == "manual"
        finally:
            delete_sample(sid)
            get_audio_service().delete_audio_file(sid)

    def test_step5_atomic_conditional_update_prevents_clobbering_during_transcribe(self):
        """
        Verify that if the user manually edits the transcript while Whisper is actively
        transcribing (after Step 2 transitioned to 'transcribing'), Step 5's atomic
        conditional update rejects the write and preserves the user's manual edit.
        """
        sid = f"step5_race_{uuid.uuid4().hex[:8]}"
        wav_bytes = _make_wav_bytes(duration_s=0.2)
        audio_info = get_audio_service().process_and_save_audio(wav_bytes, sid, f"{sid}.wav")

        try:
            # 1. Create sample in 'transcribing' state (Step 2 succeeded)
            create_sample(
                name=f"{sid}.wav",
                filename=f"{sid}.wav",
                file_path=str(audio_info["file_path"]),
                duration_seconds=audio_info["duration"],
                sample_id=sid,
                transcript="",
                transcript_status="transcribing",
            )

            # 2. While Whisper transcribes, user edits transcript via PUT API
            user_text = "Ręczna edycja w trakcie transkrypcji (Łódź, Gęś, Jaźń)"
            client = TestClient(app)
            put_resp = client.put(f"/api/samples/{sid}", json={"transcript": user_text})
            assert put_resp.status_code == 200
            assert put_resp.json()["status"] == "manual"

            # 3. Whisper completes transcription and attempts Step 5 conditional update
            whisper_output = "WHISPER_OUTPUT_THAT_MUST_BE_DISCARDED"
            sample_rec = update_sample_conditional(
                sid,
                expected_status="transcribing",
                transcript=whisper_output,
                transcript_status="transcribed",
            )
            assert sample_rec is None, "Atomic conditional write must reject because status was changed to 'manual'"

            # 4. Confirm in DB that manual edit survived intact
            final_rec = get_sample(sid)
            assert final_rec is not None
            assert final_rec["transcript"] == user_text
            assert final_rec["transcript_status"] == "manual"
        finally:
            delete_sample(sid)
            get_audio_service().delete_audio_file(sid)

    def test_whisper_worker_task_end_to_end_manual_preservation(self):
        """
        Full end-to-end execution of _transcription_worker_task on a sample that is
        manually edited beforehand. Confirms worker returns manual text without error
        and does not modify the sample.
        """
        whisper_svc = get_whisper_service()
        sid = f"worker_task_{uuid.uuid4().hex[:8]}"
        wav_bytes = _make_wav_bytes(duration_s=0.2)
        audio_info = get_audio_service().process_and_save_audio(wav_bytes, sid, f"{sid}.wav")

        try:
            create_sample(
                name=f"{sid}.wav",
                filename=f"{sid}.wav",
                file_path=str(audio_info["file_path"]),
                duration_seconds=audio_info["duration"],
                sample_id=sid,
                transcript="Tekst wpisany przez użytkownika",
                transcript_status="manual",
            )

            result = whisper_svc._transcription_worker_task(sid, Path(audio_info["file_path"]))
            assert result == "Tekst wpisany przez użytkownika"

            final_rec = get_sample(sid)
            assert final_rec["transcript"] == "Tekst wpisany przez użytkownika"
            assert final_rec["transcript_status"] == "manual"
        finally:
            delete_sample(sid)
            get_audio_service().delete_audio_file(sid)


# ==============================================================================
# TEST SUITE 2: RFC 7233 RANGE CLAMPING & BOUNDS MATRIX
# ==============================================================================

class TestRFC7233RangeBoundsMatrix:
    """Rigorous verification of RFC 7233 byte-range clamping and boundary behaviors."""

    @pytest.fixture(autouse=True)
    def setup_sample(self):
        self.client = TestClient(app)
        self.sid = f"rfc_test_{uuid.uuid4().hex[:8]}"
        self.wav_bytes = _make_wav_bytes(duration_s=0.5, sr=44100)
        self.audio_info = get_audio_service().process_and_save_audio(self.wav_bytes, self.sid, f"{self.sid}.wav")
        self.sample = create_sample(
            name=f"{self.sid}.wav",
            filename=f"{self.sid}.wav",
            file_path=str(self.audio_info["file_path"]),
            duration_seconds=self.audio_info["duration"],
            sample_id=self.sid,
            transcript="Test audio range streaming",
            transcript_status="transcribed",
        )
        self.total_len = os.path.getsize(self.audio_info["file_path"])
        yield
        delete_sample(self.sid)
        get_audio_service().delete_audio_file(self.sid)

    def test_range_clamped_to_eof_zero_to_billion(self):
        """bytes=0-999999999 must clamp to EOF, return HTTP 206, and match full file bytes."""
        r = self.client.get(f"/api/samples/{self.sid}/audio", headers={"Range": "bytes=0-999999999"})
        assert r.status_code == 206
        assert r.headers.get("Content-Range") == f"bytes 0-{self.total_len - 1}/{self.total_len}"
        assert len(r.content) == self.total_len
        disk_bytes = open(self.audio_info["file_path"], "rb").read()
        assert r.content == disk_bytes

    def test_range_clamped_with_offset_to_billion(self):
        """bytes=100-999999999 must clamp to EOF, return HTTP 206, and match sliced payload."""
        r = self.client.get(f"/api/samples/{self.sid}/audio", headers={"Range": "bytes=100-999999999"})
        assert r.status_code == 206
        assert r.headers.get("Content-Range") == f"bytes 100-{self.total_len - 1}/{self.total_len}"
        assert len(r.content) == self.total_len - 100

    def test_range_suffix_exceeding_file_size(self):
        """bytes=-999999999 (suffix > total_len) per RFC 7233 must return full representation (206)."""
        r = self.client.get(f"/api/samples/{self.sid}/audio", headers={"Range": "bytes=-999999999"})
        assert r.status_code == 206
        assert r.headers.get("Content-Range") == f"bytes 0-{self.total_len - 1}/{self.total_len}"
        assert len(r.content) == self.total_len

    def test_range_start_at_eof_returns_416(self):
        """bytes={total_len}-{total_len+100} must return HTTP 416 with Content-Range bytes */{total_len}."""
        r = self.client.get(f"/api/samples/{self.sid}/audio", headers={"Range": f"bytes={self.total_len}-{self.total_len+100}"})
        assert r.status_code == 416
        assert r.headers.get("Content-Range") == f"bytes */{self.total_len}"

    def test_range_inverted_boundaries_returns_416(self):
        """bytes=500-100 (start > end) must return HTTP 416."""
        r = self.client.get(f"/api/samples/{self.sid}/audio", headers={"Range": "bytes=500-100"})
        assert r.status_code == 416
        assert r.headers.get("Content-Range") == f"bytes */{self.total_len}"


# ==============================================================================
# TEST SUITE 3: AUDIO INGESTION MULTI-CHANNEL & SAMPLE RATE MATRIX
# ==============================================================================

class TestAudioIngestionAcousticMatrix:
    """Stress test audio decoding across uncommon channel counts and sample rates."""

    @pytest.mark.parametrize("channels", [1, 2, 4, 6, 8])
    def test_multi_channel_downmix(self, channels: int):
        """Audio with 1, 2, 4 (quad), 6 (5.1 surround), or 8 (7.1 surround) channels downmixes to mono."""
        wav_data = _make_wav_bytes(duration_s=0.2, sr=44100, channels=channels)
        audio_svc = get_audio_service()
        sid = f"chan_{channels}_{uuid.uuid4().hex[:6]}"

        info = audio_svc.process_and_save_audio(wav_data, sid, f"{sid}.wav")
        assert info["channels"] == 1
        assert info["sample_rate"] == 44100
        assert info["duration"] > 0.15

        # Verify disk file is playable 16-bit mono
        data, sr = sf.read(str(info["file_path"]))
        assert sr == 44100
        assert data.ndim == 1

        audio_svc.delete_audio_file(sid)

    @pytest.mark.parametrize("sr", [8000, 16000, 22050, 32000, 48000, 96000])
    def test_sample_rate_resampling_fidelity(self, sr: int):
        """Audio sampled at standard and non-standard rates resamples accurately to 44100 Hz."""
        wav_data = _make_wav_bytes(duration_s=0.3, sr=sr, channels=1)
        audio_svc = get_audio_service()
        sid = f"sr_{sr}_{uuid.uuid4().hex[:6]}"

        info = audio_svc.process_and_save_audio(wav_data, sid, f"{sid}.wav")
        assert info["sample_rate"] == 44100
        assert abs(info["duration"] - 0.3) < 0.05

        data, target_sr = sf.read(str(info["file_path"]))
        assert target_sr == 44100
        assert len(data) > 0

        audio_svc.delete_audio_file(sid)


# ==============================================================================
# TEST SUITE 4: POLISH DIACRITICS NFC CANONICAL NORMALIZATION
# ==============================================================================

class TestPolishDiacriticsFidelity:
    """Verify complete Polish diacritic character set and NFC normalization."""

    def test_all_18_polish_diacritics_preserved_in_nfc(self):
        """Ensure all 9 lowercase and 9 uppercase Polish diacritics persist perfectly."""
        client = TestClient(app)
        sid = f"pl_diacritics_{uuid.uuid4().hex[:6]}"
        wav_data = _make_wav_bytes(duration_s=0.2)
        info = get_audio_service().process_and_save_audio(wav_data, sid, f"{sid}.wav")

        create_sample(
            name=f"{sid}.wav",
            filename=f"{sid}.wav",
            file_path=str(info["file_path"]),
            duration_seconds=info["duration"],
            sample_id=sid,
            transcript="",
            transcript_status="pending",
        )

        all_diacritics = "aąbcćdeęfghijklłmnńoóprsśtuwyzźż AĄBCĆDEĘFGHIJKLŁMNŃOÓPRSŚTUWYZŹŻ"
        pangram = "Pchnąć w tę łódź jeża lub ośm skrzyń fig: Zażółć gęślą jaźń."
        complex_text = f"{all_diacritics}\n{pangram}"

        # Submit as decomposed NFD
        nfd_payload = unicodedata.normalize("NFD", complex_text)
        assert nfd_payload != complex_text  # Proves it is decomposed

        resp = client.put(f"/api/samples/{sid}", json={"transcript": nfd_payload})
        assert resp.status_code == 200

        # Verify returned JSON is normalized NFC
        returned = resp.json()["transcript"]
        assert returned == complex_text
        assert unicodedata.is_normalized("NFC", returned)

        # Verify SQLite row directly
        db_rec = get_sample(sid)
        assert db_rec is not None
        assert db_rec["transcript"] == complex_text
        assert unicodedata.is_normalized("NFC", db_rec["transcript"])

        delete_sample(sid)
        get_audio_service().delete_audio_file(sid)


# ==============================================================================
# TEST SUITE 5: CONCURRENT DELETIONS AND UNLINKING INTEGRITY
# ==============================================================================

class TestConcurrentDeletionsAndUnlinking:
    """Stress test concurrent deletion requests and filesystem cleanup."""

    def test_concurrent_deletions_single_winner(self):
        """
        10 threads attempting to delete the exact same sample concurrently.
        Strictly one thread must receive HTTP 200, 9 threads must receive HTTP 404.
        Zero 500 server crashes.
        """
        client = TestClient(app)
        sid = f"del_race_{uuid.uuid4().hex[:6]}"
        wav_data = _make_wav_bytes(duration_s=0.2)
        info = get_audio_service().process_and_save_audio(wav_data, sid, f"{sid}.wav")

        create_sample(
            name=f"{sid}.wav",
            filename=f"{sid}.wav",
            file_path=str(info["file_path"]),
            duration_seconds=info["duration"],
            sample_id=sid,
            transcript="Sample for deletion race",
            transcript_status="transcribed",
        )

        barrier = threading.Barrier(10)
        results = []

        def del_worker():
            try:
                barrier.wait(timeout=5.0)
                r = client.delete(f"/api/samples/{sid}")
                results.append(r.status_code)
            except Exception as e:
                results.append(str(e))

        threads = [threading.Thread(target=del_worker) for _ in range(10)]
        for t in threads: t.start()
        for t in threads: t.join(timeout=10.0)

        assert results.count(200) == 1, f"Expected exactly 1 HTTP 200, got: {results}"
        assert results.count(404) == 9, f"Expected exactly 9 HTTP 404, got: {results}"

        # Confirm audio file is removed from disk
        assert not Path(info["file_path"]).exists()
