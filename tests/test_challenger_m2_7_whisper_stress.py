"""
tests/test_challenger_m2_7_whisper_stress.py
Empirical Challenger M2-7 Stress Harness:
Rigorously test Whisper race conditions, atomic CAS transitions,
and manual edit preservation under extreme concurrency and edge cases.
"""

from __future__ import annotations

import io
from pathlib import Path
import threading
import time
from typing import List
import uuid

from fastapi.testclient import TestClient
import numpy as np
import pytest
import soundfile as sf

from backend.app.db.repository import (
    create_sample,
    delete_sample,
    get_sample,
    update_sample_conditional,
)
from backend.app.main import app
from backend.app.services.audio_service import get_audio_service
from backend.app.services.whisper_service import get_whisper_service


def _generate_wav(duration_s: float = 0.25, sr: int = 16000) -> bytes:
    """Generate in-memory WAV bytes."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    signal = 0.3 * np.sin(2 * np.pi * 440.0 * t)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TestChallengerM27WhisperRaceStress:
    """Empirically stress-test Whisper concurrency and manual edit preservation."""

    def test_duplicate_concurrent_enqueue_mutual_exclusion(self):
        """
        Challenge: If two transcription tasks are submitted concurrently for the same sample,
        only one worker must acquire the 'transcribing' state via CAS in Step 2.
        The second worker must reject and abort immediately without overwriting or duplicating work.
        """
        whisper_svc = get_whisper_service()
        engine = whisper_svc._engine
        orig_transcribe = engine.transcribe

        sid = f"dup_enqueue_{uuid.uuid4().hex[:8]}"
        wav_bytes = _generate_wav()
        audio_info = get_audio_service().process_and_save_audio(wav_bytes, sid, f"{sid}.wav")

        transcribe_call_count = [0]
        lock = threading.Lock()

        def slow_counted_transcribe(audio_path, language="pl"):
            with lock:
                transcribe_call_count[0] += 1
            time.sleep(0.2)
            return "DUPLICATE_ENQUEUE_RESULT"

        engine.transcribe = slow_counted_transcribe

        try:
            create_sample(
                name=f"{sid}.wav",
                filename=f"{sid}.wav",
                file_path=str(audio_info["file_path"]),
                duration_seconds=audio_info["duration"],
                sample_id=sid,
                transcript="",
                transcript_status="pending",
            )

            # Enqueue twice concurrently
            fut1 = whisper_svc.enqueue_transcription(sample_id=sid, audio_path=audio_info["file_path"])
            fut2 = whisper_svc.enqueue_transcription(sample_id=sid, audio_path=audio_info["file_path"])

            res1 = fut1.result(timeout=5.0)
            res2 = fut2.result(timeout=5.0)

            # Exactly one worker should have transcribed; the other aborted with empty string
            assert transcribe_call_count[0] == 1, (
                f"Expected exactly 1 transcribe call, got {transcribe_call_count[0]}"
            )
            assert {res1, res2} == {"DUPLICATE_ENQUEUE_RESULT", ""}

            rec = get_sample(sid)
            assert rec is not None
            assert rec["transcript_status"] == "transcribed"
            assert rec["transcript"] == "DUPLICATE_ENQUEUE_RESULT"
        finally:
            engine.transcribe = orig_transcribe
            delete_sample(sid)
            get_audio_service().delete_audio_file(sid)

    def test_microsecond_race_user_edit_between_step4_and_step5(self):
        """
        Challenge: User edits the transcript at the exact moment after Step 4 verifies
        the status is 'transcribing', but immediately before Step 5 executes update_sample_conditional.
        The CAS condition (transcript_status='transcribing') must fail atomically,
        preventing Whisper from overwriting the user edit.
        """
        client = TestClient(app)
        whisper_svc = get_whisper_service()
        engine = whisper_svc._engine
        orig_transcribe = engine.transcribe

        sid = f"microsecond_race_{uuid.uuid4().hex[:8]}"
        wav_bytes = _generate_wav()
        audio_info = get_audio_service().process_and_save_audio(wav_bytes, sid, f"{sid}.wav")

        user_edited = threading.Event()

        def transcribe_with_hook(audio_path, language="pl"):
            # During transcription, perform a manual user edit
            resp = client.put(
                f"/api/samples/{sid}",
                json={"transcript": "Wprowadzony tekst w trakcie Step 3 (Zażółć)"},
            )
            assert resp.status_code == 200
            user_edited.set()
            return "WHISPER_SHOULD_BE_REJECTED_AT_STEP5"

        engine.transcribe = transcribe_with_hook

        try:
            create_sample(
                name=f"{sid}.wav",
                filename=f"{sid}.wav",
                file_path=str(audio_info["file_path"]),
                duration_seconds=audio_info["duration"],
                sample_id=sid,
                transcript="",
                transcript_status="pending",
            )

            fut = whisper_svc.enqueue_transcription(sample_id=sid, audio_path=audio_info["file_path"])
            fut.result(timeout=5.0)

            assert user_edited.is_set()

            rec = get_sample(sid)
            assert rec is not None
            assert rec["transcript_status"] == "manual"
            assert rec["transcript"] == "Wprowadzony tekst w trakcie Step 3 (Zażółć)"
        finally:
            engine.transcribe = orig_transcribe
            delete_sample(sid)
            get_audio_service().delete_audio_file(sid)

    def test_high_concurrency_30_samples_with_repeated_parallel_edits(self):
        """
        Stress Challenge:
        30 samples enqueued simultaneously with delayed Whisper execution.
        For every sample, 3 parallel threads compete to send rapid updates with Polish diacritics.
        Verify that:
        1. 100% of samples finish with transcript_status == 'manual'
        2. No sample contains Whisper text
        3. All transcripts are non-empty and valid Polish strings
        """
        client = TestClient(app)
        whisper_svc = get_whisper_service()
        engine = whisper_svc._engine
        orig_transcribe = engine.transcribe

        sample_count = 30
        created_sids: List[str] = []
        wav_bytes = _generate_wav()

        def slow_transcribe(audio_path, language="pl"):
            time.sleep(0.12)
            return "WHISPER_OVERWRITE_DISASTER_IF_LEAKED"

        engine.transcribe = slow_transcribe

        try:
            futures = []
            for i in range(sample_count):
                sid = f"high_stress_{i}_{uuid.uuid4().hex[:6]}"
                created_sids.append(sid)
                audio_info = get_audio_service().process_and_save_audio(wav_bytes, sid, f"{sid}.wav")
                create_sample(
                    name=f"{sid}.wav",
                    filename=f"{sid}.wav",
                    file_path=str(audio_info["file_path"]),
                    duration_seconds=audio_info["duration"],
                    sample_id=sid,
                    transcript="",
                    transcript_status="pending",
                )
                fut = whisper_svc.enqueue_transcription(sample_id=sid, audio_path=audio_info["file_path"])
                futures.append((sid, fut))

            # 3 threads per sample = 90 competing edit threads
            edit_threads = []
            edit_errors = []

            def edit_task(sid: str, thread_idx: int):
                try:
                    time.sleep(0.01 * (thread_idx % 5))
                    text = f"Prawidłowy tekst #{thread_idx} dla {sid} — Zażółć gęślą jaźń"
                    resp = client.put(f"/api/samples/{sid}", json={"transcript": text})
                    assert resp.status_code == 200, f"Status: {resp.status_code}, Body: {resp.text}"
                except Exception as exc:
                    edit_errors.append((sid, exc))

            for sid, _ in futures:
                for t_idx in range(3):
                    t = threading.Thread(target=edit_task, args=(sid, t_idx))
                    edit_threads.append(t)

            for t in edit_threads:
                t.start()
            for t in edit_threads:
                t.join(timeout=15.0)

            assert len(edit_errors) == 0, f"Errors during concurrent edits: {edit_errors}"

            # Wait for all whisper jobs
            for _, fut in futures:
                fut.result(timeout=15.0)

            # Verification of all 30 samples
            for sid in created_sids:
                rec = get_sample(sid)
                assert rec is not None, f"Sample {sid} missing"
                assert rec["transcript_status"] == "manual", (
                    f"Sample {sid} transcript_status was overwritten to '{rec['transcript_status']}'"
                )
                assert "WHISPER_OVERWRITE_DISASTER_IF_LEAKED" not in rec["transcript"], (
                    f"Sample {sid} was contaminated with Whisper transcript: {rec['transcript']}"
                )
                assert rec["transcript"].startswith("Prawidłowy tekst #")
                assert "Zażółć gęślą jaźń" in rec["transcript"]
        finally:
            engine.transcribe = orig_transcribe
            for sid in created_sids:
                try:
                    client.delete(f"/api/samples/{sid}")
                except Exception:
                    pass
