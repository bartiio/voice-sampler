"""
tests/test_m4_challenger_tts.py
Adversarial Empirical Challenger Test Suite for Milestone 4: Voice Generation (TTS)

Dimensions of Challenge:
1. Extreme Unicode & Diacritics: Precomposed NFC vs Decomposed NFD for all 18 Polish diacritics
   (ą, ć, ę, ł, ń, ó, ś, ź, ż and Ą, Ć, Ę, Ł, Ń, Ó, Ś, Ź, Ż), mixed forms, and special Polish punctuation.
2. Whitespace Boundary Attacks: Whitespace-only strings (spaces, tabs, newlines, carriage returns,
   non-breaking spaces U+00A0, en/em/thin spaces) strictly returning HTTP 400, plus collapsing of valid whitespace.
3. Text Length Scaling & Proportionality: Single character, short phrase, medium sentence, complex prose,
   verifying strictly monotonic duration growth and safety ceiling clamping.
4. Audio Signal & File Compliance: Byte-level RIFF/WAVE header verification, 24000 Hz sample rate,
   1 channel (mono), 16-bit PCM, non-zero RMS energy, peak normalization, and interrogative pitch rise.
5. High-Concurrency Stress & Thread Safety: Multi-threaded barrier generation, unique filenames and UUIDs,
   filesystem persistence, SQLite concurrency, and concurrent history reads.
6. Parameter Boundary & Defensive Error Handling: Extreme speed clamping (0.5 to 2.0), missing model (404),
   missing reference sample (404), and empty model name (400).
"""

from __future__ import annotations

import concurrent.futures
import math
from pathlib import Path
import struct
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

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.db.connection import init_storage_directories, set_db_path
from backend.app.main import app
from backend.app.services.f5tts_service import get_f5tts_service


# ---------------------------------------------------------------------------
# Test Environment Fixture
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def setup_challenger_environment(tmp_path: Path):
    """
    Hermetically isolates the database and filesystem storage per test.
    Guarantees no state leakage between test invocations.
    """
    test_db = tmp_path / "test_m4_challenger_tts.db"
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
# Helpers
# ---------------------------------------------------------------------------

def _seed_test_audio_sample(
    sample_id: str = "challenger_ref_sample_1",
    filename: str = "challenger_ref_sample_1.wav",
    transcript: str = "Wzorcowa próbka głosu referencyjnego",
    duration_s: float = 2.0,
    sample_rate: int = 24000,
    freq_hz: float = 220.0,
) -> Dict[str, Any]:
    """Creates a synthetic reference audio file on disk and registers it in SQLite."""
    settings = get_settings()
    file_path = settings.samples_raw_dir / filename
    file_path.parent.mkdir(parents=True, exist_ok=True)
    t = np.linspace(0, duration_s, int(duration_s * sample_rate), endpoint=False, dtype=np.float32)
    audio = (0.6 * np.sin(2 * np.pi * freq_hz * t) * 32767.0).astype(np.int16)
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


def _seed_test_f5tts_model(
    model_name: str = "challenger_voice_model",
    sample_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Creates a simulated F5-TTS model directory, checkpoint, vocab, and DB record."""
    settings = get_settings()
    model_dir = settings.models_dir / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = settings.models_dir / f"{model_name}_f5tts.pt"
    ckpt_path.write_bytes(b"CHALLENGER_F5TTS_CKPT_BYTES")
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


# All 18 Polish diacritics (9 lowercase, 9 uppercase)
POLISH_DIACRITICS_LOWER = ["ą", "ć", "ę", "ł", "ń", "ó", "ś", "ź", "ż"]
POLISH_DIACRITICS_UPPER = ["Ą", "Ć", "Ę", "Ł", "Ń", "Ó", "Ś", "Ź", "Ż"]
ALL_POLISH_DIACRITICS = POLISH_DIACRITICS_LOWER + POLISH_DIACRITICS_UPPER


# ===========================================================================
# DIMENSION 1: EXTREME UNICODE & POLISH DIACRITIC HANDLING (NFC vs NFD)
# ===========================================================================

class TestExtremeUnicodeVariations:
    """Stress-tests unicode normalization, canonical decomposition, and diacritics."""

    def test_nfc_precomposed_all_polish_diacritics(self):
        """
        Verify that a complete Polish pangram containing all lower and upper
        diacritics in precomposed NFC form generates valid audio and stores properly.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_nfc")

        pangram_nfc = (
            "Zażółć gęślą jaźń! Pchnąć w tę łódź jeża lub ośm skrzyń fig. "
            "ZAŻÓŁĆ GĘŚLĄ JAŹŃ! PCHNĄĆ W TĘ ŁÓDŹ JEŻA LUB OŚM SKRZYŃ FIG."
        )
        assert unicodedata.is_normalized("NFC", pangram_nfc)

        resp = client.post("/api/generate/tts", json={"text": pangram_nfc, "model_name": "model_nfc"})
        assert resp.status_code == 200, f"Failed with {resp.text}"
        data = resp.json()

        assert "generation_id" in data
        assert data["duration"] > 3.0
        assert data["sample_rate"] == 24000

        # Verify DB stored text is normalized
        rec = repository.get_generation(data["generation_id"])
        assert rec is not None
        assert unicodedata.is_normalized("NFC", rec["input_text"])
        assert "zażółć" in rec["input_text"].lower()

    def test_nfd_decomposed_all_polish_diacritics(self):
        """
        Adversarial test: Send the exact same Polish pangram explicitly decomposed into NFD
        (base letters + combining diacritical marks: ogonek, acute, overdot, etc.).
        Verify the backend transparently normalizes to NFC, avoids corrupting text,
        synthesizes audio, and stores the precomposed form in SQLite.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_nfd")

        pangram_nfc = "Zażółć gęślą jaźń! Pchnąć w tę łódź jeża lub ośm skrzyń fig."
        pangram_nfd = unicodedata.normalize("NFD", pangram_nfc)

        # Decomposed string has strictly more codepoints due to combining accents
        assert len(pangram_nfd) > len(pangram_nfc)
        assert unicodedata.is_normalized("NFD", pangram_nfd)
        assert not unicodedata.is_normalized("NFC", pangram_nfd)

        resp = client.post("/api/generate/tts", json={"text": pangram_nfd, "model_name": "model_nfd"})
        assert resp.status_code == 200, f"NFD submission failed: {resp.text}"
        data = resp.json()

        # Check SQLite record: MUST be normalized to NFC
        rec = repository.get_generation(data["generation_id"])
        assert rec is not None
        assert unicodedata.is_normalized("NFC", rec["input_text"])
        assert rec["input_text"] == pangram_nfc

        # Verify generated audio file on disk is valid WAV
        settings = get_settings()
        out_file = settings.generated_dir / data["filename"]
        assert out_file.is_file()
        audio, sr = sf.read(str(out_file))
        assert sr == 24000
        assert len(audio) > 24000

    @pytest.mark.parametrize("char", ALL_POLISH_DIACRITICS)
    def test_each_individual_polish_character_nfd_and_nfc(self, char: str):
        """
        Exhaustively test each of the 18 Polish characters in both NFC and NFD form
        embedded within an utterance to ensure no combining mark causes a crash.
        """
        client = TestClient(app)
        _seed_test_f5tts_model(f"model_char_{ord(char)}")

        char_nfc = unicodedata.normalize("NFC", char)
        char_nfd = unicodedata.normalize("NFD", char)

        phrase_nfc = f"Litera {char_nfc} w języku polskim."
        phrase_nfd = f"Litera {char_nfd} w języku polskim."

        # Test NFC
        resp_nfc = client.post(
            "/api/generate/tts",
            json={"text": phrase_nfc, "model_name": f"model_char_{ord(char)}"},
        )
        assert resp_nfc.status_code == 200

        # Test NFD
        resp_nfd = client.post(
            "/api/generate/tts",
            json={"text": phrase_nfd, "model_name": f"model_char_{ord(char)}"},
        )
        assert resp_nfd.status_code == 200

        # Both records should store the exact same normalized text in DB
        rec_nfc = repository.get_generation(resp_nfc.json()["generation_id"])
        rec_nfd = repository.get_generation(resp_nfd.json()["generation_id"])
        assert rec_nfc["input_text"] == rec_nfd["input_text"]

    def test_mixed_nfc_nfd_interleaved_words(self):
        """
        Stress-test: sentence alternating between NFC and NFD on adjacent words.
        Ensures tokenizer and vocoder simulation do not fracture or corrupt.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_interleaved")

        words = ["Bądź", "wierny", "i", "idź", "wyprostowany", "wśród", "tych", "co", "na", "kolanach"]
        interleaved_parts = []
        for i, w in enumerate(words):
            if i % 2 == 0:
                interleaved_parts.append(unicodedata.normalize("NFD", w))
            else:
                interleaved_parts.append(unicodedata.normalize("NFC", w))

        interleaved_text = " ".join(interleaved_parts)
        resp = client.post("/api/generate/tts", json={"text": interleaved_text, "model_name": "model_interleaved"})
        assert resp.status_code == 200
        gen_id = resp.json()["generation_id"]

        rec = repository.get_generation(gen_id)
        assert unicodedata.is_normalized("NFC", rec["input_text"])
        assert "bądź" in rec["input_text"].lower()

    def test_special_punctuation_and_symbols_polish_context(self):
        """
        Verify common Polish typographic marks: em-dash, en-dash, Polish low-high quotes („ ”),
        ellipsis (…), currency (zł, €), percentages, digits, and mathematical signs.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_punct")

        rich_text = (
            "„Polszczyzna to skarb” — powiedział profesor. "
            "Cena książki to 49,99 zł (około 12 €), co stanowi 100% wartości… "
            "Czy to prawda? Tak, na 5+!"
        )
        resp = client.post("/api/generate/tts", json={"text": rich_text, "model_name": "model_punct"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["duration"] > 2.0


# ===========================================================================
# DIMENSION 2: WHITESPACE BOUNDARY ATTACKS (HTTP 400 ENFORCEMENT)
# ===========================================================================

class TestWhitespaceHandlingAndValidation:
    """Stress-tests whitespace-only rejections and multi-whitespace collapsing."""

    WHITESPACE_ATTACK_VECTORS = [
        "",                                         # Empty string
        " ",                                        # Single ASCII space
        "      ",                                   # Multiple ASCII spaces
        "\t",                                       # Single tab
        "\t\t\t\t",                                 # Multiple tabs
        "\n",                                       # Newline
        "\r",                                       # Carriage return
        "\r\n",                                     # CRLF
        "\r\n\t  \r\n \t ",                         # Mixed CRLF and tabs
        "\u00A0",                                   # Non-breaking space (NBSP)
        "\u00A0\u00A0\u00A0",                       # Multiple NBSP
        "\u2002",                                   # En space
        "\u2003",                                   # Em space
        "\u2009",                                   # Thin space
        "\u3000",                                   # Ideographic full-width space
        "\t\r\n \u00A0 \u2002\u2003\u2009\u3000 ",  # Extreme multi-category whitespace blend
    ]

    @pytest.mark.parametrize("ws_input", WHITESPACE_ATTACK_VECTORS)
    def test_whitespace_only_strings_rejected_http400(self, ws_input: str):
        """
        Strict requirement: all whitespace-only strings must return HTTP 400 Bad Request
        and contain 'Text cannot be empty' in the error detail.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_ws")

        resp = client.post("/api/generate/tts", json={"text": ws_input, "model_name": "model_ws"})
        assert resp.status_code == 400, f"Expected 400 for input {repr(ws_input)}, got {resp.status_code}"
        assert "Text cannot be empty" in resp.json()["detail"]

    def test_surrounding_and_multiline_whitespace_accepted(self):
        """
        Valid text wrapped in severe leading, trailing, and newline whitespace
        must succeed (HTTP 200) and be cleanly trimmed.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_ws_trim")

        messy_text = "\n\r\t \u00A0   Dzień dobry wita serdecznie Voice Sampler!   \u00A0 \r\n\t "
        resp = client.post("/api/generate/tts", json={"text": messy_text, "model_name": "model_ws_trim"})
        assert resp.status_code == 200
        data = resp.json()

        rec = repository.get_generation(data["generation_id"])
        assert rec is not None
        assert rec["input_text"] == "Dzień dobry wita serdecznie Voice Sampler!"

    def test_internal_consecutive_whitespaces_collapsed(self):
        """
        Runs of multiple tabs, spaces, and newlines inside the text must be collapsed
        to single spaces in storage, preserving natural sentence pacing.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_ws_collapse")

        spaced_text = "Warszawa\t\t\tjest\r\n\r\nstolicą\u00A0\u00A0Polski."
        resp = client.post("/api/generate/tts", json={"text": spaced_text, "model_name": "model_ws_collapse"})
        assert resp.status_code == 200
        data = resp.json()

        rec = repository.get_generation(data["generation_id"])
        assert rec is not None
        assert rec["input_text"] == "Warszawa jest stolicą Polski."


# ===========================================================================
# DIMENSION 3: TEXT LENGTH SCALING & PROPORTIONAL DURATION
# ===========================================================================

class TestTextLengthScalingAndProportionality:
    """Stress-tests synthesis duration scaling across text orders of magnitude."""

    def test_single_character_speech_duration(self):
        """A single letter or vowel should produce a safe non-zero duration (>= 1.0s)."""
        client = TestClient(app)
        _seed_test_f5tts_model("model_len")

        for single in ["A", "ą", "O", "Ś"]:
            resp = client.post("/api/generate/tts", json={"text": single, "model_name": "model_len"})
            assert resp.status_code == 200
            data = resp.json()
            assert data["duration"] >= 1.0
            assert data["duration"] <= 2.0

    def test_duration_monotonic_growth_across_lengths(self):
        """
        Duration must strictly increase monotonically with text length and syllable count:
        single char <= short phrase < medium sentence < complex paragraph < long multi-paragraph prose.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_dur_scale")

        texts = {
            "single": "A",
            "short": "Cześć, witam serdecznie.",
            "medium": "Dzisiaj w Krakowie i Warszawie świeci piękne słońce, a wiatr jest bardzo ciepły.",
            "complex": (
                "Kiedy wreszcie nadszedł wieczór, wszyscy usiedliśmy wokół ogniska, "
                "śpiewając stare polskie pieśni i wspominając młodzieńcze lata; "
                "czy pamiętasz jeszcze zapach sosnowego lasu o zmierzchu?"
            ),
            "prose": (
                "Litwo! Ojczyzno moja! ty jesteś jak zdrowie. Nazywam się Tadeusz, a to jest opowieść o dawnych czasach. "
                "Śród takich pól przed laty, nad brzegiem ruczaju, na pagórku niewielkim, we brzozowym gaju, "
                "stał dwór szlachecki, z drzewa, lecz podmurowany; świeciły się z daleka pobielane ściany, "
                "tym bielsze, że odbite od ciemnej zieleni topoli, co go bronią od wiatrów jesieni. "
                "Domy mieszkalne i stodoły były zadbane, a w ogrodzie kwitły malwy i słoneczniki. "
                "Wszyscy mieszkańcy cieszyli się spokojem i urodzajem."
            ),
        }

        durations: Dict[str, float] = {}
        for key, text in texts.items():
            resp = client.post("/api/generate/tts", json={"text": text, "model_name": "model_dur_scale"})
            assert resp.status_code == 200
            durations[key] = resp.json()["duration"]

        # Strict monotonicity assertion
        assert durations["single"] <= durations["short"]
        assert durations["short"] < durations["medium"]
        assert durations["medium"] < durations["complex"]
        assert durations["complex"] < durations["prose"]

        # Quantitative proportionality sanity: prose is at least 3x longer than short phrase
        assert durations["prose"] >= 3.0 * durations["short"]

    def test_safety_ceiling_duration_clamp(self):
        """
        Massive text input (e.g. 2,000 words) should be safely clamped to the 120.0s ceiling
        without memory exhaustion or hanging.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_clamp")

        huge_text = "To jest powtarzające się zdanie z wieloma polskimi znakami ąęćźż. " * 300
        resp = client.post("/api/generate/tts", json={"text": huge_text, "model_name": "model_clamp"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["duration"] <= 120.0


# ===========================================================================
# DIMENSION 4: AUDIO FILE VALIDITY & AUDIO ENGINEERING METRICS
# ===========================================================================

class TestAudioFileValidityAndFormatCompliance:
    """Byte-level and acoustic validation of output WAV audio files."""

    def test_riff_wave_header_byte_structure(self):
        """
        Adversarial byte inspection of generated WAV file:
        - Bytes 0..4: b'RIFF'
        - Bytes 8..12: b'WAVE'
        - Subchunk 1: b'fmt ' with format code 1 (PCM uncompressed)
        - Subchunk 1 channels: 1 (mono)
        - Subchunk 1 sample rate: 24000 Hz
        - Subchunk 1 byte rate: 48000 (24000 * 1 * 2)
        - Subchunk 1 bits per sample: 16
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_audio_spec")

        resp = client.post(
            "/api/generate/tts",
            json={"text": "Test specyfikacji formatu WAV.", "model_name": "model_audio_spec"},
        )
        assert resp.status_code == 200
        filename = resp.json()["filename"]

        settings = get_settings()
        file_path = settings.generated_dir / filename
        assert file_path.is_file()

        raw_bytes = file_path.read_bytes()
        assert len(raw_bytes) > 44

        # RIFF header
        assert raw_bytes[0:4] == b"RIFF"
        file_size_minus_8 = struct.unpack("<I", raw_bytes[4:8])[0]
        assert file_size_minus_8 == len(raw_bytes) - 8
        assert raw_bytes[8:12] == b"WAVE"

        # Locate 'fmt ' chunk
        fmt_idx = raw_bytes.find(b"fmt ")
        assert fmt_idx != -1, "Missing 'fmt ' chunk in WAV header"

        fmt_size = struct.unpack("<I", raw_bytes[fmt_idx + 4 : fmt_idx + 8])[0]
        assert fmt_size >= 16

        audio_fmt = struct.unpack("<H", raw_bytes[fmt_idx + 8 : fmt_idx + 10])[0]
        assert audio_fmt == 1, f"Expected PCM format code 1, got {audio_fmt}"

        num_channels = struct.unpack("<H", raw_bytes[fmt_idx + 10 : fmt_idx + 12])[0]
        assert num_channels == 1, f"Expected 1 channel (mono), got {num_channels}"

        sample_rate = struct.unpack("<I", raw_bytes[fmt_idx + 12 : fmt_idx + 16])[0]
        assert sample_rate == 24000, f"Expected 24000 Hz sample rate, got {sample_rate}"

        byte_rate = struct.unpack("<I", raw_bytes[fmt_idx + 16 : fmt_idx + 20])[0]
        assert byte_rate == 48000, f"Expected byte rate 48000, got {byte_rate}"

        block_align = struct.unpack("<H", raw_bytes[fmt_idx + 20 : fmt_idx + 22])[0]
        assert block_align == 2, f"Expected block align 2, got {block_align}"

        bits_per_sample = struct.unpack("<H", raw_bytes[fmt_idx + 22 : fmt_idx + 24])[0]
        assert bits_per_sample == 16, f"Expected 16-bit PCM, got {bits_per_sample}"

    def test_audio_energy_and_amplitude_metrics(self):
        """
        Verify signal integrity using soundfile and numpy:
        - Non-zero RMS energy (definitely not silence)
        - Peak normalized within safety headroom (0.50 <= peak <= 1.0)
        - Negligible DC offset (|mean| < 0.05)
        - Duration in file matches database duration within 0.05s tolerance
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_acoustic")

        resp = client.post(
            "/api/generate/tts",
            json={"text": "Sprawdzamy energię akustyczną mowy.", "model_name": "model_acoustic"},
        )
        assert resp.status_code == 200
        data = resp.json()

        settings = get_settings()
        file_path = settings.generated_dir / data["filename"]

        audio, sr = sf.read(str(file_path), dtype="float32")
        assert sr == 24000
        assert audio.ndim == 1  # Strictly mono

        # Acoustic checks
        rms = float(np.sqrt(np.mean(audio ** 2)))
        peak = float(np.max(np.abs(audio)))
        dc_offset = float(np.abs(np.mean(audio)))

        assert rms > 0.05, f"Audio energy too low (silence detected?): RMS = {rms:.4f}"
        assert 0.70 <= peak <= 1.0, f"Peak amplitude out of normal range: {peak:.4f}"
        assert dc_offset < 0.05, f"Significant DC offset detected: {dc_offset:.4f}"

        # Duration match
        file_dur = len(audio) / float(sr)
        assert abs(file_dur - data["duration"]) < 0.05

    def test_question_intonation_contour(self):
        """
        Interrogative sentences ending with '?' should exhibit a terminal pitch rise.
        Verify by comparing the instantaneous frequency of the last 200ms vs preceding segment.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_intonation")

        statement_text = "Idziemy dzisiaj do parku na spacer."
        question_text = "Idziemy dzisiaj do parku na spacer?"

        resp_stmt = client.post("/api/generate/tts", json={"text": statement_text, "model_name": "model_intonation"})
        resp_q = client.post("/api/generate/tts", json={"text": question_text, "model_name": "model_intonation"})

        assert resp_stmt.status_code == 200
        assert resp_q.status_code == 200

        settings = get_settings()
        file_stmt = settings.generated_dir / resp_stmt.json()["filename"]
        file_q = settings.generated_dir / resp_q.json()["filename"]

        audio_stmt, sr = sf.read(str(file_stmt), dtype="float32")
        audio_q, _ = sf.read(str(file_q), dtype="float32")

        # Zero-crossing rate of terminal 200ms (higher ZCR indicates higher frequency)
        tail_samples = int(0.20 * sr)
        tail_stmt = audio_stmt[-tail_samples:]
        tail_q = audio_q[-tail_samples:]

        zcr_stmt = np.sum(np.abs(np.diff(np.sign(tail_stmt)))) / len(tail_stmt)
        zcr_q = np.sum(np.abs(np.diff(np.sign(tail_q)))) / len(tail_q)

        assert zcr_q > zcr_stmt, f"Question intonation expected higher pitch than statement: q={zcr_q:.4f}, stmt={zcr_stmt:.4f}"


# ===========================================================================
# DIMENSION 5: HIGH-CONCURRENCY STRESS & THREAD SAFETY
# ===========================================================================

class TestConcurrentTTSRequestsAndThreadSafety:
    """Stress-tests multi-threaded barrier execution and race conditions."""

    def test_concurrent_generation_requests_thread_safety(self):
        """
        Fire 10 concurrent generation requests across 10 threads simultaneously using a barrier.
        Verify:
        - All 10 requests succeed with HTTP 200
        - Exactly 10 unique generation IDs generated
        - Exactly 10 unique output filenames generated
        - All 10 audio files exist on disk with valid headers
        - Exactly 10 records exist in SQLite generations table
        """
        _seed_test_f5tts_model("model_concurrent")

        num_threads = 10
        barrier = threading.Barrier(num_threads)
        results: List[Dict[str, Any]] = []
        errors: List[str] = []
        lock = threading.Lock()

        def _worker(thread_idx: int):
            t_client = TestClient(app)
            text = f"Wątek równoległy numer {thread_idx}: Zażółć gęślą jaźń i sprawdź wielowątkowość."
            barrier.wait()  # Synchronized release

            try:
                resp = t_client.post(
                    "/api/generate/tts",
                    json={"text": text, "model_name": "model_concurrent", "speed": 1.0},
                )
                if resp.status_code == 200:
                    with lock:
                        results.append(resp.json())
                else:
                    with lock:
                        errors.append(f"Thread {thread_idx} failed: {resp.status_code} {resp.text}")
            except Exception as e:
                with lock:
                    errors.append(f"Thread {thread_idx} exception: {e}")

        threads = [threading.Thread(target=_worker, args=(i,)) for i in range(num_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=15.0)

        assert len(errors) == 0, f"Concurrent execution errors: {errors}"
        assert len(results) == num_threads, f"Expected {num_threads} results, got {len(results)}"

        # Uniqueness checks
        gen_ids = [r["generation_id"] for r in results]
        filenames = [r["filename"] for r in results]
        assert len(set(gen_ids)) == num_threads, "Collision detected in generation_id"
        assert len(set(filenames)) == num_threads, "Collision detected in output_filename"

        # Verify filesystem & SQLite integrity
        settings = get_settings()
        for fn in filenames:
            file_path = settings.generated_dir / fn
            assert file_path.is_file(), f"Output file {fn} missing from disk"
            audio, sr = sf.read(str(file_path))
            assert sr == 24000
            assert len(audio) > 1000

        all_records = repository.list_generations(gen_type="tts", limit=50)
        stored_ids = {r["id"] for r in all_records}
        for gid in gen_ids:
            assert gid in stored_ids, f"Generation {gid} missing from SQLite"

    def test_concurrent_generation_and_history_queries(self):
        """
        Simultaneously execute speech generation while background threads query
        /api/generate/history to stress test SQLite reader/writer concurrency.
        """
        _seed_test_f5tts_model("model_rw_stress")

        stop_event = threading.Event()
        read_successes = []
        gen_results = []
        errors = []
        lock = threading.Lock()

        def _reader():
            client = TestClient(app)
            while not stop_event.is_set():
                try:
                    resp = client.get("/api/generate/history")
                    if resp.status_code == 200:
                        with lock:
                            read_successes.append(len(resp.json()["generations"]))
                    else:
                        with lock:
                            errors.append(f"Reader failed: {resp.status_code}")
                except Exception as ex:
                    with lock:
                        errors.append(f"Reader exception: {ex}")
                time.sleep(0.02)

        def _writer(idx: int):
            client = TestClient(app)
            text = f"Pisarz {idx}: generowanie w obecności czytelników bazy."
            try:
                resp = client.post(
                    "/api/generate/tts",
                    json={"text": text, "model_name": "model_rw_stress"},
                )
                if resp.status_code == 200:
                    with lock:
                        gen_results.append(resp.json())
                else:
                    with lock:
                        errors.append(f"Writer {idx} failed: {resp.status_code}")
            except Exception as ex:
                with lock:
                    errors.append(f"Writer {idx} exception: {ex}")

        reader_thread = threading.Thread(target=_reader)
        reader_thread.start()

        writer_threads = [threading.Thread(target=_writer, args=(i,)) for i in range(5)]
        for wt in writer_threads:
            wt.start()
        for wt in writer_threads:
            wt.join(timeout=10.0)

        stop_event.set()
        reader_thread.join(timeout=3.0)

        assert len(errors) == 0, f"Errors in concurrent read/write test: {errors}"
        assert len(gen_results) == 5
        assert len(read_successes) > 0


# ===========================================================================
# DIMENSION 6: PARAMETERS, DEFENSIVE EDGES & STREAMING / DOWNLOAD
# ===========================================================================

class TestSpeechParametersAndStreamingDownload:
    """Stress-tests speed clamping, audio range streaming, and attachment downloading."""

    def test_speed_parameter_duration_scaling(self):
        """
        Verify that speech rate multiplier directly influences duration:
        speed=0.5 (slow) vs speed=1.0 (normal) vs speed=2.0 (fast).
        Duration(0.5) > Duration(1.0) > Duration(2.0).
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_speed")

        text = "Testujemy wpływ parametru szybkości mowy na czas trwania syntezy audio."
        resp_slow = client.post("/api/generate/tts", json={"text": text, "model_name": "model_speed", "speed": 0.5})
        resp_norm = client.post("/api/generate/tts", json={"text": text, "model_name": "model_speed", "speed": 1.0})
        resp_fast = client.post("/api/generate/tts", json={"text": text, "model_name": "model_speed", "speed": 2.0})

        assert resp_slow.status_code == 200
        assert resp_norm.status_code == 200
        assert resp_fast.status_code == 200

        dur_slow = resp_slow.json()["duration"]
        dur_norm = resp_norm.json()["duration"]
        dur_fast = resp_fast.json()["duration"]

        assert dur_slow > dur_norm > dur_fast
        assert dur_slow >= 1.6 * dur_norm
        assert dur_norm >= 1.5 * dur_fast

    def test_extreme_speed_values_clamped_safely(self):
        """
        Negative, near-zero, and extreme speeds should be safely clamped to [0.5, 2.0]
        without division by zero or crashing.
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_speed_clamp")

        text = "Szybkość skrajna."
        # Ultra fast (100.0) -> clamped to 2.0
        resp_high = client.post("/api/generate/tts", json={"text": text, "model_name": "model_speed_clamp", "speed": 100.0})
        assert resp_high.status_code == 200

        # Ultra slow (0.001) -> clamped to 0.5
        resp_low = client.post("/api/generate/tts", json={"text": text, "model_name": "model_speed_clamp", "speed": 0.001})
        assert resp_low.status_code == 200

        # Negative speed -> clamped to 0.5
        resp_neg = client.post("/api/generate/tts", json={"text": text, "model_name": "model_speed_clamp", "speed": -5.0})
        assert resp_neg.status_code == 200

    def test_reference_audio_sample_interaction(self):
        """
        Verify:
        - Valid reference audio sample is used and pitch is extracted
        - Omitted reference audio falls back gracefully to synthetic anchor
        - Non-existent reference sample UUID returns HTTP 404
        """
        client = TestClient(app)
        _seed_test_audio_sample("ref_sample_valid", freq_hz=220.0)
        _seed_test_f5tts_model("model_ref", "ref_sample_valid")

        # 1. With valid reference sample
        resp_valid = client.post(
            "/api/generate/tts",
            json={"text": "Synteza z referencją.", "model_name": "model_ref", "reference_sample_id": "ref_sample_valid"},
        )
        assert resp_valid.status_code == 200

        # 2. Without reference sample (fallback)
        resp_none = client.post(
            "/api/generate/tts",
            json={"text": "Synteza bez referencji.", "model_name": "model_ref"},
        )
        assert resp_none.status_code == 200

        # 3. Non-existent reference sample -> HTTP 404
        resp_ghost = client.post(
            "/api/generate/tts",
            json={"text": "Synteza z duchem.", "model_name": "model_ref", "reference_sample_id": "ghost_sample_uuid"},
        )
        assert resp_ghost.status_code == 404
        assert "not found" in resp_ghost.json()["detail"].lower()

    def test_missing_and_empty_model_name_errors(self):
        """Verify strict error handling on missing or nonexistent model names."""
        client = TestClient(app)

        # Empty model name -> HTTP 400
        resp_empty = client.post("/api/generate/tts", json={"text": "Tekst", "model_name": "   "})
        assert resp_empty.status_code == 400

        # Non-existent model name -> HTTP 404
        resp_missing = client.post("/api/generate/tts", json={"text": "Tekst", "model_name": "ghost_model_12345"})
        assert resp_missing.status_code == 404

    def test_streaming_and_download_end_to_end(self):
        """
        Full lifecycle verification:
        1. Generate speech via POST /api/generate/tts
        2. Stream header bytes (0-43) via GET /api/audio/generated/{filename} (HTTP 206)
        3. Download full file via GET /api/audio/download/{filename} with Content-Disposition
        4. Traversal attack on generated filename returns HTTP 400/404
        """
        client = TestClient(app)
        _seed_test_f5tts_model("model_stream_dl")

        gen_resp = client.post(
            "/api/generate/tts",
            json={"text": "Test pobierania i strumieniowania nagłówka.", "model_name": "model_stream_dl"},
        )
        assert gen_resp.status_code == 200
        filename = gen_resp.json()["filename"]

        # Stream exact 44-byte WAV header
        stream_resp = client.get(f"/api/audio/generated/{filename}", headers={"Range": "bytes=0-43"})
        assert stream_resp.status_code == 206
        assert len(stream_resp.content) == 44
        assert stream_resp.content[:4] == b"RIFF"
        assert stream_resp.content[8:12] == b"WAVE"

        # Download attachment
        dl_resp = client.get(f"/api/audio/download/{filename}")
        assert dl_resp.status_code == 200
        assert "attachment" in dl_resp.headers.get("Content-Disposition", "")
        assert filename in dl_resp.headers.get("Content-Disposition", "")
        assert len(dl_resp.content) > 1000

        # Path traversal defense
        bad_resp = client.get("/api/audio/generated/../../windows/system32/cmd.exe")
        assert bad_resp.status_code in (400, 404)
