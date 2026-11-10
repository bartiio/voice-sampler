# Voice Sampler: End-to-End Test Infrastructure Specification

**Document Version:** 1.0.0  
**Author:** test_writer_e2e_1 (E2E Testing Track Lead)  
**Date:** 2026-11-06  
**Status:** Active & Authoritative  
**Test Suite Path:** `tests/`  
**Runner:** `tests/run_e2e_tests.py`

---

## 1. Test Philosophy & Principles

Voice Sampler is a precision audio ML desktop application combining an Electron/React frontend, a Python FastAPI backend, neural speech synthesis (F5-TTS), and neural voice conversion (RVC v2), with primary optimization for the Polish language.

The testing strategy is anchored on five inviolable principles:

1. **Requirement-Driven & Opaque-Box**:
   Tests treat the application and backend through its public interfaces (REST API, WebSocket telemetry, audio streams, and persisted artifacts). Tests do not rely on private implementation details, ensuring long-term resilience against internal refactoring.

2. **Progressive Testability & Hardware Independence**:
   Because production ML training requires an NVIDIA CUDA GPU while development hosts and CI environments may feature AMD GPUs, Intel integrated graphics, or CPU-only containers, the test suite supports both live CUDA execution and simulated/mock execution. Tests verify the contract, telemetry, and artifact pipelines identically in both modes.

3. **Multi-Tiered Coverage**:
   Tests are partitioned into 4 distinct verification tiers:
   - **Tier 1 (Feature Coverage)**: >=5 isolated tests per functional capability ensuring baseline correctness.
   - **Tier 2 (Boundary & Corner Cases)**: Stress testing limits, empty inputs, non-CUDA fallbacks, extreme hyperparameters, and malformed audio.
   - **Tier 3 (Cross-Feature Combinations)**: Multi-step lifecycle flows (Upload -> Transcribe -> Edit -> Train -> Inference -> Download).
   - **Tier 4 (Real-World Scenarios)**: Authentic Polish voice cloning workflows (diacritics, tongue-twisters, singing voice conversion, dialogue synthesis).

4. **Self-Contained & Deterministic**:
   Every test creates its own fixtures (using deterministic synthetic audio generators), executes independently without ordering dependencies, and cleans up all temporary state.

5. **Actionable Telemetry & Strict Exit Codes**:
   The test runner produces unambiguous visual reports (per-tier and aggregate metrics, timing, error stack traces) and returns exit code 0 if and only if 100% of non-skipped tests pass.

---

## 2. Feature Inventory Mapped to Test Tiers

The following matrix maps every requirement from `ORIGINAL_REQUEST.md` and feature from `PROJECT.md` to its corresponding test tiers and coverage count:

| Feature ID | Feature Name | Tier 1 (Features) | Tier 2 (Boundary) | Tier 3 (Combos) | Tier 4 (Scenarios) | Minimum Target Tests |
|---|---|---|---|---|---|---|
| **F01** | Backend Health & Lifecycle | 5 tests | 3 tests | 2 tests | 1 test | >= 11 tests |
| **F02** | Settings & GPU Hardware Detection | 5 tests | 4 tests | 1 test | 1 test | >= 11 tests |
| **F03** | Audio Recording Ingestion | 5 tests | 4 tests | 3 tests | 2 tests | >= 14 tests |
| **F04** | Multi-Format File Upload (WAV, MP3, FLAC) | 5 tests | 5 tests | 3 tests | 2 tests | >= 15 tests |
| **F05** | Sample CRUD Operations | 5 tests | 3 tests | 2 tests | 1 test | >= 11 tests |
| **F06** | Transcript Editing & Unicode Persistence | 5 tests | 4 tests | 3 tests | 3 tests | >= 15 tests |
| **F07** | Byte-Range Audio Streaming | 5 tests | 4 tests | 2 tests | 1 test | >= 12 tests |
| **F08** | Whisper Polish Transcription Engine | 5 tests | 4 tests | 3 tests | 3 tests | >= 15 tests |
| **F09** | F5-TTS Dataset Preparation & Config | 5 tests | 4 tests | 3 tests | 2 tests | >= 14 tests |
| **F10** | F5-TTS Model Training & Telemetry | 5 tests | 4 tests | 3 tests | 2 tests | >= 14 tests |
| **F11** | RVC v2 Dataset Prep & Pipeline | 5 tests | 4 tests | 3 tests | 2 tests | >= 14 tests |
| **F12** | RVC v2 Model Training & Telemetry | 5 tests | 4 tests | 3 tests | 2 tests | >= 14 tests |
| **F13** | Model Registry & Checkpoint Query | 5 tests | 3 tests | 3 tests | 1 test | >= 12 tests |
| **F14** | F5-TTS Speech Synthesis (TTS) | 5 tests | 4 tests | 3 tests | 3 tests | >= 15 tests |
| **F15** | RVC v2 Voice Conversion (STS) | 5 tests | 5 tests | 3 tests | 3 tests | >= 16 tests |
| **F16** | Generated Audio Playback & Download | 5 tests | 3 tests | 2 tests | 2 tests | >= 12 tests |
| **TOTAL** | **Full Application System** | **>= 80** | **>= 62** | **>= 39** | **>= 29** | **>= 210** |

---

## 3. Test Architecture & Directory Layout

The E2E test suite is located entirely under `tests/` and structured as follows:

```
c:\Users\bartoszk\Desktop\voice sampler\
├── TEST_INFRA.md                  # This test infrastructure specification
├── TEST_READY.md                  # Readiness checklist and execution instructions
└── tests/
    ├── __init__.py
    ├── run_e2e_tests.py           # Unified CLI runner executing all tiers
    └── e2e/
        ├── __init__.py
        ├── harness.py             # Test harness, API/WS clients, synthetic audio generator
        ├── test_tier1_features.py # Tier 1: Feature coverage (>=5 tests per feature)
        ├── test_tier2_boundary.py # Tier 2: Boundary, limits, corner cases, error handling
        ├── test_tier3_combos.py   # Tier 3: Cross-feature end-to-end combinations
        └── test_tier4_scenarios.py# Tier 4: Real-world Polish voice cloning scenarios
```

### 3.1 Test Harness (`tests/e2e/harness.py`)
Provides reusable testing infrastructure:
- **`VoiceSamplerTestClient`**: High-level HTTP client wrapping all REST endpoints (`health`, `settings`, `samples`, `train`, `models`, `generate`, `audio`). Supports custom headers, multipart file uploads, and byte-range streaming.
- **`WebSocketCollector`**: Async/threaded WebSocket listener that subscribes to `/ws/training` and `/ws/notifications`, collecting progress frames, metrics, and terminal completion/error events with configurable timeout and assertion helpers.
- **`SyntheticAudioGenerator`**: Generates valid in-memory audio buffers:
  * Pure sine waves (440Hz, 1kHz, 24kHz/40kHz/44.1kHz sample rates, 16-bit PCM).
  * Multi-harmonic synthetic speech dummy WAVs simulating vocal formants.
  * Silence buffers and extreme short/long audio buffers.
  * Multi-format mock files (valid WAV headers, MP3 framing dummy, FLAC signature).
  * Corrupted/truncated audio files for boundary and error testing.
- **`EmbeddedMockServer`**: An ASGI/FastAPI mock server conforming exactly to `PROJECT.md` contracts. If no external backend is active on the target port, the test harness boots this server automatically, allowing immediate offline verification.

### 3.2 Tier 1: Feature Coverage (`tests/e2e/test_tier1_features.py`)
Contains >=5 distinct tests for each of the 10 feature groups:
1. Health & Lifecycle
2. Settings & GPU Detection
3. Sample CRUD
4. Audio Ingestion & Byte-Range Streaming
5. Transcript Editing & Unicode
6. Whisper Polish Transcription
7. F5-TTS Preparation & Training
8. RVC v2 Preparation & Training
9. TTS Voice Synthesis (F5-TTS)
10. STS Voice Conversion (RVC v2)

### 3.3 Tier 2: Boundary & Corner Cases (`tests/e2e/test_tier2_boundary.py`)
Systematically probes edge cases and error bounds:
- Empty payloads (empty text, 0-byte audio, blank filenames).
- Non-existent resource IDs (sample IDs, model IDs, generation IDs).
- Extreme pitch shift values: -24, -12, 0, +12, +24, and out-of-range bounds (>24 or <-24).
- Extreme index rates: 0.0 (weights only), 1.0 (pure feature retrieval), <0.0 or >1.0.
- Non-CUDA host simulation handling: ensures backend reports `cuda_available: false`, provides explanatory diagnostics, and succeeds via simulation fallback without crash.
- Byte-range boundary tests (negative ranges, out-of-bounds byte offsets, multi-range requests).
- Concurrency and cancellation: canceling non-existent or completed training runs.

### 3.4 Tier 3: Cross-Feature Combinations (`tests/e2e/test_tier3_combos.py`)
Validates complete user journeys combining multiple features:
- **Combo A (Full F5-TTS Pipeline)**: Upload WAV -> Whisper transcribes in Polish -> Edit transcript with diacritics -> Start F5-TTS training -> Stream WebSocket progress -> Verify model registered -> Synthesize TTS speech -> Download generated audio.
- **Combo B (Full RVC v2 Pipeline)**: Record microphone audio -> Auto-transcribe -> Start RVC v2 training -> Stream WebSocket telemetry to completion -> Verify `.pth` and `.index` registered -> Run STS conversion with pitch shift -> Stream output audio.
- **Combo C (Multi-Sample Batch Training)**: Upload multiple distinct audio samples -> Transcribe all -> Select subset of sample IDs for training -> Train model -> Verify dataset preparation includes only selected samples.
- **Combo D (Sample Deletion Cascading)**: Upload sample -> Transcribe -> Delete sample -> Attempt training with deleted ID -> Verify clean 400/404 validation error without server crash.
- **Combo E (Active Training Cancellation)**: Start training -> Receive progress frames -> Send `POST /api/train/stop` -> Receive cancellation event -> Verify status is stopped and partial models are cleaned up.

### 3.5 Tier 4: Real-World Polish Voice Cloning Scenarios (`tests/e2e/test_tier4_scenarios.py`)
Simulates authentic target-use workflows:
- **Scenario 1 (Polish Diacritics & Tongue-Twisters)**:
  Input: "W Szczebrzeszynie chrząszcz brzmi w trzcinie, a chrabąszcz brzmi w trzcinie."
  Tests vocabulary generation for `ą, ć, ę, ł, ń, ó, ś, ź, ż`, tokenization accuracy, and phonetic preservation.
- **Scenario 2 (Singing Voice Conversion & Melodic Pitch Shift)**:
  Input: Polish vocal melody with semitone shifts (+3, -4 semitones) and high index rate (0.85) to preserve singer timbre.
- **Scenario 3 (Mixed Language & Technical Polish)**:
  Input: "Nasz nowy model AI generuje ultra-realistyczny speech synthesis w czasie rzeczywistym."
  Validates hybrid Polish-English vocabulary and pronunciation stability.
- **Scenario 4 (Short vs Long Utterances & Punctuation)**:
  Validates synthesis of short interjections ("Tak!", "Nie?") versus paragraph-length Polish prose with complex punctuation.
- **Scenario 5 (Batch Dialogue Voiceover Workflow)**:
  Generates a sequential multi-sentence dialogue between two cloned voice models, verifying unique generation IDs and audio assets.

---

## 4. Coverage & Quality Thresholds

| Metric | Target Threshold | Strict / Warning |
|---|---|---|
| **Tier 1 Feature Pass Rate** | **100%** | Strict (Exit failure if < 100%) |
| **Tier 2 Boundary Pass Rate** | **100%** | Strict (Exit failure if < 100%) |
| **Tier 3 Combination Pass Rate** | **100%** | Strict (Exit failure if < 100%) |
| **Tier 4 Scenario Pass Rate** | **100%** | Strict (Exit failure if < 100%) |
| **Minimum Tests Per Feature** | **>= 5 tests** | Strict |
| **Total Test Count** | **>= 150 tests** | Strict |
| **Zero Regressions** | **0 failures** | Strict |

---

## 5. Verification Commands

Run all tiers:
```powershell
python tests/run_e2e_tests.py --all
```

Run specific tier:
```powershell
python tests/run_e2e_tests.py --tier 1
python tests/run_e2e_tests.py --tier 2
python tests/run_e2e_tests.py --tier 3
python tests/run_e2e_tests.py --tier 4
```

Targeting a live server:
```powershell
python tests/run_e2e_tests.py --all --url http://127.0.0.1:8000
```
