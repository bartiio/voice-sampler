# Voice Sampler 🎙️🇵🇱

> **Native Desktop Application for Custom Polish Voice Cloning**  
> Collect microphone recordings with 60fps waveform visualization, upload audio files, automatically transcribe Polish speech with Whisper, fine-tune **F5-TTS** (Text-to-Speech) and **RVC v2** (Speech-to-Speech) neural models with real-time WebSocket telemetry, and synthesize cloned voices with hardware GPU acceleration.

---

[![Electron](https://img.shields.io/badge/Electron-34.2.0-47848F?logo=electron&logoColor=white)](https://www.electronjs.org/)
[![React](https://img.shields.io/badge/React-19.0.0-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![TypeScript](https://img.shields.io/badge/TypeScript-5.7.3-3178C6?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![Vite](https://img.shields.io/badge/Vite-6.x-646CFF?logo=vite&logoColor=white)](https://vitejs.dev/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Python](https://img.shields.io/badge/Python-3.10%20|%203.14-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/E2E%20Tests-106%20Passed%20(100%25)-brightgreen)](#testing--quality-assurance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Table of Contents

1. [Overview](#overview)
2. [Key Features](#key-features)
3. [System Architecture](#system-architecture)
4. [Prerequisites](#prerequisites)
5. [Setup & Installation](#setup--installation)
6. [Running the Application](#running-the-application)
   - [Unified Desktop Launch (Recommended)](#1-unified-desktop-launch-recommended)
   - [Standalone Backend Server](#2-standalone-backend-server)
   - [Standalone Frontend Renderer](#3-standalone-frontend-renderer)
   - [Production Build](#4-production-build)
7. [Feature Guide (The 4 Views)](#feature-guide-the-4-views)
   - [1. Samples View](#1-samples-view)
   - [2. Training View](#2-training-view)
   - [3. Generate View](#3-generate-view)
   - [4. Settings View](#4-settings-view)
8. [Dual-Execution Engine (CUDA vs. Simulation)](#dual-execution-engine-cuda-vs-simulation)
9. [Testing & Quality Assurance](#testing--quality-assurance)
   - [Automated E2E Test Suite (Tiers 1–4)](#automated-e2e-test-suite-tiers-14)
   - [Backend Unit Tests](#backend-unit-tests)
   - [TypeScript & Frontend Verification](#typescript--frontend-verification)
   - [Adversarial & Regression Test Suites](#adversarial--regression-test-suites)
10. [Project Directory Layout](#project-directory-layout)
11. [License](#license)

---

## Overview

**Voice Sampler** is a full-stack, native desktop application engineered specifically for custom **Polish voice cloning**. It provides an end-to-end workflow from vocal sample capture to high-quality audio synthesis:

1. **Audio Acquisition**: Record pristine vocal samples via microphone with a real-time 60fps canvas visualizer, or drag-and-drop existing audio files (WAV, MP3, FLAC).
2. **Automatic Polish Whisper Transcription**: Background speech recognition powered by `faster-whisper` (CTranslate2) with Polish language prioritization, paired with an inline transcript editor featuring dedicated Polish diacritic helpers (`ą, ć, ę, ł, ń, ó, ś, ź, ż`).
3. **Dual Model Training**:
   - **F5-TTS (Text-to-Speech)**: Flow-Matching Diffusion Transformer fine-tuned on custom voice datasets to generate natural Polish speech from arbitrary text.
   - **RVC v2 (Speech-to-Speech)**: Retrieval-based Voice Conversion (NSF-VITS Generator with RMVPE pitch extraction and HuBERT features) to convert any vocal recording into the target voice.
4. **Live WebSocket Telemetry**: Stream real-time training progress, loss values, steps, epochs, and logs into interactive SVG charts.
5. **Voice Generation & Playback**: Synthesize speech using trained F5-TTS models (with Polish pangram presets) or convert audio via RVC v2 (with semitone pitch shift and index retrieval controls), complete with RFC 7233 byte-range streaming and WAV file download.
6. **Hardware Adaptability**: Automatic NVIDIA CUDA GPU detection with fallback to a high-fidelity parametric simulation engine on non-CUDA machines (e.g. AMD/Intel iGPUs, CPU-only systems).

---

## Key Features

- **Microphone Waveform Visualizer**: 60fps Canvas audio visualizer powered by Web Audio API `AnalyserNode`.
- **Multi-Format Ingestion**: Ingests WAV, MP3, and FLAC audio files using PyAV and SoundFile, normalizing to 16-bit PCM WAV.
- **Polish Language Priority**: Whisper transcription configured for Polish phonetics, plus one-click Polish diacritic toolbar for fast transcript refinement.
- **F5-TTS Dataset Preparation**: Resamples to 24kHz mono WAV and generates pipe-delimited `metadata.csv` datasets.
- **RVC v2 Dataset Preparation**: Resamples to 40kHz, runs RMVPE fundamental frequency ($F_0$) extraction, extracts 768-dimensional HuBERT features, and builds Faiss retrieval indexes.
- **WebSocket Streaming**: Bidirectional real-time progress updates for both background transcription notifications (`/ws/notifications`) and model training telemetry (`/ws/training`).
- **Resilient Child Process Supervisor**: Electron automatically manages the FastAPI backend lifespan, detects free ports, performs HTTP health probing, and guarantees zero orphaned processes on exit via `tree-kill`.
- **Dual-Execution ML Pipeline**: Seamlessly runs authentic CUDA PyTorch execution on NVIDIA hardware or high-fidelity parametric DSP simulation (formant vocal synthesis, phase-vocoder pitch transposition) on non-CUDA hosts.

---

## System Architecture

```
+-----------------------------------------------------------------------------------------+
|                                 Electron Desktop Shell                                  |
|  +-----------------------------------------------------------------------------------+  |
|  |                    React 19 + TypeScript 5.7 + Vite 6 Frontend                    |  |
|  |                                                                                   |  |
|  |   [Samples View]           [Training View]        [Generate View]    [Settings]   |  |
|  |   - 60fps Waveform Mic     - F5-TTS Engine        - TTS (F5-TTS)     - GPU Info   |  |
|  |   - Drag & Drop Upload     - RVC v2 Engine        - STS (RVC v2)     - VRAM       |  |
|  |   - Byte-Range Audio       - Hyperparameters      - Pitch Shift      - Backend    |  |
|  |   - Polish Diacritics Bar  - Live WS Loss Chart   - Diacritics Bar     Health     |  |
|  |   - Whisper Transcripts    - Model Checkpoints    - WAV Downloader   - Diagnostics|  |
|  +-----------------------------------------------------------------------------------+  |
|          │ REST Requests (Axios / Fetch)       ▲ WebSocket (Telemetry & Events)         |
|          ▼                                     │                                        |
+----------┼─────────────────────────────────────┼────────────────────────────────--------+
           │ Child Process Supervisor            │
           ▼ (Managed by Electron Main)          │
+------------------------------------------------┴----------------------------------------+
|                                 FastAPI Python Backend                                  |
|                                                                                         |
|   REST Endpoints:                                                                       |
|     /api/health            /api/settings/gpu      /api/samples (CRUD & Upload)          |
|     /api/train/f5-tts      /api/train/rvc         /api/train/stop                       |
|     /api/generate/tts      /api/generate/sts      /api/audio/download/{id}              |
|                                                                                         |
|   WebSockets:                                                                           |
|     /ws/training (live loss telemetry)    /ws/notifications (transcription events)      |
|                                                                                         |
|   +-----------------------+  +------------------------+  +---------------------------+  |
|   |    Whisper Service    |  |   Training Pipelines   |  |     Inference Engines     |  |
|   |  - faster-whisper     |  |  - F5-TTS (Flow-Match) |  |  - F5-TTS Speech Synth    |  |
|   |  - Polish priority    |  |  - RVC v2 (NSF-VITS)   |  |  - RVC Voice Conversion   |  |
|   |  - Async worker queue |  |  - Dual-Execution Mode |  |  - Audio Streamer (206)   |  |
|   +-----------------------+  +------------------------+  +---------------------------+  |
|                                           │                                             |
|   +---------------------------------------┴-----------------------------------------+   |
|   |  Storage Layer: SQLite (data/db/voice_sampler.db, WAL) + Filesystem (data/)    |   |
|   +---------------------------------------------------------------------------------+   |
+-----------------------------------------------------------------------------------------+
```

### Component Breakdown

| Layer | Technologies | Responsibilities |
|---|---|---|
| **Frontend UI** | React 19, TypeScript 5.7, Vite 6, Tailwind CSS 3.4, Lucide Icons | Responsive single-page application with sidebar navigation, real-time waveform visualizer, dynamic SVG loss charts, and diacritic helpers. |
| **Desktop Shell** | Electron 34, Node.js 22/24 | Window management, security sandboxing, local filesystem access, and backend process lifecycle supervision. |
| **Process Supervisor** | Node `net` port scanner, `child_process.spawn`, `tree-kill` | Finds free port, spawns FastAPI backend, polls `/api/health` until ready, and executes clean graceful shutdown handshake. |
| **Backend API** | Python 3.10+, FastAPI 0.115+, Uvicorn 0.30+, Starlette, Pydantic v2 | High-performance asynchronous REST API, WebSocket pub/sub broadcasters, audio normalization, and background worker threads. |
| **Database** | SQLite 3 with Write-Ahead Logging (WAL) | ACID storage for audio samples, metadata, transcript history, training runs, and registered model checkpoints. |
| **ML & Audio** | faster-whisper, PyTorch, Torchaudio, SciPy, SoundFile, PyAV | Deep neural execution for Whisper, F5-TTS, and RVC v2 with automatic fallback to parametric simulation engine. |

---

## Prerequisites

Ensure your system meets the following software requirements:

- **Node.js**: `18.0.0+` (Node.js 24 LTS recommended and verified)
- **Python**: `3.10+` (Python 3.10 through 3.14 verified)
- **Operating System**: Windows 10/11, macOS, or Linux (Windows 11 verified)
- **Audio Device**: Working microphone (for recording) and speakers/headphones (for playback)
- **GPU (Optional)**: NVIDIA GPU with CUDA 11.8+ or 12.x for neural hardware acceleration. *(Non-CUDA systems automatically run in high-fidelity simulation mode with full functionality.)*

---

## Setup & Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/voice-sampler.git
cd "voice sampler"
```

### 2. Python Backend Setup

Create and activate a Python virtual environment, then install the dependencies:

**On Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

*(Note: PyTorch with CUDA can be installed via `pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121` if you have an NVIDIA GPU.)*

### 3. Frontend & Electron Setup

Install Node dependencies in the project root:

```bash
npm install
```

---

## Running the Application

### 1. Unified Desktop Launch (Recommended)

To run the complete native application with Electron, React frontend, and auto-spawned FastAPI backend:

```bash
npm run dev
```

**What happens behind the scenes:**
1. Starts the Vite dev server on `http://localhost:5173`.
2. Compiles Electron main and preload TypeScript scripts (`tsconfig.node.json`).
3. Launches the Electron shell.
4. Electron scans for an available TCP port (starting at `8000`).
5. Electron spawns `run_server.py --port <port> --parent-pid <pid> --dev` as a managed child process.
6. Electron polls `GET /api/health` until the backend reports ready.
7. The desktop application window opens, connected to both REST and WebSocket endpoints.
8. On window close, Electron gracefully stops the backend process with `tree-kill` cleanup.

---

### 2. Standalone Backend Server

You can run the FastAPI backend independently for API testing, development, or headless mode:

```bash
python run_server.py --port 8000 --dev
```

#### CLI Options for `run_server.py`

| Flag | Description | Default | Environment Variable |
|---|---|---|---|
| `--port`, `-p` | Port to bind the HTTP and WebSocket server | `8000` | `VOICE_SAMPLER_PORT` |
| `--host` | Host address to bind to | `127.0.0.1` | `VOICE_SAMPLER_HOST` |
| `--dev` | Enables development mode (OpenAPI docs at `/docs`, debug logging) | `False` | `VOICE_SAMPLER_DEV_MODE` |
| `--reload` | Enables Uvicorn hot-reload on source code changes | `False` | — |
| `--log-level` | Server log level (`critical`, `error`, `warning`, `info`, `debug`, `trace`) | `info` | `VOICE_SAMPLER_LOG_LEVEL` |
| `--parent-pid` | Monitors parent process PID and terminates backend automatically if parent exits | `None` | — |

Interactive Swagger documentation is available in dev mode at:  
👉 **http://127.0.0.1:8000/docs**

---

### 3. Standalone Frontend Renderer

To run the React web frontend in a standard browser (e.g., Chrome, Edge) against a running backend:

```bash
npx vite
```

Open your browser at **http://localhost:5173**.

---

### 4. Production Build

To compile and verify production bundles for both frontend and Electron:

```bash
# 1. Typecheck TypeScript across the codebase
npm run typecheck

# 2. Build Vite production bundle and Electron scripts
npm run build

# 3. Launch the compiled Electron application
npm start
# or:
npx electron .
```

---

## Feature Guide (The 4 Views)

The desktop application is divided into four intuitive sections accessible via the sidebar navigation:

```
[🎙️ Samples]   [🧠 Training]   [✨ Generate]   [⚙️ Settings]
```

---

### 1. Samples View

The foundation of voice cloning: capture, upload, inspect, and transcribe voice recordings.

- **Real-Time Microphone Recording**:
  - Click **Start Recording** to capture vocal audio via the browser `MediaRecorder` API.
  - A real-time **60fps Canvas Waveform Visualizer** draws the audio amplitude envelope with peak indicators.
  - Controls to Pause, Resume, and Stop recording with a live timer.
  - Recorded audio is automatically normalized to 16-bit 44.1kHz mono PCM WAV.
- **Drag-and-Drop File Upload**:
  - Dropzone supporting `.wav`, `.mp3`, and `.flac` audio formats.
  - Multi-file batch processing with individual upload progress bars.
  - PyAV and SoundFile decode compressed formats and normalize headers.
- **Automatic Polish Whisper Transcription**:
  - As soon as a sample is added, an asynchronous background task transcribes the audio using `faster-whisper`.
  - Configured with Polish language priority (`language="pl"`).
  - Status updates (`pending` → `transcribing` → `transcribed`) stream to the UI in real time via `/ws/notifications`.
- **Inline Transcript Editor & Polish Diacritics**:
  - Click any transcript to edit it directly.
  - Includes a dedicated **Polish Diacritics Toolbar**:
    - Lowercase: `ą`, `ć`, `ę`, `ł`, `ń`, `ó`, `ś`, `ź`, `ż`
    - Uppercase: `Ą`, `Ć`, `Ę`, `Ł`, `Ń`, `Ó`, `Ś`, `Ź`, `Ż`
  - Changes persist immediately to SQLite upon saving.
- **RFC 7233 Byte-Range Audio Streaming**:
  - In-app playback scrubber supporting partial content requests (`HTTP 206 Partial Content`) for instant seeking.

---

### 2. Training View

Configure, execute, and monitor deep learning fine-tuning for custom voice cloning models.

- **Dual Engine Selection**:
  - **F5-TTS (Flow-Matching Diffusion Transformer)**: Ideal for Text-to-Speech generation. Prepares pipe-delimited `metadata.csv` datasets and fine-tunes the base model.
  - **RVC v2 (Retrieval-based Voice Conversion)**: Ideal for Speech-to-Speech voice conversion. Runs RMVPE pitch extraction, HuBERT 768-dim embeddings, and builds Faiss index files.
- **Configurable Hyperparameters**:
  - **Model Name**: Unique identifier for the output checkpoint.
  - **Epochs**: Number of training iterations (e.g. 10–50 for F5-TTS, 20–100 for RVC).
  - **Batch Size**: Mini-batch size (1–32) scaled to available VRAM.
  - **Learning Rate**: Optimization step size for F5-TTS (default `0.0001`).
  - **Target Sample Rate**: Audio fidelity for RVC v2 (`32000`, `40000`, or `48000` Hz; default `40000`).
- **Sample Selection Checklist**:
  - Select all or cherry-pick specific transcribed samples from your library to include in the training dataset.
- **Live WebSocket Telemetry (`/ws/training`)**:
  - **Dynamic SVG Loss Curve Chart**: Visualizes training loss progression in real time with auto-scaling coordinate axes.
  - **Progress Metrics**: Real-time display of current epoch, step, percentage, loss, and elapsed time.
  - **Status Console**: Scrolling log terminal with timestamped events for epoch completions, checkpoint writes, and errors.
  - **Graceful Cancellation**: Stop training at any time with cooperative checkpoint preservation.
- **Model Registry & Checkpoint Management**:
  - Inspect all trained models saved in `data/models/`.
  - View model metadata: engine type, final loss, training sample count, and creation timestamp.
  - Delete obsolete models with confirmation.

---

### 3. Generate View

Synthesize new vocal audio using your trained custom voice models.

- **Text-to-Speech (TTS) Mode**:
  - **Text Input**: Enter any Polish phrase or text passage.
  - **Polish Pangram Presets**: One-click quick test buttons containing every Polish diacritic:
    - *"Zażółć gęślą jaźń."*
    - *"Mężny bądź, chroń pułk twój i sześć flag."*
    - *"Pchnąć w tę łódź jeża lub ośm skrzyń fig."*
  - **Diacritic Toolbar**: Quick insert buttons for `ą, ć, ę, ł, ń, ó, ś, ź, ż` and uppercase variants.
  - **Model Selector**: Select any trained F5-TTS model.
  - **Reference Audio**: Choose a reference sample from your library to condition vocal timbre and prosody.
  - **Speed Slider**: Adjust speech synthesis pacing from `0.5x` to `2.0x`.
- **Speech-to-Speech (STS) Mode**:
  - **Input Audio**: Choose an existing recording from your sample library.
  - **Model Selector**: Select any trained RVC v2 model.
  - **Pitch Shift Slider**: Transpose pitch across **-24 to +24 semitones** (`0` = original pitch, `+12` = one octave up, `-12` = one octave down).
  - **Index Rate Slider**: Control HuBERT feature retrieval blending from `0.0` to `1.0` (default `0.75`).
- **Playback & Export**:
  - Audio player with instant streaming playback.
  - One-click **Download WAV** button (`Content-Disposition: attachment`).
  - Generation History table recording all outputs with duration, parameters, and timestamps.

---

### 4. Settings View

Hardware diagnostics, GPU detection, and system health status.

- **Hardware Acceleration Card**:
  - **Detected GPU Device**: NVIDIA GPU name (or AMD / Intel / generic WMI device).
  - **CUDA Availability**: Real-time status badge (`CUDA ACTIVE` vs. `SIMULATION MODE`).
  - **CUDA Runtime Version**: Detected CUDA runtime (e.g. `12.1`, `11.8`, or `None`).
  - **VRAM Total & Allocated**: Total video memory and actively allocated VRAM.
  - **PyTorch Version**: Installed PyTorch runtime string.
- **Prominent Amber NO-CUDA Warning Banner**:
  - Whenever CUDA is not available, a clear amber warning banner is displayed explaining that the application is operating in high-fidelity simulation mode.
- **FastAPI Backend Health Card**:
  - Backend status indicator (`ONLINE` vs. `DISCONNECTED`).
  - API endpoint URL (`http://127.0.0.1:8000`).
  - Software version (`v0.1.0`).
  - Live roundtrip ping latency (ms).
  - Server uptime clock (`Xh Ym Zs`).
  - **Refresh Diagnostics** button to re-scan hardware on demand.

---

## Dual-Execution Engine (CUDA vs. Simulation)

Voice Sampler is engineered to deliver a seamless developer and user experience regardless of the underlying host hardware:

```
                      +-----------------------------+
                      | Hardware Detection at Boot |
                      +--------------+--------------+
                                     │
                    Is NVIDIA CUDA GPU available?
                                     │
                   ┌─────────────────┴─────────────────┐
                   ▼ YES                               ▼ NO
      +-------------------------+         +-------------------------+
      |     CUDA GPU Mode       |         | High-Fidelity Simulation|
      +-------------------------+         +-------------------------+
      | • Real PyTorch Tensors  |         | • Exponential Loss Dec. |
      | • F5-TTS Flow-Matching  |         | • Formant Vocal Synth   |
      | • RVC v2 NSF-VITS       |         | • Phase-Vocoder Pitch   |
      | • HuBERT Embeddings     |         | • Authentic Checkpoints |
      | • Real Neural Weights   |         | • 100% E2E Test Passing |
      +-------------------------+         +-------------------------+
```

### 1. CUDA GPU Mode
- **Requirements**: NVIDIA GPU (Turing, Ampere, Ada Lovelace, or Hopper architecture), NVIDIA display drivers, CUDA runtime, and PyTorch CUDA build.
- **Training**: Fine-tunes actual neural weights using `accelerate` (F5-TTS) and RMVPE + HuBERT feature extraction (RVC v2).
- **Inference**: Executes full neural model forward passes to generate synthetic speech.

### 2. High-Fidelity Simulation Mode
- **Activation**: Automatically enabled when CUDA is unavailable (e.g. on laptops with AMD Radeon / Intel Iris Xe integrated graphics, or CPU-only test runners).
- **Realistic Loss Telemetry**: Simulates mathematical training dynamics using calibrated exponential decay with stochastic noise:
  $$L(t) = L_{final} + (L_{init} - L_{final}) e^{-\alpha t} + \mathcal{N}(0, \sigma^2)$$
- **Authentic Checkpoints**: Generates genuine `.pth`, `.index`, and `config.json` checkpoint bundles in `data/models/` identical in structure to real model outputs.
- **Formant Vocal Synthesis (TTS)**: Synthesizes valid 24kHz mono PCM16 audio modulated by reference audio pitch ($F_0$) autocorrelation and Polish text vowel-consonant phonetic timing.
- **Phase-Vocoder Pitch Transposition (STS)**: Transposes audio frequency by $2^{\Delta/12}$ using STFT phase-vocoder math while preserving duration, combined with dual-peak resonant vowel formant filtering.
- **Benefit**: Ensures 100% of API endpoints, WebSockets, UI flows, and automated test suites pass without requiring a multi-gigabyte GPU cluster.

---

## Testing & Quality Assurance

Voice Sampler includes an exhaustive, multi-tier automated test suite guaranteeing robust stability, boundary correctness, and regression resistance.

### Automated E2E Test Suite (Tiers 1–4)

Run the unified E2E test runner executing all 106 End-to-End tests:

```bash
python tests/run_e2e_tests.py
```

#### E2E Test Results:
```
================================================================================
 VOICE SAMPLER -- END-TO-END (E2E) TEST SUITE
================================================================================
 Target Mode  : Auto-Detect
 Python Ver   : 3.14.6
 Platform     : win32
 Active Tiers : [1, 2, 3, 4]
================================================================================

[>] Running Tier 1: Tier 1: Feature Coverage (>=5 tests per feature)...
[OK] Completed Tier 1 in 3.63s

[>] Running Tier 2: Tier 2: Boundary & Corner Cases...
[OK] Completed Tier 2 in 2.59s

[>] Running Tier 3: Tier 3: Cross-Feature Integration Combinations...
[OK] Completed Tier 3 in 1.06s

[>] Running Tier 4: Tier 4: Real-World Polish Voice Cloning Scenarios...
[OK] Completed Tier 4 in 1.61s

================================================================================
 E2E TEST EXECUTION SUMMARY REPORT
================================================================================
Tier    Tier Name                                 Total   Passed  Failed  Time    
--------------------------------------------------------------------------------
Tier 1  Tier 1: Feature Coverage (>=5 tests per   60      60      0       3.63s   
Tier 2  Tier 2: Boundary & Corner Cases           34      34      0       2.59s   
Tier 3  Tier 3: Cross-Feature Integration Combin  6       6       0       1.06s   
Tier 4  Tier 4: Real-World Polish Voice Cloning   6       6       0       1.61s   
--------------------------------------------------------------------------------
TOTAL   All Selected Tiers                        106     106     0       8.89s   
================================================================================

 [SUCCESS] ALL E2E TESTS PASSED (100% SUCCESS RATE)
```

#### E2E Test Runner CLI Options

```bash
# Run a specific tier (e.g. Tier 1 only)
python tests/run_e2e_tests.py --tier 1

# Run with verbose output showing individual test names
python tests/run_e2e_tests.py -v

# Target an existing external server instance
python tests/run_e2e_tests.py --url http://127.0.0.1:8000

# Force embedded simulation mode
python tests/run_e2e_tests.py --mock
```

---

### Backend Unit Tests

Run the complete suite of 94 backend unit tests covering individual routers, services, SQLite WAL transactions, and audio processing:

```bash
pytest tests/unit -v
```

**Unit Test Modules:**
- `test_audio_service.py`: Normalization, resampling, header verification, clipping prevention.
- `test_config.py`: Environment variable propagation, directory initialization.
- `test_connection.py`: SQLite connection lifecycle, WAL mode validation, foreign keys.
- `test_repository.py`: CRUD operations for samples, models, runs, and generation records.
- `test_gpu_service.py`: PyTorch detection, WMI query fallbacks, VRAM calculation.
- `test_routers.py` & `test_samples_router.py`: REST endpoint contracts and audio streaming.
- `test_whisper_service.py`: Background worker queuing, Polish transcription logic.
- `test_train_router.py` & `test_training_services.py`: F5-TTS / RVC training initialization and cancellation.
- `test_generate_router.py`: TTS and STS parameter validation, audio generation.
- `test_models_router.py`: Model checkpoint discovery and deletion.

---

### TypeScript & Frontend Verification

Verify TypeScript types and build assets:

```bash
# Verify TypeScript without emitting files
npm run typecheck
# or: npx tsc --noEmit

# Test full production build
npm run build
```

---

### Adversarial & Regression Test Suites

In addition to standard unit and E2E tests, the repository includes adversarial stress suites designed to test edge cases:

```bash
# Milestone 1: Lifecycle & Hardware Stress
pytest tests/test_m1_lifecycle_stress.py -v
pytest tests/test_m1_empirical_stress.py -v

# Milestone 2: Audio Ingestion, Whisper Stress & Adversarial Inputs
pytest tests/test_m2_adversarial_stress.py -v
pytest tests/test_challenger_m2_7_whisper_stress.py -v

# Milestone 3: Training Concurrency, Hyperparameter Boundaries & Barrier Tests
pytest tests/test_m3_challenger_concurrency.py -v
pytest tests/test_m3_challenger_hyperparams.py -v
pytest tests/test_m3_reviewer_adversarial.py -v

# Milestone 4: Inference Stress, Pitch Boundaries & Audio Streaming
pytest tests/test_m4_challenger_tts.py -v
pytest tests/test_m4_challenger_sts_streaming.py -v
pytest tests/test_m4_reviewer_adversarial.py -v

# Milestone 5: Electron Process Lifecycle & WebSocket Unmount Cleanup
pytest tests/test_m5_challenger_electron_lifecycle.py -v
pytest tests/test_m5_challenger_ws_unmount.py -v
```

---

## Project Directory Layout

```
voice-sampler/
├── package.json                   # Root npm configuration (scripts, dependencies)
├── tsconfig.json                  # Root TypeScript configuration (React app)
├── tsconfig.node.json             # Electron main & preload TypeScript configuration
├── vite.config.ts                 # Vite + React bundler configuration
├── tailwind.config.js             # Tailwind CSS design system configuration
├── postcss.config.js              # PostCSS configuration
├── index.html                     # Frontend HTML document entry point
├── run_server.py                  # CLI Python runner for FastAPI backend
├── requirements.txt               # Python backend dependencies
├── README.md                      # Comprehensive developer and user guide
│
├── electron/                      # Electron Native Desktop Shell
│   ├── main.ts                    # Main process, window creation, child process manager
│   └── preload.ts                 # Preload context bridge (IPC APIs)
│
├── src/                           # React 19 Frontend Application
│   ├── main.tsx                   # React root entry point
│   ├── App.tsx                    # Shell layout with sidebar navigation
│   ├── types.ts                   # Unified TypeScript interfaces and data models
│   ├── index.css                  # Global styles, Tailwind directives, dark theme
│   ├── api/                       # API Communication Layer
│   │   ├── client.ts              # REST API client wrapper (Axios / Fetch)
│   │   └── websocket.ts           # Resilient WebSocket client with auto-reconnect
│   ├── components/                # Reusable UI Components
│   │   ├── Sidebar.tsx            # Accessible navigation sidebar
│   │   ├── AudioPlayer.tsx        # Audio playback bar with scrubber & time display
│   │   ├── WaveformVisualizer.tsx # 60fps real-time microphone waveform visualizer
│   │   ├── LossChart.tsx          # Dynamic SVG loss chart for training telemetry
│   │   ├── StatusBadge.tsx        # Status pills (health, training, GPU mode)
│   │   └── DeleteConfirmModal.tsx # Confirmation modal for resource deletion
│   └── views/                     # Main Section Views
│       ├── SamplesView.tsx        # Mic recorder, drag-drop upload, Polish transcript editor
│       ├── TrainingView.tsx       # F5-TTS & RVC engine selector, hyperparams, live telemetry
│       ├── GenerateView.tsx       # TTS tab, STS tab, pangram presets, player & download
│       └── SettingsView.tsx       # GPU diagnostic card, amber NO-CUDA banner, backend health
│
├── backend/                       # Python FastAPI Backend
│   └── app/
│       ├── __init__.py
│       ├── main.py                # FastAPI app creation, CORS, lifespan management
│       ├── config.py              # Environment configuration, directory setup
│       ├── db/                    # SQLite Storage Layer
│       │   ├── connection.py      # SQLite connection manager with WAL mode
│       │   └── repository.py      # CRUD repository for samples, models, runs
│       ├── routers/               # REST API Endpoints & WebSocket Routes
│       │   ├── health.py          # /api/health and /api/shutdown
│       │   ├── settings.py        # /api/settings/gpu
│       │   ├── samples.py         # /api/samples (upload, record, stream, CRUD)
│       │   ├── train.py           # /api/train (start, stop, status)
│       │   ├── generate.py        # /api/generate (tts, sts, streaming, download)
│       │   ├── models.py          # /api/models (registry query & delete)
│       │   └── websocket.py       # /ws/training and /ws/notifications
│       ├── services/              # Core ML & Audio Processing Services
│       │   ├── gpu_service.py     # GPU & CUDA detection (PyTorch + WMI)
│       │   ├── audio_service.py   # Audio normalization, resampling, WAV encoding
│       │   ├── whisper_service.py # Polish Whisper transcription (faster-whisper)
│       │   ├── f5tts_service.py   # F5-TTS dataset preparation, training, inference
│       │   ├── rvc_service.py     # RVC v2 dataset preparation, training, inference
│       │   └── simulation.py      # High-fidelity parametric DSP simulation engine
│       └── utils/
│           └── logger.py          # Standardized logging configuration
│
├── tests/                         # Test Suites & Quality Assurance
│   ├── e2e/                       # End-to-End Test Suite (Tiers 1–4)
│   │   ├── harness.py             # E2E test client harness
│   │   ├── test_tier1_features.py # Tier 1: Complete feature coverage (60 tests)
│   │   ├── test_tier2_boundary.py # Tier 2: Boundary & corner cases (34 tests)
│   │   ├── test_tier3_combos.py   # Tier 3: Cross-feature combinations (6 tests)
│   │   └── test_tier4_scenarios.py# Tier 4: Polish voice cloning scenarios (6 tests)
│   ├── unit/                      # Backend Unit Test Suite (94 tests)
│   │   ├── test_audio_service.py
│   │   ├── test_config.py
│   │   ├── test_connection.py
│   │   ├── test_generate_router.py
│   │   ├── test_gpu_service.py
│   │   ├── test_models_router.py
│   │   ├── test_repository.py
│   │   ├── test_routers.py
│   │   ├── test_samples_router.py
│   │   ├── test_train_router.py
│   │   ├── test_training_services.py
│   │   └── test_whisper_service.py
│   ├── run_e2e_tests.py           # Unified E2E test runner CLI (106 tests)
│   └── test_m*.py                 # Milestone-specific adversarial stress suites
│
└── data/                          # Runtime Persistent Storage (gitignored)
    ├── db/                        # SQLite database (voice_sampler.db)
    ├── samples/                   # Uploaded and recorded WAV samples
    ├── models/                    # Model checkpoints (.pth, .index, config.json)
    └── generated/                 # Generated output audio files
```

---

## License

This project is licensed under the **MIT License**. See the `LICENSE` file for details.

---

### Acknowledgements

- **faster-whisper**: Fast Whisper reimplementation using CTranslate2.
- **F5-TTS**: Non-autoregressive speech synthesis with flow-matching diffusion transformers.
- **RVC (Retrieval-based Voice Conversion)**: Voice conversion based on VITS and HuBERT audio representations.
- **Electron & Vite**: Fast native desktop developer experience.
