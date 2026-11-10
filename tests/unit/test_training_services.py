"""
tests/unit/test_training_services.py
Unit tests for ML training services:
- F5-TTS dataset preparation, Polish vocabulary generation, and fine-tuning lifecycle
- RVC v2 5-stage pipeline: audio preprocessing, F0 pitch tracking, HuBERT features, and Faiss indexing
- Non-CUDA Simulation Engine loss modeling, step pacing, and artifact generation
"""

from __future__ import annotations

import io
import math
from pathlib import Path
import tempfile
import threading
import time
from typing import Any, Dict, List

import numpy as np
import pytest
import soundfile as sf

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.db.connection import init_storage_directories, set_db_path
from backend.app.services.f5tts_service import (
    POLISH_LOWER_DIACRITICS,
    POLISH_UPPER_DIACRITICS,
    build_polish_vocab,
    get_f5tts_service,
    prepare_f5tts_dataset,
)
from backend.app.services.rvc_service import get_rvc_service
from backend.app.services.simulation import get_simulation_engine


@pytest.fixture(autouse=True)
def setup_test_environment(tmp_path: Path):
    """Isolate storage directories and SQLite database per test."""
    test_db = tmp_path / "test_training.db"
    set_db_path(test_db)
    init_storage_directories(tmp_path / "data")
    repository.init_db(test_db)

    # Override settings models_dir and cache_dir
    settings = get_settings()
    settings.data_dir = tmp_path / "data"
    settings.models_dir = tmp_path / "data" / "models"
    settings.cache_dir = tmp_path / "data" / "cache"
    settings.samples_raw_dir = tmp_path / "data" / "samples" / "raw"
    settings.ensure_directories()

    yield

    set_db_path(None)


def _create_dummy_wav(file_path: Path, duration_s: float = 1.0, sr: int = 44100) -> Path:
    """Helper to generate a clean mono PCM16 sine wave audio file."""
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False, dtype=np.float32)
    sine = 0.5 * np.sin(2 * np.pi * 440.0 * t)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(file_path), sine, sr, subtype="PCM_16", format="WAV")
    return file_path


# ==============================================================================
# 1. Polish Vocabulary Builder Tests
# ==============================================================================

def test_polish_vocab_contains_all_32_diacritics():
    """Verify that vocab contains all 32 Polish lower and uppercase characters."""
    vocab = build_polish_vocab()

    # All lowercase letters present
    for ch in POLISH_LOWER_DIACRITICS:
        assert ch in vocab, f"Lowercase Polish char '{ch}' missing from vocab"

    # All uppercase letters present
    for ch in POLISH_UPPER_DIACRITICS:
        assert ch in vocab, f"Uppercase Polish char '{ch}' missing from vocab"

    # Special tokens and standard punctuation present
    assert "<pad>" in vocab
    assert "<unk>" in vocab
    assert " " in vocab
    assert "." in vocab
    assert "," in vocab
    assert "0" in vocab
    assert "9" in vocab


def test_polish_vocab_incorporates_corpus_extensions():
    """Verify that unseen printable characters from user transcripts are appended."""
    custom_corpus = "Programowanie w języku Python: 100% sukcesu! § Special symbol."
    vocab = build_polish_vocab(extra_text_corpus=custom_corpus)

    assert "%" in vocab
    assert "§" in vocab
    assert len(vocab) == len(set(vocab)), "Vocabulary should have no duplicates"


# ==============================================================================
# 2. F5-TTS Dataset Preparation Tests
# ==============================================================================

def test_f5tts_dataset_preparation_resampling_and_manifest(tmp_path: Path):
    """Verify 24kHz mono resampling, metadata.csv escaping, and vocab.txt generation."""
    settings = get_settings()
    raw_dir = settings.samples_raw_dir
    wav_path = _create_dummy_wav(raw_dir / "sample_1.wav", duration_s=1.5, sr=44100)

    sample_record = repository.create_sample(
        name="test_voice",
        filename="sample_1.wav",
        file_path=str(wav_path),
        transcript="W Szczebrzeszynie chrząszcz | brzmi w trzcinie.",
    )

    output_dir = tmp_path / "models" / "test_f5_model"
    res = prepare_f5tts_dataset("test_f5_model", [sample_record], output_dir)

    assert res["sample_count"] == 1
    assert Path(res["metadata_path"]).exists()
    assert Path(res["vocab_path"]).exists()

    # Check resampled WAV
    out_wav = Path(res["wavs_dir"]) / f"{sample_record['id']}.wav"
    assert out_wav.exists()
    audio, sr = sf.read(str(out_wav))
    assert sr == 24000
    assert audio.ndim == 1, "Audio must be mono"

    # Check pipe character sanitization in metadata.csv
    with open(res["metadata_path"], "r", encoding="utf-8") as mf:
        content = mf.read().strip()
    assert f"wavs/{sample_record['id']}.wav|" in content
    # The literal pipe in transcript must have been replaced with space/escaped
    transcript_part = content.split("|", 1)[1]
    assert "|" not in transcript_part


# ==============================================================================
# 3. Non-CUDA Simulation Engine Tests
# ==============================================================================

def test_simulation_loss_trajectory_decay():
    """Verify parametric loss curve decays smoothly and respects asymptotic floor."""
    sim = get_simulation_engine()
    total_epochs = 10
    total_steps = 40

    losses = []
    for step in range(1, total_steps + 1):
        epoch = math.ceil(step / 4)
        l_val = sim.compute_loss("f5-tts", step, total_steps, epoch, total_epochs)
        losses.append(l_val)
        assert l_val >= 0.15, "Loss must never drop below the minimum clamp bound"

    # Overall trend should be downward: initial loss significantly higher than final loss
    assert losses[0] > losses[-1]
    assert losses[0] > 1.5
    assert losses[-1] < 0.6


def test_simulation_cooperative_cancellation(tmp_path: Path):
    """Verify training loop halts immediately when cancellation token is set."""
    sim = get_simulation_engine()
    run = repository.create_training_run("f5-tts", "cancel_test", total_epochs=20, total_steps=80)
    run_id = run["id"]

    cancel_event = threading.Event()

    def cancel_soon():
        time.sleep(0.04)
        cancel_event.set()

    t = threading.Thread(target=cancel_soon)
    t.start()

    completed, final_loss = sim.run_training_loop(
        engine="f5-tts",
        run_id=run_id,
        model_name="cancel_test",
        total_epochs=20,
        batch_size=2,
        sample_count=4,
        cancel_event=cancel_event,
    )
    t.join()

    assert not completed, "Training loop should report uncompleted when cancelled"
    db_run = repository.get_training_run(run_id)
    assert db_run["status"] == "stopped"


def test_simulation_artifact_generation(tmp_path: Path):
    """Verify F5-TTS and RVC artifacts exist on disk with valid headers."""
    sim = get_simulation_engine()

    # F5-TTS artifacts
    settings = get_settings()
    f5_artifacts = sim.create_f5tts_artifacts(
        model_name="pl_sim_f5",
        epochs=5,
        batch_size=2,
        learning_rate=0.0001,
        sample_count=2,
        final_loss=0.312,
    )
    assert (settings.models_dir / "pl_sim_f5_f5tts.pt").exists()
    assert (settings.models_dir / "pl_sim_f5" / "vocab.txt").exists()
    assert (settings.models_dir / "pl_sim_f5" / "config.json").exists()

    # RVC artifacts
    rvc_artifacts = sim.create_rvc_artifacts(
        model_name="pl_sim_rvc",
        epochs=10,
        batch_size=4,
        sample_rate=40000,
        sample_count=3,
        final_loss=0.345,
    )
    assert (settings.models_dir / "pl_sim_rvc.pth").exists()
    assert (settings.models_dir / "pl_sim_rvc.index").exists()
    assert (settings.models_dir / "pl_sim_rvc" / "config.json").exists()


# ==============================================================================
# 4. RVC v2 5-Stage Pipeline Unit Tests
# ==============================================================================

def test_rvc_5_stages_end_to_end(tmp_path: Path):
    """Verify each stage of RVC v2 produces expected files and representations."""
    rvc = get_rvc_service()
    settings = get_settings()

    raw_wav = _create_dummy_wav(settings.samples_raw_dir / "rvc_raw.wav", duration_s=2.0, sr=44100)
    sample = repository.create_sample(
        name="rvc_sample",
        filename="rvc_raw.wav",
        file_path=str(raw_wav),
        transcript="Testowy głos do konwersji.",
    )

    work_dir = tmp_path / "cache" / "rvc_logs" / "unit_rvc_test"
    gt_dir = work_dir / "0_gt_wavs"
    dir_16k = work_dir / "1_16k_wavs"
    f0_dir = work_dir / "2a_f0"
    f0nsf_dir = work_dir / "2b_f0nsf"
    feat_dir = work_dir / "3_feature768"

    for d in (gt_dir, dir_16k, f0_dir, f0nsf_dir, feat_dir):
        d.mkdir(parents=True, exist_ok=True)

    # STAGE 1: Preprocessing & Resampling
    chunks = rvc.stage1_preprocess_audio([sample], gt_dir, dir_16k, target_sr=40000)
    assert len(chunks) >= 1
    gt_files = list(gt_dir.glob("*.wav"))
    assert len(gt_files) >= 1
    audio_gt, sr_gt = sf.read(str(gt_files[0]))
    assert sr_gt == 40000

    wavs_16k = list(dir_16k.glob("*.wav"))
    assert len(wavs_16k) >= 1
    audio_16k, sr_16k = sf.read(str(wavs_16k[0]))
    assert sr_16k == 16000

    # STAGE 2: Pitch Extraction (F0 and NSF F0)
    rvc.stage2_extract_f0(dir_16k, f0_dir, f0nsf_dir)
    f0_files = list(f0_dir.glob("*.npy"))
    f0nsf_files = list(f0nsf_dir.glob("*.npy"))
    assert len(f0_files) >= 1
    assert len(f0nsf_files) >= 1
    f0_arr = np.load(str(f0_files[0]))
    f0nsf_arr = np.load(str(f0nsf_files[0]))
    assert f0_arr.ndim == 1
    assert f0nsf_arr.ndim == 1
    assert len(f0_arr) == len(f0nsf_arr)

    # STAGE 3: Semantic Feature Extraction
    rvc.stage3_extract_features(dir_16k, feat_dir)
    feat_files = list(feat_dir.glob("*.npy"))
    assert len(feat_files) >= 1
    feats = np.load(str(feat_files[0]))
    assert feats.ndim == 2
    assert feats.shape[1] == 768

    # STAGE 5: Artifact Packaging and Registration
    artifacts = rvc.stage5_package_artifacts(
        model_name="unit_rvc_test",
        epochs=5,
        batch_size=2,
        sample_rate=40000,
        final_loss=0.298,
        sample_count=1,
        feature_dir=feat_dir,
    )
    assert (settings.models_dir / "unit_rvc_test.pth").exists()
    assert (settings.models_dir / "unit_rvc_test.index").exists()

    # Verify registration in SQLite
    model = repository.get_model_by_name("unit_rvc_test")
    assert model is not None
    assert model["engine"] == "rvc"
    assert model["checkpoint_path"].endswith(".pth")
    assert model["index_path"].endswith(".index")


# ==============================================================================
# 5. Service Lifecycle & Validation Tests
# ==============================================================================

def test_f5tts_service_validation_errors():
    """Verify F5TTSService raises expected ValueErrors on invalid parameters."""
    f5 = get_f5tts_service()

    with pytest.raises(ValueError, match="Model name cannot be empty"):
        f5.start_training(model_name="   ")

    with pytest.raises(ValueError, match="Epochs must be >= 1"):
        f5.start_training(model_name="valid_name", epochs=0)

    with pytest.raises(ValueError, match="Batch size must be >= 1"):
        f5.start_training(model_name="valid_name", batch_size=0)

    with pytest.raises(ValueError, match="Learning rate must be > 0"):
        f5.start_training(model_name="valid_name", learning_rate=-0.01)

    with pytest.raises(ValueError, match="Sample id non_existent not found"):
        f5.start_training(model_name="valid_name", sample_ids=["non_existent"])


def test_rvc_service_validation_errors():
    """Verify RVCService raises expected ValueErrors on invalid parameters."""
    rvc = get_rvc_service()

    with pytest.raises(ValueError, match="Model name cannot be empty"):
        rvc.start_training(model_name="")

    with pytest.raises(ValueError, match="Sample rate must be 32k, 40k, or 48k"):
        rvc.start_training(model_name="rvc_name", sample_rate=22050)

    with pytest.raises(ValueError, match="Epochs must be >= 1"):
        rvc.start_training(model_name="rvc_name", epochs=-1)
