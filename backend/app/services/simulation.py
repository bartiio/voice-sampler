"""
backend/app/services/simulation.py
Non-CUDA Simulation Engine for ML Training Pipelines & Model Checkpoint Persistence.
Delivers realistic exponential loss descent, step pacing, cooperative cancellation,
and authentic on-disk artifact synthesis for F5-TTS and RVC v2 on non-CUDA hosts.
Part of Voice Sampler Milestone 3 (M3).
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
import logging
import math
import os
from pathlib import Path
import pickle
import shutil
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.signal as signal
import soundfile as sf

from backend.app.config import get_settings
from backend.app.db import repository
from backend.app.routers.websocket import broadcast_training_telemetry

logger = logging.getLogger("voice_sampler.simulation")


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


class SimulationEngine:
    """
    Simulates ML fine-tuning dynamics for F5-TTS (Flow-Matching Diffusion Transformer)
    and RVC v2 (NSF-VITS Generator/Discriminator) on systems without CUDA hardware.
    Maintains authentic state, real-time progress broadcasts, and genuine file generation.
    """

    def __init__(self) -> None:
        self.settings = get_settings()

    def get_step_delay(self, total_steps: int) -> float:
        """
        Calculates calibrated sleep delay per step to ensure responsive telemetry
        without slowing down test suites. Can be overridden via environment variable.
        """
        env_val = os.getenv("VOICE_SAMPLER_TRAIN_PACE")
        if env_val is not None:
            try:
                return max(0.001, float(env_val))
            except ValueError:
                pass
        # Default adaptive pacing: scale inversely with total steps (between 10ms and 50ms)
        return max(0.01, min(0.05, 0.4 / max(1, total_steps)))

    def compute_loss(
        self,
        engine: str,
        step: int,
        total_steps: int,
        epoch: int,
        total_epochs: int,
    ) -> float:
        """
        Computes mathematically grounded decaying loss trajectory:
        L(t) = L_min + (L_init - L_min) * exp(-(t - 1) / tau) + epsilon(t)
        """
        if engine == "f5-tts":
            init_loss = 2.35
            min_loss = 0.32
            tau = max(2.5, total_epochs * 0.35)
        else:  # rvc
            init_loss = 2.45
            min_loss = 0.32
            tau = max(3.0, total_epochs * 0.35)

        progress_fraction = (step - 1) / max(1, total_steps - 1)
        decay = math.exp(-progress_fraction * (total_epochs / max(1.0, tau)))
        jitter = float(np.random.normal(0, 0.015))
        loss = min_loss + (init_loss - min_loss) * decay + jitter
        return round(max(0.18, loss), 4)

    def run_training_loop(
        self,
        engine: str,
        run_id: str,
        model_name: str,
        total_epochs: int,
        batch_size: int,
        sample_count: int,
        cancel_event: Optional[threading.Event] = None,
        on_step_hook: Optional[Callable[[int, int, float, float], None]] = None,
    ) -> Tuple[bool, float]:
        """
        Executes simulated training loop over epochs and steps with telemetry streaming.
        Returns: (completed_successfully: bool, final_loss: float).
        If cancelled cooperatively, returns (False, current_loss).
        """
        steps_per_epoch = max(1, math.ceil(sample_count / max(1, batch_size)))
        total_steps = total_epochs * steps_per_epoch
        step_delay = self.get_step_delay(total_steps)

        current_loss = 2.40
        global_step = 0

        logger.info(
            f"Starting simulation training loop for {engine} (run: {run_id}, "
            f"model: {model_name}, epochs: {total_epochs}, steps: {total_steps}, pace: {step_delay:.3f}s)"
        )

        for epoch in range(1, total_epochs + 1):
            for step_in_epoch in range(1, steps_per_epoch + 1):
                global_step += 1

                # Cooperative cancellation check
                if cancel_event is not None and cancel_event.is_set():
                    logger.info(f"Simulation run {run_id} cancelled at step {global_step}/{total_steps}")
                    repository.complete_training_run(
                        run_id=run_id,
                        status="stopped",
                        error_message="Training cancelled by user request",
                    )
                    broadcast_training_telemetry({
                        "type": "stopped",
                        "engine": engine,
                        "run_id": run_id,
                        "status": "stopped",
                        "model_name": model_name,
                        "message": "Training cancelled by user request",
                        "timestamp": _utc_now_iso(),
                    })
                    return False, current_loss

                # Calculate loss and progress metrics
                current_loss = self.compute_loss(engine, global_step, total_steps, epoch, total_epochs)
                progress_pct = round(min(100.0, (global_step / float(total_steps)) * 100.0), 1)
                message = f"Epoch {epoch}/{total_epochs} (step {global_step}/{total_steps}, loss: {current_loss:.4f})"

                # Update database record
                repository.update_training_run_progress(
                    run_id=run_id,
                    current_epoch=epoch,
                    current_step=global_step,
                    current_loss=current_loss,
                    progress_percent=progress_pct,
                    status_message=message,
                    status="running",
                )

                # Broadcast real-time WebSocket telemetry frame
                broadcast_training_telemetry({
                    "type": "progress",
                    "engine": engine,
                    "run_id": run_id,
                    "epoch": epoch,
                    "total_epochs": total_epochs,
                    "step": global_step,
                    "total_steps": total_steps,
                    "loss": current_loss,
                    "progress_percent": progress_pct,
                    "message": message,
                    "status": "training",
                    "timestamp": _utc_now_iso(),
                })

                if on_step_hook:
                    on_step_hook(epoch, global_step, current_loss, progress_pct)

                if step_delay > 0:
                    time.sleep(step_delay)

        return True, current_loss

    def create_f5tts_artifacts(
        self,
        model_name: str,
        epochs: int,
        batch_size: int,
        learning_rate: float,
        sample_count: int,
        final_loss: float,
        vocab_tokens: Optional[List[str]] = None,
    ) -> Dict[str, str]:
        """
        Creates genuine F5-TTS model artifacts:
        - data/models/{model_name}/{model_name}_f5tts.pt
        - data/models/{model_name}/vocab.txt
        - data/models/{model_name}/config.json
        - Root alias at data/models/{model_name}_f5tts.pt
        """
        models_root = self.settings.models_dir
        model_dir = (models_root / model_name).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)

        primary_ckpt = model_dir / f"{model_name}_f5tts.pt"
        vocab_path = model_dir / "vocab.txt"
        config_path = model_dir / "config.json"

        # 1. Build authentic vocabulary if not supplied
        if not vocab_tokens:
            from backend.app.services.f5tts_service import build_polish_vocab
            vocab_tokens = build_polish_vocab()

        with open(vocab_path, "w", encoding="utf-8") as vf:
            for token in vocab_tokens:
                vf.write(f"{token}\n")

        # 2. Synthesize authentic checkpoint dictionary
        ckpt_bundle = {
            "model_type": "F5TTS",
            "architecture": "Diffusion-Transformer",
            "model_name": model_name,
            "vocab_size": len(vocab_tokens),
            "dim": 1024,
            "depth": 22,
            "heads": 16,
            "final_loss": final_loss,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "sample_rate": 24000,
            "created_at": _utc_now_iso(),
            "simulation": True,
            "ema_weights": {
                "transformer.embed.weight": np.random.randn(len(vocab_tokens), 1024).astype(np.float32),
                "transformer.final_proj.weight": np.random.randn(1024, 100).astype(np.float32),
            },
        }

        # Save checkpoint using PyTorch if available, else standard pickle binary
        try:
            import torch
            torch.save(ckpt_bundle, str(primary_ckpt))
        except (ImportError, Exception):
            with open(primary_ckpt, "wb") as pf:
                pickle.dump(ckpt_bundle, pf, protocol=4)

        # 3. Create root flat convenience copy (data/models/{model_name}_f5tts.pt)
        root_ckpt = models_root / f"{model_name}_f5tts.pt"
        shutil.copyfile(str(primary_ckpt), str(root_ckpt))

        # Root flat vocab and config
        root_vocab = models_root / f"{model_name}_vocab.txt"
        shutil.copyfile(str(vocab_path), str(root_vocab))

        # 4. Save config.json
        config_data = {
            "model_name": model_name,
            "engine": "f5-tts",
            "architecture": "Diffusion-Transformer",
            "sample_rate": 24000,
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "final_loss": final_loss,
            "sample_count": sample_count,
            "checkpoint_path": f"data/models/{model_name}_f5tts.pt",
            "vocab_path": f"data/models/{model_name}/vocab.txt",
            "config_path": f"data/models/{model_name}/config.json",
            "simulation": True,
            "created_at": _utc_now_iso(),
        }
        with open(config_path, "w", encoding="utf-8") as cf:
            json.dump(config_data, cf, indent=2)

        root_config = models_root / f"{model_name}_config.json"
        shutil.copyfile(str(config_path), str(root_config))

        rel_ckpt = f"data/models/{model_name}_f5tts.pt"
        rel_vocab = f"data/models/{model_name}/vocab.txt"
        rel_config = f"data/models/{model_name}/config.json"

        return {
            "checkpoint_path": rel_ckpt,
            "vocab_path": rel_vocab,
            "config_path": rel_config,
        }

    def create_rvc_artifacts(
        self,
        model_name: str,
        epochs: int,
        batch_size: int,
        sample_rate: int,
        sample_count: int,
        final_loss: float,
        feature_vectors: Optional[np.ndarray] = None,
    ) -> Dict[str, str]:
        """
        Creates genuine RVC v2 model artifacts:
        - data/models/{model_name}/{model_name}.pth
        - data/models/{model_name}/{model_name}.index
        - data/models/{model_name}/config.json
        - Root aliases at data/models/{model_name}.pth and .index
        """
        models_root = self.settings.models_dir
        model_dir = (models_root / model_name).resolve()
        model_dir.mkdir(parents=True, exist_ok=True)

        primary_pth = model_dir / f"{model_name}.pth"
        primary_index = model_dir / f"{model_name}.index"
        config_path = model_dir / "config.json"

        # 1. Build Faiss Index from feature vectors
        if feature_vectors is None or len(feature_vectors) == 0:
            feature_vectors = np.random.randn(300, 768).astype(np.float32)

        n_samples = len(feature_vectors)
        try:
            import faiss
            if n_samples >= 256:
                nlist = min(256, max(1, n_samples // 39))
                quantizer = faiss.IndexFlatL2(768)
                idx = faiss.IndexIVFFlat(quantizer, 768, nlist, faiss.METRIC_L2)
                idx.train(feature_vectors)
                idx.add(feature_vectors)
            else:
                idx = faiss.IndexFlatL2(768)
                idx.add(feature_vectors)
            faiss.write_index(idx, str(primary_index))
        except (ImportError, Exception):
            # Standalone structured binary index container with magic bytes
            with open(primary_index, "wb") as f:
                f.write(b"IXFL")  # Faiss IndexFlatL2 magic header
                f.write(np.int32(768).tobytes())
                f.write(np.int64(n_samples).tobytes())
                f.write(feature_vectors.tobytes())

        # 2. Package Small Model Generator Checkpoint (.pth)
        rvc_bundle = {
            "weight": {
                "enc_p.emb.weight": np.random.randn(256, 768).astype(np.float32),
                "dec.conv_pre.weight": np.random.randn(512, 192, 7).astype(np.float32),
            },
            "config": [
                1025, 32, 192, 192, 768, 2, 6, 3, 0, "1",
                [3, 7, 11],
                [[1, 3, 5], [1, 3, 5], [1, 3, 5]],
                [10, 10, 2, 2],
                512,
                [16, 16, 4, 4],
                0,
                256,
                sample_rate,
            ],
            "params": {
                "sr": f"{sample_rate // 1000}k",
                "f0": 1,
                "version": "v2",
                "model_name": model_name,
                "created_at": _utc_now_iso(),
                "epochs": epochs,
                "batch_size": batch_size,
            },
            "info": "Trained with Voice Sampler RVC v2 Pipeline (Simulation)",
        }

        try:
            import torch
            torch.save(rvc_bundle, str(primary_pth))
        except (ImportError, Exception):
            with open(primary_pth, "wb") as pf:
                pickle.dump(rvc_bundle, pf, protocol=4)

        # 3. Create root flat convenience copies
        root_pth = models_root / f"{model_name}.pth"
        root_index = models_root / f"{model_name}.index"
        shutil.copyfile(str(primary_pth), str(root_pth))
        shutil.copyfile(str(primary_index), str(root_index))

        # 4. Save config.json
        meta_dict = {
            "model_name": model_name,
            "engine": "rvc",
            "architecture": "NSF-VITS-v2",
            "sample_rate": sample_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "final_loss": final_loss,
            "sample_count": sample_count,
            "checkpoint_path": f"data/models/{model_name}.pth",
            "index_path": f"data/models/{model_name}.index",
            "simulation": True,
            "created_at": _utc_now_iso(),
        }
        with open(config_path, "w", encoding="utf-8") as cf:
            json.dump(meta_dict, cf, indent=2)

        root_config = models_root / f"{model_name}_config.json"
        shutil.copyfile(str(config_path), str(root_config))

        rel_pth = f"data/models/{model_name}.pth"
        rel_index = f"data/models/{model_name}.index"
        rel_config = f"data/models/{model_name}/config.json"

        return {
            "checkpoint_path": rel_pth,
            "index_path": rel_index,
            "config_path": rel_config,
        }

    def generate_tts_audio(
        self,
        text: str,
        reference_path: Optional[Path] = None,
        speed: float = 1.0,
        sample_rate: int = 24000,
    ) -> Tuple[np.ndarray, int]:
        """
        Authentic formant vocal synthesis modulated by text length/syllables and reference pitch.
        Output valid 24kHz mono PCM16 audio.
        """
        sr = sample_rate
        speed_clamped = max(0.5, min(2.0, float(speed)))

        # 1. Pitch & Timbre Extraction from Reference Audio
        base_f0 = 150.0  # Median fundamental frequency (Hz)
        if reference_path and Path(reference_path).exists():
            try:
                ref_data, ref_sr = sf.read(str(reference_path), dtype="float32")
                if ref_data.ndim > 1:
                    ref_data = np.mean(ref_data, axis=-1)
                # Compute autocorrelation on central 1.0 second
                center = len(ref_data) // 2
                chunk_len = min(len(ref_data), ref_sr)
                start_idx = max(0, center - chunk_len // 2)
                chunk = ref_data[start_idx : start_idx + chunk_len]
                if len(chunk) > 512:
                    corr = np.correlate(chunk, chunk, mode="full")
                    corr = corr[len(corr) // 2 :]
                    min_lag = max(1, ref_sr // 350)  # max 350 Hz
                    max_lag = min(len(corr) - 1, ref_sr // 75)   # min 75 Hz
                    if max_lag > min_lag:
                        best_lag = min_lag + np.argmax(corr[min_lag:max_lag])
                        est_f0 = ref_sr / float(best_lag)
                        if 80.0 <= est_f0 <= 320.0:
                            base_f0 = est_f0
            except Exception as e:
                logger.debug(f"Could not extract pitch from reference audio: {e}")
        else:
            base_f0 = 135.0 + (abs(hash(text)) % 60)

        # 2. Timing & Syllable Estimation from Polish Text
        vowels = set("aąeęioóuyAĄEĘIOÓUY")
        words = text.split()
        syllable_count = sum(1 for c in text if c in vowels)
        syllable_count = max(len(words), max(1, syllable_count))

        avg_syllable_dur = 0.20 / speed_clamped
        pause_time = (
            text.count(",") * 0.18 +
            text.count(";") * 0.25 +
            text.count(".") * 0.40 +
            text.count("!") * 0.45 +
            text.count("?") * 0.45
        ) / speed_clamped

        total_duration = max(1.0, min(120.0, syllable_count * avg_syllable_dur + pause_time))
        num_samples = int(total_duration * sr)
        t = np.linspace(0, total_duration, num_samples, endpoint=False, dtype=np.float32)

        # 3. Prosodic F0 Contour (Declination + Question Intonation + Vibrato)
        f0_contour = base_f0 * (1.0 - 0.10 * (t / total_duration))
        if text.strip().endswith("?"):
            q_samples = min(int(0.3 * sr), num_samples // 3)
            if q_samples > 0:
                f0_contour[-q_samples:] += np.linspace(0.0, 30.0, q_samples, dtype=np.float32)

        vibrato = 0.012 * np.sin(2 * np.pi * 5.2 * t)
        f0_actual = f0_contour * (1.0 + vibrato)
        phase = 2 * np.pi * np.cumsum(f0_actual) / sr

        # 4. Multi-Harmonic Glottal Source
        audio = (
            0.45 * np.sin(phase) +
            0.28 * np.sin(2 * phase) +
            0.18 * np.sin(3 * phase) +
            0.11 * np.sin(4 * phase) +
            0.07 * np.sin(5 * phase) +
            0.04 * np.sin(6 * phase) +
            0.02 * np.sin(8 * phase)
        )

        # 5. Formant Vocal Tract Resonance (F1 ~600Hz, F2 ~1600Hz)
        audio += 0.16 * np.sin(phase * (600.0 / base_f0))
        audio += 0.10 * np.sin(phase * (1600.0 / base_f0))

        # 6. Syllabic Envelope & Smoothing
        syllable_freq = (syllable_count / total_duration) * 2 * np.pi
        envelope = 0.60 + 0.40 * np.sin(syllable_freq * t)
        audio = audio * envelope

        fade_in_len = min(int(0.03 * sr), num_samples // 4)
        fade_out_len = min(int(0.06 * sr), num_samples // 4)
        if fade_in_len > 0:
            audio[:fade_in_len] *= np.linspace(0.0, 1.0, fade_in_len, dtype=np.float32)
        if fade_out_len > 0:
            audio[-fade_out_len:] *= np.linspace(1.0, 0.0, fade_out_len, dtype=np.float32)

        # 7. Peak Normalization to -0.5 dBFS (0.95) & PCM16 Conversion
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-7:
            audio = audio * (0.95 / peak)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16_data = (audio * 32767.0).astype(np.int16)

        return pcm16_data, sr

    def generate_sts_audio(
        self,
        input_audio_path: Path,
        pitch_shift: int = 0,
        index_rate: float = 0.75,
        sample_rate: int = 40000,
    ) -> Tuple[np.ndarray, int]:
        """
        Phase-vocoder pitch transposition (2^(Δ/12)) and resonant formant timbre blending.
        Output valid 40kHz (or specified) mono PCM16 audio.
        """
        in_path = Path(input_audio_path)
        if not in_path.exists() or not in_path.is_file():
            raise ValueError(f"Input audio file not found: {input_audio_path}")

        # 1. Read input audio
        audio, in_sr = sf.read(str(in_path), dtype="float32")
        if audio.ndim > 1:
            audio = np.mean(audio, axis=-1)
        audio = audio.flatten()

        if len(audio) == 0:
            raise ValueError("Input audio file is empty")

        # 2. Resample to target sample rate (40,000 Hz)
        if in_sr != sample_rate:
            gcd = math.gcd(in_sr, sample_rate)
            up = sample_rate // gcd
            down = in_sr // gcd
            audio = signal.resample_poly(audio, up, down).astype(np.float32)

        # 3. Phase-vocoder Pitch Transposition
        if pitch_shift != 0:
            audio = self._pitch_shift_phase_vocoder(audio, float(pitch_shift), sample_rate)

        # 4. Resonant Formant Timbre Blending
        clamped_index_rate = max(0.0, min(1.0, float(index_rate)))
        if clamped_index_rate > 0.0:
            f1 = 650.0   # F1 vowel formant ~650 Hz
            f2 = 1950.0  # F2 vowel formant ~1950 Hz
            b1, a1 = signal.iirpeak(f1, 3.0, fs=sample_rate)
            b2, a2 = signal.iirpeak(f2, 4.0, fs=sample_rate)

            colored = signal.lfilter(b1, a1, audio) + 0.6 * signal.lfilter(b2, a2, audio)
            audio = (1.0 - clamped_index_rate) * audio + clamped_index_rate * (colored * 1.5)

        # 5. Peak Normalization & PCM16 Conversion
        peak = float(np.max(np.abs(audio)))
        if peak > 1e-7:
            audio = audio * (0.95 / peak)
        audio = np.clip(audio, -1.0, 1.0)
        pcm16_data = (audio * 32767.0).astype(np.int16)

        return pcm16_data, sample_rate

    def _pitch_shift_phase_vocoder(self, audio: np.ndarray, semitones: float, sr: int) -> np.ndarray:
        """
        Phase-vocoder pitch transposition preserving time duration.
        Transposes frequency by 2^(semitones / 12.0) while keeping sample length constant.
        """
        if semitones == 0:
            return audio.copy()

        ratio = 2.0 ** (semitones / 12.0)
        n_fft = min(2048, len(audio))
        n_fft = 2 ** int(np.floor(np.log2(n_fft)))
        if n_fft < 64:
            return audio.copy()

        hop_analysis = max(16, n_fft // 4)
        hop_synthesis = max(16, int(round(hop_analysis * ratio)))
        window = np.hanning(n_fft)
        pad_len = n_fft
        padded = np.pad(audio, (pad_len // 2, pad_len // 2), mode="reflect")
        n_frames = (len(padded) - n_fft) // hop_analysis + 1
        if n_frames < 2:
            return audio.copy()

        frames = np.array([padded[i * hop_analysis : i * hop_analysis + n_fft] * window for i in range(n_frames)])
        spectra = np.fft.rfft(frames, axis=1)

        phase_acc = np.angle(spectra[0])
        omega = 2.0 * np.pi * np.arange(spectra.shape[1]) * hop_analysis / n_fft
        syn_spectra = np.zeros_like(spectra)
        syn_spectra[0] = spectra[0]

        for i in range(1, n_frames):
            dphi = np.angle(spectra[i]) - np.angle(spectra[i - 1]) - omega
            dphi_wrapped = (dphi + np.pi) % (2.0 * np.pi) - np.pi
            true_freq = omega + dphi_wrapped
            phase_acc += true_freq * ratio
            syn_spectra[i] = np.abs(spectra[i]) * np.exp(1j * phase_acc)

        syn_frames = np.fft.irfft(syn_spectra, axis=1) * window
        out_len = (n_frames - 1) * hop_synthesis + n_fft
        stretched = np.zeros(out_len, dtype=np.float32)
        norm = np.zeros(out_len, dtype=np.float32)
        for i in range(n_frames):
            st = i * hop_synthesis
            stretched[st : st + n_fft] += syn_frames[i]
            norm[st : st + n_fft] += window ** 2
        norm[norm < 1e-4] = 1.0
        stretched /= norm

        valid = stretched[pad_len // 2 : max(pad_len // 2 + 1, out_len - pad_len // 2)]
        if len(valid) == 0:
            return audio.copy()
        resampled = signal.resample(valid, len(audio)).astype(np.float32)
        return resampled


_simulation_engine: Optional[SimulationEngine] = None


def get_simulation_engine() -> SimulationEngine:
    """Return SimulationEngine singleton instance."""
    global _simulation_engine
    if _simulation_engine is None:
        _simulation_engine = SimulationEngine()
    return _simulation_engine


def generate_tts_audio(
    text: str,
    reference_path: Optional[Path] = None,
    speed: float = 1.0,
    sample_rate: int = 24000,
) -> Tuple[np.ndarray, int]:
    """Module-level shortcut for SimulationEngine.generate_tts_audio."""
    return get_simulation_engine().generate_tts_audio(
        text=text,
        reference_path=reference_path,
        speed=speed,
        sample_rate=sample_rate,
    )


def generate_sts_audio(
    input_audio_path: Path,
    pitch_shift: int = 0,
    index_rate: float = 0.75,
    sample_rate: int = 40000,
) -> Tuple[np.ndarray, int]:
    """Module-level shortcut for SimulationEngine.generate_sts_audio."""
    return get_simulation_engine().generate_sts_audio(
        input_audio_path=input_audio_path,
        pitch_shift=pitch_shift,
        index_rate=index_rate,
        sample_rate=sample_rate,
    )

