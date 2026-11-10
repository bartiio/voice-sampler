"""
Tier 1: Feature Coverage E2E Tests
Requires >=5 tests per core feature:
1. Health & Lifecycle
2. Settings & GPU Detection
3. Samples CRUD Operations
4. Audio Ingestion & Byte-Range Streaming
5. Transcript Editing & Unicode
6. Whisper Polish Transcription Engine
7. F5-TTS Dataset Preparation & Training
8. RVC v2 Dataset Preparation & Training
9. F5-TTS Speech Synthesis (TTS)
10. RVC v2 Voice Conversion (STS)
"""

import unittest
from tests.e2e.harness import (
    EmbeddedMockServer,
    SyntheticAudioGenerator,
    VoiceSamplerTestClient,
    get_test_client,
)


class TestTier1Health(unittest.TestCase):
    """Feature 1: Health & Lifecycle (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())

    def test_health_returns_http_200(self):
        resp = self.client.get_health()
        self.assertEqual(resp.status_code, 200)

    def test_health_status_field_is_ok(self):
        resp = self.client.get_health()
        data = resp.json()
        self.assertEqual(data.get("status"), "ok")

    def test_health_version_matches_semver(self):
        resp = self.client.get_health()
        data = resp.json()
        version = data.get("version")
        self.assertIsNotNone(version)
        self.assertTrue(len(version.split(".")) >= 2)

    def test_health_backend_ready_flag_is_true(self):
        resp = self.client.get_health()
        data = resp.json()
        self.assertTrue(data.get("backend_ready"))

    def test_shutdown_endpoint_returns_shutting_down(self):
        resp = self.client.post_shutdown()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertEqual(data.get("status"), "shutting_down")


class TestTier1SettingsGPU(unittest.TestCase):
    """Feature 2: Settings & GPU Detection (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())

    def test_settings_gpu_returns_http_200(self):
        resp = self.client.get_gpu_settings()
        self.assertEqual(resp.status_code, 200)

    def test_settings_gpu_reports_cuda_boolean(self):
        resp = self.client.get_gpu_settings()
        data = resp.json()
        self.assertIn("cuda_available", data)
        self.assertIsInstance(data["cuda_available"], bool)

    def test_settings_gpu_provides_device_name(self):
        resp = self.client.get_gpu_settings()
        data = resp.json()
        self.assertIn("gpu_name", data)
        self.assertTrue(len(data["gpu_name"]) > 0)

    def test_settings_gpu_contains_vram_metrics(self):
        resp = self.client.get_gpu_settings()
        data = resp.json()
        self.assertIn("vram_total_mb", data)
        self.assertIn("vram_allocated_mb", data)
        self.assertIsInstance(data["vram_total_mb"], (int, float))

    def test_settings_gpu_explains_simulation_mode_when_non_cuda(self):
        resp = self.client.get_gpu_settings()
        data = resp.json()
        if not data["cuda_available"]:
            self.assertTrue(data.get("simulation_mode"))
            self.assertIn("simulation", data.get("message", "").lower())


class TestTier1SamplesCRUD(unittest.TestCase):
    """Feature 3: Sample CRUD Operations (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        self.audio = SyntheticAudioGenerator.generate_sine_wav(duration_s=1.5)

    def test_initial_samples_list_is_empty(self):
        resp = self.client.get_samples()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"samples": []})

    def test_upload_adds_sample_to_list(self):
        self.client.upload_sample(self.audio, filename="voice1.wav")
        resp = self.client.get_samples()
        data = resp.json()
        self.assertEqual(len(data["samples"]), 1)
        self.assertEqual(data["samples"][0]["filename"], "voice1.wav")

    def test_sample_schema_has_required_attributes(self):
        up_resp = self.client.upload_sample(self.audio, filename="voice_schema.wav")
        sample = up_resp.json()
        self.assertIn("id", sample)
        self.assertIn("filename", sample)
        self.assertIn("duration", sample)
        self.assertIn("transcript", sample)
        self.assertIn("status", sample)
        self.assertIn("created_at", sample)

    def test_update_sample_transcript_succeeds(self):
        up_resp = self.client.upload_sample(self.audio)
        sample_id = up_resp.json()["id"]
        new_text = "To jest nowy tekst transkrypcji."
        upd_resp = self.client.update_sample_transcript(sample_id, new_text)
        self.assertEqual(upd_resp.status_code, 200)
        self.assertEqual(upd_resp.json()["transcript"], new_text)

    def test_delete_sample_removes_from_registry(self):
        up_resp = self.client.upload_sample(self.audio)
        sample_id = up_resp.json()["id"]
        del_resp = self.client.delete_sample(sample_id)
        self.assertEqual(del_resp.status_code, 200)
        self.assertEqual(del_resp.json()["status"], "deleted")
        # Ensure it is gone from listing
        list_resp = self.client.get_samples()
        self.assertEqual(len(list_resp.json()["samples"]), 0)


class TestTier1AudioStreaming(unittest.TestCase):
    """Feature 4: Audio Ingestion & Byte-Range Streaming (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        self.audio = SyntheticAudioGenerator.generate_speech_dummy_wav(duration_s=2.0)
        up_resp = self.client.upload_sample(self.audio, filename="stream_test.wav")
        self.sample_id = up_resp.json()["id"]

    def test_stream_returns_http_200_without_range(self):
        resp = self.client.get_sample_audio(self.sample_id)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers.get("Content-Type"), "audio/wav")

    def test_stream_advertises_accept_ranges_bytes(self):
        resp = self.client.get_sample_audio(self.sample_id)
        self.assertEqual(resp.headers.get("Accept-Ranges"), "bytes")

    def test_stream_returns_http_206_for_partial_range(self):
        resp = self.client.get_sample_audio(self.sample_id, byte_range="bytes=0-1023")
        self.assertEqual(resp.status_code, 206)
        self.assertIn("Content-Range", resp.headers)
        self.assertTrue(resp.headers["Content-Range"].startswith("bytes 0-1023/"))
        self.assertEqual(len(resp.content), 1024)

    def test_stream_returns_suffix_partial_range(self):
        resp = self.client.get_sample_audio(self.sample_id, byte_range="bytes=100-200")
        self.assertEqual(resp.status_code, 206)
        self.assertEqual(len(resp.content), 101)

    def test_microphone_recording_ingestion_endpoint(self):
        mic_audio = SyntheticAudioGenerator.generate_sine_wav(duration_s=1.0)
        rec_resp = self.client.record_sample(mic_audio, filename="mic_take_1.wav")
        self.assertEqual(rec_resp.status_code, 200)
        rec_data = rec_resp.json()
        self.assertEqual(rec_data["filename"], "mic_take_1.wav")
        self.assertEqual(rec_data["status"], "transcribed")


class TestTier1TranscriptEditing(unittest.TestCase):
    """Feature 5: Transcript Editing & Unicode Persistence (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        audio = SyntheticAudioGenerator.generate_sine_wav(duration_s=1.0)
        up_resp = self.client.upload_sample(audio)
        self.sample_id = up_resp.json()["id"]

    def test_edit_transcript_persists_polish_diacritics(self):
        polish_text = "Zażółć gęślą jaźń — chrząszcz brzmi w trzcinie."
        resp = self.client.update_sample_transcript(self.sample_id, polish_text)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["transcript"], polish_text)

    def test_edit_transcript_uppercase_polish_diacritics(self):
        caps_text = "ĄĆĘŁŃÓŚŹŻ"
        resp = self.client.update_sample_transcript(self.sample_id, caps_text)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["transcript"], caps_text)

    def test_edit_transcript_persists_across_sample_reloads(self):
        text = "Trwałe zapisanie transkrypcji w bazie danych."
        self.client.update_sample_transcript(self.sample_id, text)
        list_resp = self.client.get_samples()
        matched = [s for s in list_resp.json()["samples"] if s["id"] == self.sample_id]
        self.assertEqual(len(matched), 1)
        self.assertEqual(matched[0]["transcript"], text)

    def test_edit_transcript_with_punctuation_and_numbers(self):
        punct_text = "Próba 1, 2, 3: czy głos brzmi czysto?! (100% poprawności)."
        resp = self.client.update_sample_transcript(self.sample_id, punct_text)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["transcript"], punct_text)

    def test_edit_transcript_on_nonexistent_sample_returns_404(self):
        resp = self.client.update_sample_transcript("non-existent-uuid", "Nowy tekst")
        self.assertEqual(resp.status_code, 404)


class TestTier1WhisperTranscription(unittest.TestCase):
    """Feature 6: Whisper Polish Transcription Engine (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        self.speech_audio = SyntheticAudioGenerator.generate_speech_dummy_wav(duration_s=3.0)

    def test_upload_sample_automatically_generates_transcript(self):
        resp = self.client.upload_sample(self.speech_audio)
        sample = resp.json()
        self.assertIsNotNone(sample.get("transcript"))
        self.assertTrue(len(sample["transcript"]) > 0)

    def test_auto_transcript_sets_status_to_transcribed(self):
        resp = self.client.upload_sample(self.speech_audio)
        self.assertEqual(resp.json().get("status"), "transcribed")

    def test_mic_record_automatically_generates_polish_transcript(self):
        resp = self.client.record_sample(self.speech_audio)
        sample = resp.json()
        self.assertIn("transcript", sample)
        # Should contain Polish phrase
        self.assertIn("polsk", sample["transcript"].lower())

    def test_transcription_persists_in_sample_listing(self):
        up_resp = self.client.upload_sample(self.speech_audio)
        sid = up_resp.json()["id"]
        samples = self.client.get_samples().json()["samples"]
        target = [s for s in samples if s["id"] == sid][0]
        self.assertEqual(target["status"], "transcribed")

    def test_duration_calculated_from_audio_length(self):
        # 2 seconds WAV
        audio_2s = SyntheticAudioGenerator.generate_sine_wav(duration_s=2.0)
        resp = self.client.upload_sample(audio_2s)
        self.assertAlmostEqual(resp.json()["duration"], 2.0, delta=0.5)


class TestTier1F5TTSTraining(unittest.TestCase):
    """Feature 7: F5-TTS Dataset Preparation & Training (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        audio = SyntheticAudioGenerator.generate_speech_dummy_wav(duration_s=2.5)
        up_resp = self.client.upload_sample(audio)
        self.sample_id = up_resp.json()["id"]

    def test_start_f5tts_training_returns_run_id(self):
        resp = self.client.train_f5tts(
            model_name="pl_voice_alpha", epochs=5, sample_ids=[self.sample_id]
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("run_id", data)
        self.assertEqual(data["status"], "started")
        self.assertEqual(data["engine"], "f5-tts")

    def test_f5tts_registers_checkpoint_in_models_registry(self):
        self.client.train_f5tts(
            model_name="pl_model_checkpoint", epochs=2, sample_ids=[self.sample_id]
        )
        models_resp = self.client.get_models()
        data = models_resp.json()
        matching = [m for m in data["models"] if m["name"] == "pl_model_checkpoint"]
        self.assertEqual(len(matching), 1)
        self.assertEqual(matching[0]["engine"], "f5-tts")
        self.assertTrue(matching[0]["checkpoint_path"].endswith("_f5tts.pt"))

    def test_f5tts_supports_custom_hyperparameters(self):
        resp = self.client.train_f5tts(
            model_name="pl_custom_hyperparams",
            epochs=20,
            batch_size=8,
            learning_rate=0.00005,
            sample_ids=[self.sample_id],
        )
        self.assertEqual(resp.status_code, 200)

    def test_f5tts_rejects_empty_model_name(self):
        resp = self.client.train_f5tts(model_name="", epochs=5)
        self.assertEqual(resp.status_code, 400)

    def test_f5tts_rejects_nonexistent_sample_ids(self):
        resp = self.client.train_f5tts(
            model_name="invalid_samples", epochs=5, sample_ids=["non-existent-uuid"]
        )
        self.assertEqual(resp.status_code, 400)


class TestTier1RVCTraining(unittest.TestCase):
    """Feature 8: RVC v2 Dataset Preparation & Training (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        audio = SyntheticAudioGenerator.generate_speech_dummy_wav(duration_s=3.0)
        up_resp = self.client.upload_sample(audio)
        self.sample_id = up_resp.json()["id"]

    def test_start_rvc_training_returns_run_id(self):
        resp = self.client.train_rvc(
            model_name="rvc_voice_beta", epochs=15, sample_rate=40000, sample_ids=[self.sample_id]
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("run_id", data)
        self.assertEqual(data["status"], "started")
        self.assertEqual(data["engine"], "rvc")

    def test_rvc_registers_model_and_index_paths(self):
        self.client.train_rvc(
            model_name="rvc_index_check", epochs=5, sample_ids=[self.sample_id]
        )
        models_resp = self.client.get_models()
        data = models_resp.json()
        matching = [m for m in data["models"] if m["name"] == "rvc_index_check"]
        self.assertEqual(len(matching), 1)
        self.assertTrue(matching[0]["checkpoint_path"].endswith(".pth"))
        self.assertTrue(matching[0]["index_path"].endswith(".index"))

    def test_rvc_supports_standard_40k_sample_rate(self):
        resp = self.client.train_rvc(
            model_name="rvc_40k", epochs=10, sample_rate=40000, sample_ids=[self.sample_id]
        )
        self.assertEqual(resp.status_code, 200)

    def test_rvc_rejects_invalid_sample_rate(self):
        resp = self.client.train_rvc(
            model_name="rvc_bad_sr", epochs=10, sample_rate=22050, sample_ids=[self.sample_id]
        )
        self.assertEqual(resp.status_code, 400)

    def test_stop_training_cancels_active_run(self):
        self.client.train_rvc(model_name="rvc_to_cancel", epochs=20, sample_ids=[self.sample_id])
        stop_resp = self.client.stop_training()
        self.assertEqual(stop_resp.status_code, 200)
        self.assertEqual(stop_resp.json()["status"], "stopped")


class TestTier1TTSInference(unittest.TestCase):
    """Feature 9: F5-TTS Speech Synthesis (TTS) (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        # Setup sample and model
        audio = SyntheticAudioGenerator.generate_speech_dummy_wav(duration_s=2.0)
        up_resp = self.client.upload_sample(audio)
        self.sample_id = up_resp.json()["id"]
        self.model_name = "f5_tts_inference_voice"
        self.client.train_f5tts(model_name=self.model_name, epochs=2, sample_ids=[self.sample_id])

    def test_tts_generation_returns_generation_id(self):
        resp = self.client.generate_tts(
            text="Witaj świecie, to jest synteza mowy.",
            model_name=self.model_name,
            reference_sample_id=self.sample_id,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("generation_id", data)
        self.assertIn("audio_url", data)
        self.assertIn("filename", data)

    def test_tts_generation_rejects_empty_text(self):
        resp = self.client.generate_tts(
            text="   ",
            model_name=self.model_name,
            reference_sample_id=self.sample_id,
        )
        self.assertEqual(resp.status_code, 400)

    def test_tts_generation_rejects_nonexistent_model(self):
        resp = self.client.generate_tts(
            text="Test",
            model_name="non_existent_model",
            reference_sample_id=self.sample_id,
        )
        self.assertEqual(resp.status_code, 404)

    def test_tts_generated_audio_is_playable_via_stream(self):
        gen_resp = self.client.generate_tts(
            text="Polskie zdanie testowe.",
            model_name=self.model_name,
            reference_sample_id=self.sample_id,
        )
        filename = gen_resp.json()["filename"]
        stream_resp = self.client.get_generated_audio(filename)
        self.assertEqual(stream_resp.status_code, 200)
        self.assertEqual(stream_resp.headers.get("Content-Type"), "audio/wav")
        self.assertTrue(len(stream_resp.content) > 1000)

    def test_tts_generated_audio_is_downloadable(self):
        gen_resp = self.client.generate_tts(
            text="Kolejna próba syntezy.",
            model_name=self.model_name,
            reference_sample_id=self.sample_id,
        )
        filename = gen_resp.json()["filename"]
        dl_resp = self.client.download_generated_audio(filename)
        self.assertEqual(dl_resp.status_code, 200)
        self.assertIn("attachment", dl_resp.headers.get("Content-Disposition", ""))


class TestTier1STSInference(unittest.TestCase):
    """Feature 10: RVC v2 Voice Conversion (STS) (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        # Setup sample and model
        audio = SyntheticAudioGenerator.generate_speech_dummy_wav(duration_s=3.0)
        up_resp = self.client.upload_sample(audio)
        self.sample_id = up_resp.json()["id"]
        self.model_name = "rvc_inference_voice"
        self.client.train_rvc(model_name=self.model_name, epochs=2, sample_ids=[self.sample_id])

    def test_sts_generation_returns_generation_id(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=0,
            index_rate=0.75,
        )
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIn("generation_id", data)
        self.assertIn("filename", data)

    def test_sts_supports_positive_pitch_shift(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=12,
            index_rate=0.5,
        )
        self.assertEqual(resp.status_code, 200)

    def test_sts_supports_negative_pitch_shift(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=-12,
            index_rate=0.5,
        )
        self.assertEqual(resp.status_code, 200)

    def test_sts_rejects_out_of_range_pitch_shift(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=30,  # Exceeds +24 limit
            index_rate=0.75,
        )
        self.assertEqual(resp.status_code, 400)

    def test_sts_generated_audio_stream_and_download(self):
        gen_resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=2,
            index_rate=0.8,
        )
        filename = gen_resp.json()["filename"]
        # Verify stream
        stream_resp = self.client.get_generated_audio(filename)
        self.assertEqual(stream_resp.status_code, 200)
        # Verify download
        dl_resp = self.client.download_generated_audio(filename)
        self.assertEqual(dl_resp.status_code, 200)
        self.assertIn(filename, dl_resp.headers.get("Content-Disposition", ""))


class TestTier1ModelRegistry(unittest.TestCase):
    """Feature 11: Model Registry & Checkpoint Query (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        audio = SyntheticAudioGenerator.generate_sine_wav(duration_s=1.0)
        up_resp = self.client.upload_sample(audio)
        self.sample_id = up_resp.json()["id"]

    def test_registry_starts_empty(self):
        resp = self.client.get_models()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {"models": []})

    def test_registry_lists_f5tts_model(self):
        self.client.train_f5tts("model_reg_1", epochs=2, sample_ids=[self.sample_id])
        resp = self.client.get_models()
        models = resp.json()["models"]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "model_reg_1")
        self.assertEqual(models[0]["engine"], "f5-tts")

    def test_registry_lists_rvc_model(self):
        self.client.train_rvc("model_reg_2", epochs=2, sample_ids=[self.sample_id])
        resp = self.client.get_models()
        models = resp.json()["models"]
        self.assertEqual(len(models), 1)
        self.assertEqual(models[0]["name"], "model_reg_2")
        self.assertEqual(models[0]["engine"], "rvc")

    def test_registry_lists_both_engines_concurrently(self):
        self.client.train_f5tts("m_f5", epochs=2, sample_ids=[self.sample_id])
        self.client.train_rvc("m_rvc", epochs=2, sample_ids=[self.sample_id])
        resp = self.client.get_models()
        names = [m["name"] for m in resp.json()["models"]]
        self.assertIn("m_f5", names)
        self.assertIn("m_rvc", names)

    def test_model_entry_contains_valid_checkpoint_paths(self):
        self.client.train_f5tts("m_paths", epochs=2, sample_ids=[self.sample_id])
        models = self.client.get_models().json()["models"]
        target = [m for m in models if m["name"] == "m_paths"][0]
        self.assertTrue(target["checkpoint_path"].startswith("data/models/"))


class TestTier1GeneratedAudioPlayback(unittest.TestCase):
    """Feature 12: Generated Audio Playback & Download (>=5 tests)"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        audio = SyntheticAudioGenerator.generate_speech_dummy_wav(duration_s=2.0)
        up_resp = self.client.upload_sample(audio)
        self.sample_id = up_resp.json()["id"]
        self.client.train_f5tts("gen_playback_voice", epochs=1, sample_ids=[self.sample_id])
        gen_resp = self.client.generate_tts(
            text="Głos generowany do testów odtwarzania.",
            model_name="gen_playback_voice",
            reference_sample_id=self.sample_id,
        )
        self.filename = gen_resp.json()["filename"]

    def test_generated_audio_stream_status_200(self):
        resp = self.client.get_generated_audio(self.filename)
        self.assertEqual(resp.status_code, 200)

    def test_generated_audio_content_type_is_wav(self):
        resp = self.client.get_generated_audio(self.filename)
        self.assertEqual(resp.headers.get("Content-Type"), "audio/wav")

    def test_generated_audio_supports_byte_range_request(self):
        resp = self.client.get_generated_audio(self.filename, byte_range="bytes=0-100")
        self.assertEqual(resp.status_code, 206)
        self.assertEqual(len(resp.content), 101)

    def test_generated_audio_download_contains_filename_header(self):
        resp = self.client.download_generated_audio(self.filename)
        self.assertEqual(resp.status_code, 200)
        disposition = resp.headers.get("Content-Disposition", "")
        self.assertIn("attachment", disposition)
        self.assertIn(self.filename, disposition)

    def test_generated_audio_content_length_matches_bytes(self):
        resp = self.client.download_generated_audio(self.filename)
        self.assertEqual(int(resp.headers.get("Content-Length", 0)), len(resp.content))


if __name__ == "__main__":
    unittest.main()
