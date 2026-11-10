"""
Tier 2: Boundary & Corner Cases E2E Tests
Validates boundary values, extreme parameters, non-CUDA fallbacks,
missing resources, and adversarial inputs across all interfaces.
"""

import unittest
from tests.e2e.harness import (
    EmbeddedMockServer,
    SyntheticAudioGenerator,
    VoiceSamplerTestClient,
)


class TestTier2EmptyInputs(unittest.TestCase):
    """Empty and blank input boundary conditions"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        self.valid_audio = SyntheticAudioGenerator.generate_sine_wav(duration_s=1.0)
        up_resp = self.client.upload_sample(self.valid_audio)
        self.sample_id = up_resp.json()["id"]
        self.client.train_f5tts(model_name="f5_empty_test_model", epochs=1, sample_ids=[self.sample_id])

    def test_upload_empty_zero_byte_file_returns_400(self):
        resp = self.client.upload_sample(b"", filename="empty.wav")
        self.assertEqual(resp.status_code, 400)

    def test_record_empty_zero_byte_buffer_returns_400(self):
        resp = self.client.record_sample(b"", filename="empty_rec.wav")
        self.assertEqual(resp.status_code, 400)

    def test_tts_empty_text_returns_400(self):
        resp = self.client.generate_tts(
            text="", model_name="f5_empty_test_model", reference_sample_id=self.sample_id
        )
        self.assertEqual(resp.status_code, 400)

    def test_tts_whitespace_only_text_returns_400(self):
        resp = self.client.generate_tts(
            text="   \t\n   ",
            model_name="f5_empty_test_model",
            reference_sample_id=self.sample_id,
        )
        self.assertEqual(resp.status_code, 400)

    def test_f5tts_train_empty_model_name_returns_400(self):
        resp = self.client.train_f5tts(model_name="   ", epochs=5, sample_ids=[self.sample_id])
        self.assertEqual(resp.status_code, 400)

    def test_rvc_train_empty_model_name_returns_400(self):
        resp = self.client.train_rvc(model_name="   ", epochs=5, sample_ids=[self.sample_id])
        self.assertEqual(resp.status_code, 400)


class TestTier2MissingResources(unittest.TestCase):
    """Non-existent resources, 404 responses, and missing files"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        self.valid_audio = SyntheticAudioGenerator.generate_sine_wav(duration_s=1.0)
        up_resp = self.client.upload_sample(self.valid_audio)
        self.sample_id = up_resp.json()["id"]

    def test_update_nonexistent_sample_returns_404(self):
        resp = self.client.update_sample_transcript("missing-id-9999", "Tekst")
        self.assertEqual(resp.status_code, 404)

    def test_delete_nonexistent_sample_returns_404(self):
        resp = self.client.delete_sample("missing-id-8888")
        self.assertEqual(resp.status_code, 404)

    def test_stream_nonexistent_audio_returns_404(self):
        resp = self.client.get_sample_audio("missing-id-7777")
        self.assertEqual(resp.status_code, 404)

    def test_tts_nonexistent_model_returns_404(self):
        resp = self.client.generate_tts(
            text="Test", model_name="phantom_model", reference_sample_id=self.sample_id
        )
        self.assertEqual(resp.status_code, 404)

    def test_tts_nonexistent_reference_sample_returns_404(self):
        self.client.train_f5tts(model_name="m1", epochs=1, sample_ids=[self.sample_id])
        resp = self.client.generate_tts(
            text="Test", model_name="m1", reference_sample_id="missing-ref-sample"
        )
        self.assertEqual(resp.status_code, 404)

    def test_sts_nonexistent_model_returns_404(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id, model_name="phantom_rvc"
        )
        self.assertEqual(resp.status_code, 404)

    def test_sts_nonexistent_input_sample_returns_404(self):
        self.client.train_rvc(model_name="rvc_m1", epochs=1, sample_ids=[self.sample_id])
        resp = self.client.generate_sts(
            input_sample_id="missing-input-sample", model_name="rvc_m1"
        )
        self.assertEqual(resp.status_code, 404)

    def test_stream_nonexistent_generated_audio_returns_404(self):
        resp = self.client.get_generated_audio("non_existent_gen.wav")
        self.assertEqual(resp.status_code, 404)

    def test_download_nonexistent_generated_audio_returns_404(self):
        resp = self.client.download_generated_audio("non_existent_gen.wav")
        self.assertEqual(resp.status_code, 404)


class TestTier2PitchAndIndexBoundaries(unittest.TestCase):
    """Extreme pitch shift (-24..+24) and index rate (0.0..1.0) boundaries"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        self.valid_audio = SyntheticAudioGenerator.generate_speech_dummy_wav(duration_s=2.0)
        up_resp = self.client.upload_sample(self.valid_audio)
        self.sample_id = up_resp.json()["id"]
        self.model_name = "rvc_limits_model"
        self.client.train_rvc(model_name=self.model_name, epochs=1, sample_ids=[self.sample_id])

    def test_pitch_shift_minimum_boundary_negative_24(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=-24,
            index_rate=0.5,
        )
        self.assertEqual(resp.status_code, 200)

    def test_pitch_shift_maximum_boundary_positive_24(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=24,
            index_rate=0.5,
        )
        self.assertEqual(resp.status_code, 200)

    def test_pitch_shift_underflow_negative_25_returns_400(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=-25,
            index_rate=0.5,
        )
        self.assertEqual(resp.status_code, 400)

    def test_pitch_shift_overflow_positive_25_returns_400(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=25,
            index_rate=0.5,
        )
        self.assertEqual(resp.status_code, 400)

    def test_index_rate_zero_boundary_model_weights_only(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=0,
            index_rate=0.0,
        )
        self.assertEqual(resp.status_code, 200)

    def test_index_rate_one_boundary_pure_retrieval(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=0,
            index_rate=1.0,
        )
        self.assertEqual(resp.status_code, 200)

    def test_index_rate_negative_underflow_returns_400(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=0,
            index_rate=-0.01,
        )
        self.assertEqual(resp.status_code, 400)

    def test_index_rate_overflow_returns_400(self):
        resp = self.client.generate_sts(
            input_sample_id=self.sample_id,
            model_name=self.model_name,
            pitch_shift=0,
            index_rate=1.05,
        )
        self.assertEqual(resp.status_code, 400)


class TestTier2NonCUDAHandling(unittest.TestCase):
    """Non-CUDA host behavior, simulation mode, and diagnostic warnings"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        self.valid_audio = SyntheticAudioGenerator.generate_sine_wav(duration_s=1.0)
        up_resp = self.client.upload_sample(self.valid_audio)
        self.sample_id = up_resp.json()["id"]

    def test_gpu_endpoint_handles_missing_cuda_cleanly(self):
        resp = self.client.get_gpu_settings()
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        if not data["cuda_available"]:
            self.assertFalse(data["cuda_available"])
            self.assertIsNone(data["cuda_version"])
            self.assertTrue(data["simulation_mode"])
            self.assertIn("gpu is required", data["message"].lower())

    def test_f5tts_train_completes_in_simulation_mode_without_crash(self):
        resp = self.client.train_f5tts(
            model_name="sim_f5_model", epochs=2, sample_ids=[self.sample_id]
        )
        self.assertEqual(resp.status_code, 200)
        models = self.client.get_models().json()["models"]
        self.assertTrue(any(m["name"] == "sim_f5_model" for m in models))

    def test_rvc_train_completes_in_simulation_mode_without_crash(self):
        resp = self.client.train_rvc(
            model_name="sim_rvc_model", epochs=2, sample_ids=[self.sample_id]
        )
        self.assertEqual(resp.status_code, 200)
        models = self.client.get_models().json()["models"]
        self.assertTrue(any(m["name"] == "sim_rvc_model" for m in models))


class TestTier2AudioFormatsAndByteRanges(unittest.TestCase):
    """Audio formats, streaming byte-ranges, and corrupt stream boundary tests"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        self.audio = SyntheticAudioGenerator.generate_speech_dummy_wav(duration_s=2.0)
        up_resp = self.client.upload_sample(self.audio, filename="range_boundary.wav")
        self.sample_id = up_resp.json()["id"]

    def test_upload_mp3_dummy_container_succeeds(self):
        mp3_data = SyntheticAudioGenerator.generate_mock_mp3()
        resp = self.client.upload_sample(mp3_data, filename="voice.mp3", content_type="audio/mpeg")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["filename"], "voice.mp3")

    def test_upload_flac_dummy_container_succeeds(self):
        flac_data = SyntheticAudioGenerator.generate_mock_flac()
        resp = self.client.upload_sample(flac_data, filename="voice.flac", content_type="audio/flac")
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["filename"], "voice.flac")

    def test_byte_range_out_of_bounds_returns_416(self):
        resp = self.client.get_sample_audio(self.sample_id, byte_range="bytes=99999999-999999999")
        self.assertEqual(resp.status_code, 416)

    def test_byte_range_reversed_range_returns_416(self):
        resp = self.client.get_sample_audio(self.sample_id, byte_range="bytes=500-200")
        self.assertEqual(resp.status_code, 416)

    def test_training_hyperparameters_zero_epochs_rejected(self):
        resp = self.client.train_f5tts(
            model_name="bad_epoch", epochs=0, sample_ids=[self.sample_id]
        )
        self.assertEqual(resp.status_code, 400)

    def test_training_hyperparameters_negative_epochs_rejected(self):
        resp = self.client.train_rvc(
            model_name="bad_epoch", epochs=-5, sample_ids=[self.sample_id]
        )
        self.assertEqual(resp.status_code, 400)

    def test_training_hyperparameters_negative_learning_rate_rejected(self):
        resp = self.client.train_f5tts(
            model_name="bad_lr", epochs=5, learning_rate=-0.001, sample_ids=[self.sample_id]
        )
        self.assertEqual(resp.status_code, 400)

    def test_stop_training_when_idle_returns_no_active_run(self):
        resp = self.client.stop_training()
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["status"], "no_active_run")


if __name__ == "__main__":
    unittest.main()
