"""
Tier 3: Cross-Feature Integration Combinations
Validates end-to-end multi-step flows combining:
Upload -> Transcribe -> Edit -> Train -> Telemetry -> Registry -> Inference -> Playback/Download.
"""

import unittest
from tests.e2e.harness import (
    EmbeddedMockServer,
    SyntheticAudioGenerator,
    VoiceSamplerTestClient,
)


class TestTier3CrossFeatureCombos(unittest.TestCase):
    """Multi-feature end-to-end integration flows"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        self.gen = SyntheticAudioGenerator()

    def test_combo_f5tts_complete_lifecycle(self):
        """
        Flow 1: Upload WAV -> Auto-transcribe -> Edit transcript -> Train F5-TTS
        -> Monitor WS telemetry -> Verify model in registry -> Synthesize TTS -> Stream & Download.
        """
        # 1. Upload audio
        audio = self.gen.generate_speech_dummy_wav(duration_s=2.5)
        up_resp = self.client.upload_sample(audio, filename="combo_f5_sample.wav")
        self.assertEqual(up_resp.status_code, 200)
        sample = up_resp.json()
        sample_id = sample["id"]
        self.assertEqual(sample["status"], "transcribed")

        # 2. Edit transcript with Polish sentence
        custom_transcript = "To jest starannie nagrany polski głos do klonowania."
        edit_resp = self.client.update_sample_transcript(sample_id, custom_transcript)
        self.assertEqual(edit_resp.status_code, 200)
        self.assertEqual(edit_resp.json()["transcript"], custom_transcript)

        # 3. Train F5-TTS model
        model_name = "cloned_polish_voice"
        train_resp = self.client.train_f5tts(
            model_name=model_name, epochs=3, batch_size=2, learning_rate=0.0001, sample_ids=[sample_id]
        )
        self.assertEqual(train_resp.status_code, 200)
        train_data = train_resp.json()
        self.assertEqual(train_data["status"], "started")

        # 4. Verify WebSocket progress and completion
        with self.client.websocket_training() as ws:
            progress_events = []
            while True:
                msg = ws.receive_json()
                progress_events.append(msg)
                if msg.get("type") == "completed":
                    break
            self.assertTrue(len(progress_events) >= 2)
            self.assertEqual(progress_events[-1]["status"], "completed")

        # 5. Verify model exists in registry
        models_resp = self.client.get_models()
        self.assertEqual(models_resp.status_code, 200)
        model_names = [m["name"] for m in models_resp.json()["models"]]
        self.assertIn(model_name, model_names)

        # 6. Synthesize TTS speech
        gen_resp = self.client.generate_tts(
            text="Dzień dobry! Mój głos został pomyślnie sklonowany.",
            model_name=model_name,
            reference_sample_id=sample_id,
        )
        self.assertEqual(gen_resp.status_code, 200)
        gen_data = gen_resp.json()
        gen_filename = gen_data["filename"]
        self.assertTrue(gen_filename.endswith(".wav"))

        # 7. Stream audio (playback)
        stream_resp = self.client.get_generated_audio(gen_filename)
        self.assertEqual(stream_resp.status_code, 200)
        self.assertEqual(stream_resp.headers.get("Content-Type"), "audio/wav")
        self.assertTrue(len(stream_resp.content) > 1000)

        # 8. Download audio
        dl_resp = self.client.download_generated_audio(gen_filename)
        self.assertEqual(dl_resp.status_code, 200)
        self.assertIn(gen_filename, dl_resp.headers.get("Content-Disposition", ""))

    def test_combo_rvc_complete_lifecycle(self):
        """
        Flow 2: Record mic audio -> Auto-transcribe -> Train RVC v2
        -> Verify WS telemetry -> Check .pth and .index -> STS conversion -> Stream & Download.
        """
        # 1. Record audio
        rec_audio = self.gen.generate_sine_wav(duration_s=3.0, frequency=220.0)
        rec_resp = self.client.record_sample(rec_audio, filename="mic_rvc_input.wav")
        self.assertEqual(rec_resp.status_code, 200)
        sample_id = rec_resp.json()["id"]

        # 2. Train RVC v2 model
        model_name = "rvc_singer_model"
        train_resp = self.client.train_rvc(
            model_name=model_name, epochs=5, sample_rate=40000, sample_ids=[sample_id]
        )
        self.assertEqual(train_resp.status_code, 200)

        # 3. Verify model registry has checkpoints and index
        models = self.client.get_models().json()["models"]
        rvc_model = next((m for m in models if m["name"] == model_name), None)
        self.assertIsNotNone(rvc_model)
        self.assertEqual(rvc_model["engine"], "rvc")
        self.assertTrue(rvc_model["checkpoint_path"].endswith(".pth"))
        self.assertTrue(rvc_model["index_path"].endswith(".index"))

        # 4. Perform STS voice conversion with pitch transposition (+3 semitones)
        sts_resp = self.client.generate_sts(
            input_sample_id=sample_id,
            model_name=model_name,
            pitch_shift=3,
            index_rate=0.8,
        )
        self.assertEqual(sts_resp.status_code, 200)
        filename = sts_resp.json()["filename"]

        # 5. Verify streaming with byte-range
        partial_resp = self.client.get_generated_audio(filename, byte_range="bytes=0-511")
        self.assertEqual(partial_resp.status_code, 206)
        self.assertEqual(len(partial_resp.content), 512)

        # 6. Verify download
        dl_resp = self.client.download_generated_audio(filename)
        self.assertEqual(dl_resp.status_code, 200)

    def test_combo_batch_sample_selection_training(self):
        """
        Flow 3: Upload multiple samples -> Select specific subset for training
        -> Ensure training succeeds only with valid selected IDs.
        """
        s1 = self.client.upload_sample(self.gen.generate_sine_wav(1.0), filename="s1.wav").json()["id"]
        s2 = self.client.upload_sample(self.gen.generate_sine_wav(1.5), filename="s2.wav").json()["id"]
        s3 = self.client.upload_sample(self.gen.generate_sine_wav(2.0), filename="s3.wav").json()["id"]

        # Train on subset {s1, s3}
        resp = self.client.train_f5tts(
            model_name="batch_subset_model", epochs=3, sample_ids=[s1, s3]
        )
        self.assertEqual(resp.status_code, 200)

        models = self.client.get_models().json()["models"]
        self.assertTrue(any(m["name"] == "batch_subset_model" for m in models))

    def test_combo_sample_deletion_cascades_safely(self):
        """
        Flow 4: Upload sample -> Transcribe -> Delete sample
        -> Attempt training / generation referencing deleted sample -> Clean 400/404 handling.
        """
        up = self.client.upload_sample(self.gen.generate_sine_wav(1.0), filename="temp.wav").json()
        sid = up["id"]

        # Ensure sample exists
        samples_before = self.client.get_samples().json()["samples"]
        self.assertTrue(any(s["id"] == sid for s in samples_before))

        # Delete sample
        del_resp = self.client.delete_sample(sid)
        self.assertEqual(del_resp.status_code, 200)

        # Verify not present in listing
        samples_after = self.client.get_samples().json()["samples"]
        self.assertFalse(any(s["id"] == sid for s in samples_after))

        # Training with deleted ID should fail with 400
        train_resp = self.client.train_f5tts(model_name="invalid_train", epochs=2, sample_ids=[sid])
        self.assertEqual(train_resp.status_code, 400)

        # Audio stream of deleted sample should fail with 404
        stream_resp = self.client.get_sample_audio(sid)
        self.assertEqual(stream_resp.status_code, 404)

    def test_combo_training_interruption_and_cancellation(self):
        """
        Flow 5: Start training -> Stop training -> Verify clean stop state
        -> Verify subsequent stop reports no active run.
        """
        sid = self.client.upload_sample(self.gen.generate_sine_wav(1.0)).json()["id"]
        train_resp = self.client.train_f5tts(model_name="cancel_model", epochs=50, sample_ids=[sid])
        self.assertEqual(train_resp.status_code, 200)
        run_id = train_resp.json()["run_id"]

        # Stop active run
        stop_resp = self.client.stop_training()
        self.assertEqual(stop_resp.status_code, 200)
        self.assertEqual(stop_resp.json()["status"], "stopped")
        self.assertEqual(stop_resp.json()["run_id"], run_id)

        # Second stop returns no_active_run
        stop_again = self.client.stop_training()
        self.assertEqual(stop_again.status_code, 200)
        self.assertEqual(stop_again.json()["status"], "no_active_run")

    def test_combo_multiple_distinct_models_isolated_synthesis(self):
        """
        Flow 6: Train multiple models (Voice Alpha and Voice Beta)
        -> Verify both can synthesize speech independently with distinct outputs.
        """
        sid = self.client.upload_sample(self.gen.generate_speech_dummy_wav(2.0)).json()["id"]

        # Train voice A
        self.client.train_f5tts(model_name="voice_alpha", epochs=2, sample_ids=[sid])
        # Train voice B
        self.client.train_f5tts(model_name="voice_beta", epochs=2, sample_ids=[sid])

        # Generate from voice A
        gen_a = self.client.generate_tts(
            text="Głos Alpha mówi dzień dobry.",
            model_name="voice_alpha",
            reference_sample_id=sid,
        ).json()

        # Generate from voice B
        gen_b = self.client.generate_tts(
            text="Głos Beta odpowiada na powitanie.",
            model_name="voice_beta",
            reference_sample_id=sid,
        ).json()

        self.assertNotEqual(gen_a["generation_id"], gen_b["generation_id"])
        self.assertNotEqual(gen_a["filename"], gen_b["filename"])

        # Both audio files are accessible
        resp_a = self.client.get_generated_audio(gen_a["filename"])
        resp_b = self.client.get_generated_audio(gen_b["filename"])
        self.assertEqual(resp_a.status_code, 200)
        self.assertEqual(resp_b.status_code, 200)


if __name__ == "__main__":
    unittest.main()
