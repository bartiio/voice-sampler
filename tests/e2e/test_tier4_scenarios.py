"""
Tier 4: Real-World Polish Voice Cloning Scenarios
Validates realistic end-to-end user workflows:
1. Polish tongue-twister phonetic calibration with diacritics (Zażółć gęślą jaźń / Chrząszcz)
2. Polish singing voice conversion with musical pitch transposition
3. Technical & mixed Polish-English terminology
4. Conversational interjections and rapid dialogue synthesis
5. Long-form literary prose synthesis (Pan Tadeusz)
6. Multi-format heterogeneous sample dataset ingestion (WAV, MP3, FLAC)
"""

import unittest
from tests.e2e.harness import (
    EmbeddedMockServer,
    SyntheticAudioGenerator,
    VoiceSamplerTestClient,
)


class TestTier4PolishVoiceCloningScenarios(unittest.TestCase):
    """Real-world Polish language voice cloning scenarios"""

    def setUp(self):
        self.server = EmbeddedMockServer()
        self.client = VoiceSamplerTestClient(app=self.server.get_app())
        self.gen = SyntheticAudioGenerator()

    def test_scenario_polish_diacritic_pangram_and_tonguetwister(self):
        """
        Scenario 1: Polish tongue-twister and pangram validation.
        Exercises all 9 Polish diacritics (ą, ć, ę, ł, ń, ó, ś, ź, ż) in uppercase and lowercase.
        """
        # Upload reference sample
        ref_audio = self.gen.generate_speech_dummy_wav(duration_s=3.5)
        up_resp = self.client.upload_sample(ref_audio, filename="szczebrzeszyn.wav")
        sid = up_resp.json()["id"]

        # Edit transcript with classic tongue twister
        tongue_twister = "W Szczebrzeszynie chrząszcz brzmi w trzcinie, a chrabąszcz brzmi w trzcinie."
        self.client.update_sample_transcript(sid, tongue_twister)

        # Train F5-TTS model on this phonetic dataset
        model_name = "szczebrzeszyn_model"
        train_resp = self.client.train_f5tts(
            model_name=model_name, epochs=3, sample_ids=[sid]
        )
        self.assertEqual(train_resp.status_code, 200)

        # Synthesize full Polish pangram with all diacritics
        pangram = "Pchnąć w tę łódź jeża lub ośm skrzyń fig (Zażółć gęślą jaźń)."
        gen_resp = self.client.generate_tts(
            text=pangram, model_name=model_name, reference_sample_id=sid
        )
        self.assertEqual(gen_resp.status_code, 200)
        filename = gen_resp.json()["filename"]

        # Verify output audio file is playable and non-empty
        audio_resp = self.client.get_generated_audio(filename)
        self.assertEqual(audio_resp.status_code, 200)
        self.assertTrue(len(audio_resp.content) > 2000)

    def test_scenario_singing_voice_conversion_with_musical_pitch_shift(self):
        """
        Scenario 2: Singing voice conversion using RVC v2.
        Takes a recorded singing sample and shifts pitch by +4 semitones (major third)
        with high index rate (0.85) to preserve vocal resonance.
        """
        # Record melodic sample
        vocal_audio = self.gen.generate_sine_wav(duration_s=4.0, frequency=261.63)  # Middle C (C4)
        rec_resp = self.client.record_sample(vocal_audio, filename="sung_vocal_c4.wav")
        sid = rec_resp.json()["id"]

        # Train RVC v2 model for the singer
        singer_model = "polish_singer_soprano"
        self.client.train_rvc(
            model_name=singer_model, epochs=5, sample_rate=40000, sample_ids=[sid]
        )

        # Transpose up by 4 semitones to E4 (approx 329.6 Hz)
        sts_resp = self.client.generate_sts(
            input_sample_id=sid,
            model_name=singer_model,
            pitch_shift=4,
            index_rate=0.85,
        )
        self.assertEqual(sts_resp.status_code, 200)
        out_filename = sts_resp.json()["filename"]

        # Verify playable stream
        stream = self.client.get_generated_audio(out_filename)
        self.assertEqual(stream.status_code, 200)
        self.assertEqual(stream.headers.get("Content-Type"), "audio/wav")

    def test_scenario_mixed_polish_english_technical_vocabulary(self):
        """
        Scenario 3: Mixed language and technical jargon in cloned voice.
        Validates pronunciation text containing Polish syntax + English AI terminology.
        """
        sid = self.client.upload_sample(self.gen.generate_speech_dummy_wav(3.0)).json()["id"]
        model_name = "tech_voice_model"
        self.client.train_f5tts(model_name=model_name, epochs=2, sample_ids=[sid])

        mixed_prompt = (
            "Nasz algorytm Voice Sampler wykorzystuje deep learning, transformer DiT "
            "oraz model RVC v2 do precyzyjnego speech-to-speech conversion."
        )
        gen_resp = self.client.generate_tts(
            text=mixed_prompt, model_name=model_name, reference_sample_id=sid
        )
        self.assertEqual(gen_resp.status_code, 200)
        self.assertTrue(gen_resp.json()["filename"].endswith(".wav"))

    def test_scenario_rapid_conversational_dialogue_synthesis(self):
        """
        Scenario 4: Sequential rapid dialogue synthesis.
        Generates 4 distinct dialogue utterances and verifies asset isolation.
        """
        sid = self.client.upload_sample(self.gen.generate_sine_wav(2.0)).json()["id"]
        model_name = "dialogue_actor"
        self.client.train_f5tts(model_name=model_name, epochs=2, sample_ids=[sid])

        dialogue_lines = [
            "Cześć, jak się dziś czujesz?",
            "Wszystko w najlepszym porządku, dziękuję!",
            "Czy nagranie jest gotowe do eksportu?",
            "Tak, pliki WAV są już wygenerowane na dysku.",
        ]

        generated_files = []
        for line in dialogue_lines:
            resp = self.client.generate_tts(
                text=line, model_name=model_name, reference_sample_id=sid
            )
            self.assertEqual(resp.status_code, 200)
            fname = resp.json()["filename"]
            generated_files.append(fname)

        # Ensure all filenames are unique
        self.assertEqual(len(set(generated_files)), len(dialogue_lines))

        # Ensure each file can be streamed independently
        for fname in generated_files:
            stream_resp = self.client.get_generated_audio(fname)
            self.assertEqual(stream_resp.status_code, 200)

    def test_scenario_long_form_polish_literary_prose(self):
        """
        Scenario 5: Synthesis of long-form Polish literary prose (Pan Tadeusz).
        Validates handling of commas, semicolons, exclamations, and archaic cadence.
        """
        sid = self.client.upload_sample(self.gen.generate_speech_dummy_wav(3.0)).json()["id"]
        model_name = "narrator_polish"
        self.client.train_f5tts(model_name=model_name, epochs=2, sample_ids=[sid])

        prose = (
            "Litwo! Ojczyzno moja! ty jesteś jak zdrowie. "
            "Ile cię trzeba cenić, ten tylko się dowie, kto cię stracił. "
            "Dziś piękność twą w całej ozdobie widzę i opisuję, bo tęsknię po tobie."
        )
        resp = self.client.generate_tts(
            text=prose, model_name=model_name, reference_sample_id=sid
        )
        self.assertEqual(resp.status_code, 200)
        fname = resp.json()["filename"]

        dl_resp = self.client.download_generated_audio(fname)
        self.assertEqual(dl_resp.status_code, 200)
        self.assertIn("attachment", dl_resp.headers.get("Content-Disposition", ""))

    def test_scenario_heterogeneous_format_dataset_preparation(self):
        """
        Scenario 6: Ingesting a heterogeneous mix of audio formats:
        WAV (lossless), MP3 (compressed broadcast), FLAC (lossless archive).
        All combined into a single training run.
        """
        wav_bytes = self.gen.generate_sine_wav(2.0)
        mp3_bytes = self.gen.generate_mock_mp3()
        flac_bytes = self.gen.generate_mock_flac()

        s_wav = self.client.upload_sample(wav_bytes, filename="studio.wav", content_type="audio/wav").json()["id"]
        s_mp3 = self.client.upload_sample(mp3_bytes, filename="interview.mp3", content_type="audio/mpeg").json()["id"]
        s_flac = self.client.upload_sample(flac_bytes, filename="archive.flac", content_type="audio/flac").json()["id"]

        # Train single unified model on all 3 formats
        model_name = "unified_tri_format_model"
        train_resp = self.client.train_f5tts(
            model_name=model_name, epochs=3, sample_ids=[s_wav, s_mp3, s_flac]
        )
        self.assertEqual(train_resp.status_code, 200)

        # Verify model registered
        models = self.client.get_models().json()["models"]
        self.assertTrue(any(m["name"] == model_name for m in models))


if __name__ == "__main__":
    unittest.main()
