import os
import unittest
from unittest.mock import patch

os.environ["PRELOAD_MODEL"] = "0"
os.environ.pop("TARGET_WPM_MIN", None)
os.environ.pop("TARGET_WPM_MAX", None)

from fastapi.testclient import TestClient

import main


class ContractTests(unittest.TestCase):
    def test_phonics_uses_word_accuracy_profile(self) -> None:
        context = main.validate_activity_context(
            grade=2,
            subdomain="phonics and word study",
            competency_code="en2pws-i-3",
            activity_type="phonics",
            difficulty="medium",
            reference_text="ship chat thin ring",
        )

        self.assertEqual(context[0], "Phonics and Word Study")
        self.assertEqual(context[1], "EN2PWS-I-3")
        self.assertEqual(context[2], "medium")
        self.assertEqual(context[3], "oral_word_accuracy")

    def test_oral_reading_uses_passage_profile(self) -> None:
        self.assertEqual(
            main.resolve_measurement_profile("oral_reading"),
            "oral_passage_fluency",
        )

    def test_non_speech_activity_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not speech-scored"):
            main.resolve_measurement_profile("multiple_choice")

    def test_grade_subdomain_mismatch_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not available for Grade 3"):
            main.validate_activity_context(
                grade=3,
                subdomain="Phonological Awareness",
                competency_code="TEST-1",
                activity_type="oral_reading",
                difficulty="easy",
                reference_text="A short text.",
            )

    def test_reference_text_is_required(self) -> None:
        with self.assertRaisesRegex(ValueError, "reference_text is required"):
            main.validate_activity_context(
                grade=1,
                subdomain="Phonics and Word Study",
                competency_code="TEST-1",
                activity_type="phonics",
                difficulty="easy",
                reference_text=None,
            )


class ScoringTests(unittest.TestCase):
    def test_raw_wcpm_has_no_default_interpretation(self) -> None:
        speed = main.calculate_speed(
            recognized_word_count=20,
            correct_word_count=18,
            total_seconds=60,
            active_speech_seconds=40,
        )

        self.assertEqual(speed["wcpm"], 18.0)
        self.assertIsNone(speed["speed_score"])
        self.assertEqual(speed["pace_label"], "not_interpreted")
        self.assertIsNone(speed["target_wpm_range"])

    def test_word_profile_excludes_speed_and_prosody(self) -> None:
        result = {
            "transcript": "ship chat thin",
            "word_timestamps": [
                {"word": "ship", "confidence": 0.9},
                {"word": "chat", "confidence": 0.9},
            ],
            "accuracy": {
                "accuracy_score": 75.0,
                "correct": 3,
                "substitutions": 1,
                "deletions": 0,
                "insertions": 0,
            },
            "speed": {"wcpm": 30.0, "speed_score": None},
            "prosody": {
                "duration_seconds": 8.0,
                "active_speech_seconds": 7.0,
                "prosody_score": 80.0,
                "experimental_prosody_indicator": 80.0,
            },
            "notes": {"constrained_vocabulary": False},
        }

        response = main.build_aligned_response(
            result=result,
            activity_id="act-1",
            grade=2,
            subdomain="Phonics and Word Study",
            competency_code="EN2PWS-I-3",
            activity_type="phonics",
            difficulty="medium",
            attempt_number=1,
            profile="oral_word_accuracy",
            comprehension_score=None,
            legacy_proficiency=False,
        )

        self.assertEqual(response["eligible_metrics"], ["word_accuracy"])
        self.assertIsNone(response["measurements"]["wcpm"])
        self.assertIsNone(
            response["measurements"]["experimental_prosody_indicator"]
        )
        self.assertEqual(
            response["reading_proficiency"]["status"],
            "not_calculated",
        )

    def test_quality_flags_low_confidence_and_constrained_mode(self) -> None:
        flags = main.build_quality_flags(
            {
                "transcript": "cat",
                "word_timestamps": [{"word": "cat", "confidence": 0.4}],
                "prosody": {
                    "duration_seconds": 4.0,
                    "active_speech_seconds": 1.0,
                },
                "notes": {"constrained_vocabulary": True},
            }
        )

        self.assertIn("low_asr_confidence", flags)
        self.assertIn("low_active_speech_ratio", flags)
        self.assertIn("constrained_vocabulary_may_inflate_accuracy", flags)


class RouteTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = TestClient(main.app)

    def test_root_reports_v4(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["version"], "4.0.0")

    def test_metadata_exposes_generator_contract(self) -> None:
        response = self.client.get("/metadata")
        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["grades"], [1, 2, 3])
        self.assertEqual(
            payload["measurement_profiles"]["phonics"],
            "oral_word_accuracy",
        )
        self.assertIn("multiple_choice", payload["non_speech_activity_types"])

    def test_analyze_requires_generator_metadata(self) -> None:
        response = self.client.post(
            "/analyze",
            files={"file": ("sample.wav", b"not-a-real-wave", "audio/wav")},
            data={"reference_text": "cat"},
        )
        self.assertEqual(response.status_code, 422)

    def test_analyze_returns_aligned_activity_context(self) -> None:
        fake_result = {
            "model": "test-model",
            "engine": "test",
            "transcript": "the cat sat",
            "word_timestamps": [
                {"word": "the", "confidence": 0.95},
                {"word": "cat", "confidence": 0.95},
                {"word": "sat", "confidence": 0.95},
            ],
            "accuracy": {
                "accuracy_score": 100.0,
                "correct": 3,
                "substitutions": 0,
                "deletions": 0,
                "insertions": 0,
            },
            "speed": {"wcpm": 30.0, "speed_score": None},
            "prosody": {
                "duration_seconds": 6.0,
                "active_speech_seconds": 5.0,
                "prosody_score": 70.0,
                "experimental_prosody_indicator": 70.0,
            },
            "timing": {},
            "notes": {"constrained_vocabulary": False},
        }

        with (
            patch("main.convert_to_wav", return_value=None),
            patch("main.analyze_wav", return_value=fake_result),
        ):
            response = self.client.post(
                "/analyze",
                files={"file": ("sample.webm", b"audio", "audio/webm")},
                data={
                    "activity_id": "act-9",
                    "grade": "3",
                    "subdomain": "Comprehending and Analyzing Text",
                    "competency_code": "EN3CAT-I-1",
                    "activity_type": "oral_reading",
                    "difficulty": "hard",
                    "reference_text": "The cat sat.",
                    "attempt_number": "2",
                },
            )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["activity_context"]["activity_id"], "act-9")
        self.assertEqual(payload["activity_context"]["grade"], 3)
        self.assertEqual(payload["activity_context"]["difficulty"], "hard")
        self.assertEqual(payload["measurement_profile"], "oral_passage_fluency")
        self.assertEqual(payload["measurements"]["wcpm"], 30.0)
        self.assertEqual(
            payload["reading_proficiency"]["status"],
            "not_calculated",
        )

    def test_internal_key_is_enforced_when_configured(self) -> None:
        original_key = main.APP_API_KEY
        main.APP_API_KEY = "test-secret"

        try:
            response = self.client.post(
                "/calculate-proficiency",
                data={
                    "accuracy": "90",
                    "speed": "80",
                    "prosody": "70",
                    "comprehension": "85",
                },
            )
            self.assertEqual(response.status_code, 401)
        finally:
            main.APP_API_KEY = original_key


if __name__ == "__main__":
    unittest.main()
