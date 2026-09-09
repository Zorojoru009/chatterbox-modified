import json
import unittest
from pathlib import Path

from chatterbox.studio.blueprint import (
    chunk_settings_for,
    export_blueprint_from_session,
    make_session_from_blueprint,
    normalize_blueprint_settings,
    validate_narration_blueprint,
)
from chatterbox.studio.config import (
    BLUEPRINT_DEFAULT_SETTINGS,
    MODEL_ORIGINAL,
    MODEL_TURBO,
    safe_name,
    safe_wav_filename,
)
from chatterbox.studio.session import (
    chunk_script,
    format_chunk_table,
    make_session,
    split_sentences_preserving_tags,
)
from chatterbox.studio.validator import (
    build_full_report_markdown,
    build_validation_overview,
    normalize_validation_text,
    validation_comparison,
)
from chatterbox.studio.ui import (
    on_jump_to_problem,
    update_validation_scorecard,
)


class StudioUnitTests(unittest.TestCase):
    def test_safe_name_and_filename(self):
        self.assertEqual(safe_name("My Cool / Project: Name!"), "My Cool - Project Name")
        self.assertEqual(safe_wav_filename("narration"), "narration.wav")
        self.assertEqual(safe_wav_filename("narration.wav"), "narration.wav")

    def test_normalize_validation_text(self):
        raw = "Oh, that's hilarious! [chuckle] Um anyway, [laugh] we do have a new model."
        normalized = normalize_validation_text(raw)
        self.assertNotIn("[chuckle]", normalized)
        self.assertNotIn("[laugh]", normalized)
        self.assertEqual(normalized, "oh that is hilarious anyway we do have a new model")

    def test_validation_comparison_basic(self):
        expected = "Hello world, this is a test. [sigh]"
        transcript = "hello world this is a test"
        res = validation_comparison(expected, transcript, threshold=0.9)
        self.assertTrue(res["passed"])
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(len(res["missing_words"]), 0)
        self.assertEqual(len(res["extra_words"]), 0)
        self.assertEqual(res["diagnostic_category"], "passed")

    def test_validation_resolves_false_negatives(self):
        # Whisper commonly outputs words when numbers are written, or vice versa,
        # plus expands/contracts contractions and omits filler words.
        expected = "I have 1 idea and $50 for the 1st step, so don't worry, um, ok?"
        transcript = "I have one idea and fifty dollars for the first step so do not worry okay"
        res = validation_comparison(expected, transcript, threshold=0.9)
        self.assertTrue(res["passed"])
        self.assertEqual(res["score"], 1.0)
        self.assertEqual(len(res["missing_words"]), 0)
        self.assertEqual(len(res["extra_words"]), 0)
        self.assertEqual(res["diagnostic_category"], "passed")

    def test_validation_numbers_and_symbols_normalization(self):
        # Decimals, percentages, ordinals, symbols
        expected = "We achieved 100% precision on the 2nd attempt with 3.5 seconds & +5 points."
        transcript = "we achieved one hundred percent precision on the second attempt with three point five seconds and plus five points"
        res = validation_comparison(expected, transcript, threshold=0.95)
        self.assertTrue(res["passed"])
        self.assertEqual(res["score"], 1.0)

    def test_validation_diagnostics_cutoff(self):
        expected = "The quick brown fox jumps over the lazy dog and runs away into the deep forest."
        # Audio was cut off early
        cutoff_transcript = "The quick brown fox jumps over the lazy dog"
        res = validation_comparison(expected, cutoff_transcript, threshold=0.8)
        self.assertFalse(res["passed"])
        self.assertEqual(res["diagnostic_category"], "early_cutoff")
        self.assertIn("Audio cut off early", res["diagnostic_message"])
        self.assertIn("and runs away into the deep forest", res["diagnostic_message"])

    def test_validation_diagnostics_repetition(self):
        expected = "Let us proceed to the next stage of our demonstration."
        # TTS model got stuck in repetition loop
        stuck_transcript = "Let us proceed to the next stage stage stage stage stage"
        res = validation_comparison(expected, stuck_transcript, threshold=0.8)
        self.assertFalse(res["passed"])
        self.assertEqual(res["diagnostic_category"], "repetition_loop")
        self.assertIn("Repetition loop detected", res["diagnostic_message"])

    def test_validation_diagnostics_no_speech(self):
        expected = "This sentence should have been spoken."
        blank_transcript = ""
        res = validation_comparison(expected, blank_transcript, threshold=0.8)
        self.assertFalse(res["passed"])
        self.assertEqual(res["diagnostic_category"], "no_speech")
        self.assertEqual(res["diagnostic_message"], "No speech detected in audio.")


    def test_validate_narration_blueprint_valid(self):
        valid_bp = {
            "schema_version": 1,
            "project_name": "test_proj",
            "output_filename": "test.wav",
            "default_model": "Original",
            "default_settings": {"temperature": 0.7},
            "chunks": [
                {"id": "c1", "text": "First chunk.", "note": "Hook"},
                {"id": "c2", "text": "Second chunk.", "settings": {"temperature": 0.5}},
            ],
        }
        self.assertIsNone(validate_narration_blueprint(valid_bp))

    def test_validate_narration_blueprint_invalid(self):
        self.assertIn("JSON object", validate_narration_blueprint("not a dict"))
        self.assertIn("schema_version", validate_narration_blueprint({"schema_version": 99, "chunks": []}))
        self.assertIn("non-empty `chunks`", validate_narration_blueprint({"chunks": []}))
        self.assertIn("missing non-empty `text`", validate_narration_blueprint({"chunks": [{"id": "1"}]}))
        self.assertIn("default_model", validate_narration_blueprint({"default_model": "UnknownModel", "chunks": [{"text": "hi"}]}))
        self.assertIn("single model", validate_narration_blueprint({
            "default_model": "Original",
            "chunks": [{"text": "hi", "model": "Turbo"}],
        }))
        self.assertIn("Duplicate chunk ID", validate_narration_blueprint({
            "chunks": [
                {"id": "dup", "text": "one"},
                {"id": "dup", "text": "two"},
            ]
        }))

    def test_normalize_and_resolve_chunk_settings(self):
        base = {"temperature": 0.6, "exaggeration": 0.4}
        normalized_base = normalize_blueprint_settings(base)
        self.assertEqual(normalized_base["temperature"], 0.6)
        self.assertEqual(normalized_base["exaggeration"], 0.4)
        self.assertEqual(normalized_base["top_p"], BLUEPRINT_DEFAULT_SETTINGS["top_p"])

        chunk_override = {"temperature": 0.9}
        chunk_res = normalize_blueprint_settings(chunk_override, base=normalized_base)
        self.assertEqual(chunk_res["temperature"], 0.9)
        self.assertEqual(chunk_res["exaggeration"], 0.4)  # inherited from base

        # Test chunk_settings_for in a blueprint session
        session = {
            "from_blueprint": True,
            "model_name": "Original",
            "generation_settings": normalized_base,
        }
        chunk = {
            "model_name": "Original",
            "settings": {"temperature": 0.85},
        }
        resolved = chunk_settings_for(session, chunk, ui_settings={"enable_parallel": True, "max_parallel_devices": 2})
        self.assertEqual(resolved["temperature"], 0.85)
        self.assertEqual(resolved["exaggeration"], 0.4)
        self.assertTrue(resolved["enable_parallel"])
        self.assertEqual(resolved["max_parallel_devices"], 2)

    def test_make_session_from_blueprint(self):
        bp = {
            "project_name": "my_video",
            "output_filename": "final_narration.wav",
            "default_model": "Turbo",
            "default_settings": {"temperature": 0.75},
            "merge": {"silence_ms": 250, "export_mp3": True},
            "chunks": [
                {"id": "intro", "text": "Welcome to the channel. [laugh]", "note": "High energy"},
                {"id": "main", "text": "Here is the main topic.", "settings": {"temperature": 0.65}},
            ],
        }
        session = make_session_from_blueprint(bp)
        self.assertTrue(session["from_blueprint"])
        self.assertEqual(session["project_name"], "my_video")
        self.assertEqual(session["model_name"], "Turbo")
        self.assertEqual(session["output_filename"], "final_narration.wav")
        self.assertEqual(len(session["chunks"]), 2)
        self.assertEqual(session["chunks"][0]["id"], "intro")
        self.assertEqual(session["chunks"][0]["note"], "High energy")
        self.assertEqual(session["chunks"][1]["settings"]["temperature"], 0.65)
        self.assertEqual(session["merge_settings"]["silence_ms"], 250)
        self.assertTrue(session["merge_settings"]["export_mp3"])

        # Test table formatting
        table = format_chunk_table(session)
        self.assertEqual(len(table), 2)
        self.assertEqual(table[0][0], "intro")
        self.assertEqual(table[0][3], "High energy")

        # Test export back to blueprint
        exported = export_blueprint_from_session(session)
        self.assertEqual(exported["project_name"], "my_video")
        self.assertEqual(len(exported["chunks"]), 2)
        self.assertEqual(exported["chunks"][0]["id"], "intro")

    def test_script_chunking(self):
        text = (
            "Here is the first sentence that introduces our topic in great depth. "
            "Then we move into the second detailed section of the explanation where we discuss various things. "
            "Finally, we conclude with a summary sentence that wraps up everything neatly."
        )
        chunks = chunk_script(text, target_chars=120)
        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(c) > 0 for c in chunks))

    def test_merge_validation(self):
        # Invalid silence_ms
        bp_bad_silence = {
            "chunks": [{"text": "Hello"}],
            "merge": {"silence_ms": -10},
        }
        self.assertIn("silence_ms", validate_narration_blueprint(bp_bad_silence))

        # Invalid mp3_bitrate
        bp_bad_bitrate = {
            "chunks": [{"text": "Hello"}],
            "merge": {"mp3_bitrate": "999k"},
        }
        self.assertIn("mp3_bitrate", validate_narration_blueprint(bp_bad_bitrate))

    def test_import_narration_blueprint_file(self):
        import tempfile
        from chatterbox.studio.ui import import_narration_blueprint

        bp = {
            "project_name": "temp_test",
            "output_filename": "out.wav",
            "default_model": "Original",
            "chunks": [{"id": "chunk01", "text": "Test chunk text."}],
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(bp, f)
            temp_path = f.name

        try:
            results = import_narration_blueprint(temp_path)
            # Tuple shape check
            self.assertEqual(len(results), 13)
            session = results[0]
            self.assertEqual(session["project_name"], "temp_test")
            self.assertEqual(session["output_filename"], "out.wav")
            self.assertEqual(len(session["chunks"]), 1)
            self.assertEqual(session["chunks"][0]["id"], "chunk01")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_load_session_restores_assets(self):
        import tempfile
        from chatterbox.studio.ui import load_session

        fake_session = {
            "session_id": "test_load_session",
            "project_name": "restored_proj",
            "output_filename": "restored.wav",
            "model_name": "Original",
            "full_text": "Restored text.",
            "reference_audio_path": "/fake/path/voice.wav",
            "chunks": [
                {
                    "index": 1,
                    "id": "c001",
                    "text": "First chunk text.",
                    "status": "generated",
                    "audio_path": "/fake/path/0001.wav",
                    "transcript": "first chunk text",
                    "validation_status": "passed",
                }
            ],
            "final_output_path": "/fake/path/final.wav",
            "session_dir": "/fake/dir",
        }
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(fake_session, f)
            temp_path = f.name

        try:
            results = load_session(temp_path, current_ref_audio="/fallback/voice.wav")
            self.assertEqual(len(results), 20)
            loaded_session = results[0]
            first_text = results[3]
            first_audio = results[4]
            first_transcript = results[5]
            restored_ref_audio = results[15]
            final_wav = results[16]

            self.assertEqual(loaded_session["project_name"], "restored_proj")
            self.assertEqual(first_text, "First chunk text.")
            self.assertEqual(first_audio, "/fake/path/0001.wav")
            self.assertEqual(first_transcript, "first chunk text")
            self.assertEqual(restored_ref_audio, "/fake/path/voice.wav")
            self.assertEqual(final_wav, "/fake/path/final.wav")
        finally:
            Path(temp_path).unlink(missing_ok=True)

    def test_chronological_session_numbering(self):
        import tempfile
        from chatterbox.studio.session import (
            generate_chronological_session_id,
            get_next_session_number,
            list_session_choices,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            # Starts at 1
            self.assertEqual(get_next_session_number(root), 1)

            num1, sid1 = generate_chronological_session_id("video_one", session_root=root)
            self.assertEqual(num1, 1)
            self.assertTrue(sid1.startswith("001_video_one_"))

            # Create dummy session folder 1
            s1_dir = root / sid1
            s1_dir.mkdir()
            (s1_dir / "session.json").write_text(
                json.dumps({"session_number": 1, "project_name": "video_one", "chunks": [{"id": "1"}]}),
                encoding="utf-8",
            )

            # Next should be 2
            self.assertEqual(get_next_session_number(root), 2)
            num2, sid2 = generate_chronological_session_id("video_two", session_root=root)
            self.assertEqual(num2, 2)
            self.assertTrue(sid2.startswith("002_video_two_"))

            # Create dummy session folder 2
            s2_dir = root / sid2
            s2_dir.mkdir()
            (s2_dir / "session.json").write_text(
                json.dumps({"session_number": 2, "project_name": "video_two", "chunks": [{"id": "1"}, {"id": "2"}]}),
                encoding="utf-8",
            )

            # Test choices format
            choices = list_session_choices(session_root=root)
            self.assertEqual(len(choices), 2)
            labels = [c[0] for c in choices]
            self.assertTrue(any(l.startswith("#002") for l in labels))
            self.assertTrue(any(l.startswith("#001") for l in labels))

    def test_model_adapter_voice_conditioning_cache(self):
        import tempfile
        from unittest.mock import MagicMock
        from chatterbox.studio.engine import ModelAdapter

        # Create dummy adapter with mocked model
        adapter = object.__new__(ModelAdapter)
        adapter.model_name = "Turbo"
        adapter.device = "cpu"
        adapter.cached_ref_path = None
        adapter.cached_norm_loudness = None
        mock_model = MagicMock()
        mock_model.conds = None
        adapter.model = mock_model

        with tempfile.NamedTemporaryFile("wb", suffix=".wav") as f:
            ref_path = f.name
            # First call: must run prepare_conditionals
            def fake_prep(*args, **kwargs):
                mock_model.conds = "dummy_conds"
            mock_model.prepare_conditionals.side_effect = fake_prep

            adapter.ensure_conditionals(ref_path, exaggeration=0.5, norm_loudness=True)
            self.assertEqual(mock_model.prepare_conditionals.call_count, 1)
            self.assertEqual(adapter.cached_ref_path, str(Path(ref_path).resolve()))

            # Second call with same ref: must be a NO-OP (0 calls added!)
            adapter.ensure_conditionals(ref_path, exaggeration=0.5, norm_loudness=True)
            self.assertEqual(mock_model.prepare_conditionals.call_count, 1)

    def test_build_validation_overview_clean(self):
        session = {
            "project_name": "clean_project",
            "chunks": [
                {
                    "index": 1,
                    "id": "c001",
                    "text": "Hello world.",
                    "status": "ready",
                    "audio_path": "/fake/audio1.wav",
                    "validation_status": "passed",
                    "text_score": 0.98,
                    "audio_quality_status": "passed",
                },
                {
                    "index": 2,
                    "id": "c002",
                    "text": "Everything is great.",
                    "status": "ready",
                    "audio_path": "/fake/audio2.wav",
                    "validation_status": "passed",
                    "text_score": 0.95,
                    "audio_quality_status": "passed",
                },
            ],
        }
        scorecard, problems, btn_text = build_validation_overview(session)
        self.assertIn("100% Clean! All 2 validated chunk(s) passed", scorecard)
        self.assertIn("Hello world", build_full_report_markdown(session))
        self.assertEqual(len(problems), 0)
        self.assertEqual(btn_text, "🔄 Regenerate Chunks Needing Review")

    def test_build_validation_overview_with_problems(self):
        session = {
            "project_name": "review_project",
            "chunks": [
                {
                    "index": 1,
                    "id": "c001",
                    "text": "This one is clean.",
                    "status": "ready",
                    "audio_path": "/fake/audio1.wav",
                    "validation_status": "passed",
                    "text_score": 0.97,
                    "audio_quality_status": "passed",
                },
                {
                    "index": 2,
                    "id": "c002",
                    "text": "This one has early cutoff.",
                    "status": "ready",
                    "audio_path": "/fake/audio2.wav",
                    "validation_status": "needs_review",
                    "validation_category": "early_cutoff",
                    "validation_error": "Audio cut off early: ended 4 words before expected text.",
                    "transcript": "This one has",
                    "text_score": 0.60,
                },
                {
                    "index": 3,
                    "id": "c003",
                    "text": "This one failed during generation.",
                    "status": "failed",
                    "error": "CUDA out of memory",
                },
                {
                    "index": 4,
                    "id": "c004",
                    "text": "This one has clipping audio.",
                    "status": "ready",
                    "audio_path": "/fake/audio4.wav",
                    "validation_status": "passed",
                    "text_score": 0.96,
                    "audio_quality_status": "needs_review",
                    "audio_quality_error": "Clipping detected",
                },
                {
                    "index": 5,
                    "id": "c005",
                    "text": "This one is excluded.",
                    "status": "excluded",
                },
            ],
        }
        scorecard, problems, btn_text = build_validation_overview(session)
        self.assertIn("3 chunk(s) need attention", scorecard)
        self.assertIn("Early Cutoff", scorecard)
        self.assertIn("CUDA out of memory", scorecard)
        self.assertIn("Clipping detected", scorecard)
        self.assertIn("1 chunk(s) currently marked as excluded", scorecard)
        self.assertEqual(len(problems), 3)
        self.assertEqual(btn_text, "🔄 Regenerate 3 Problem Chunk(s)")

        # Verify problem indices
        problem_indices = [idx for label, idx in problems]
        self.assertEqual(problem_indices, [3, 2, 4])

        # Verify full report markdown table
        full_md = build_full_report_markdown(session)
        self.assertIn("| 1 | `c001` | `ready` | ✅ `passed` |", full_md)
        self.assertIn("| 2 | `c002` | `ready` | ⚠️ `needs_review` |", full_md)
        self.assertIn("| 3 | `c003` | `failed` | 💥 `unvalidated` |", full_md)

    def test_ui_jump_to_problem_and_scorecard_updates(self):
        session = {
            "project_name": "jump_test",
            "chunks": [
                {"index": 1, "id": "c001", "text": "First chunk", "status": "pending"},
                {"index": 2, "id": "c002", "text": "Second problem chunk", "status": "failed", "error": "Synth error"},
            ],
        }
        # Jump using integer
        res = on_jump_to_problem(session, 2)
        self.assertEqual(res[0], 2)
        self.assertEqual(res[1], "Second problem chunk")
        self.assertIn("Jumped to problem chunk", res[5])

        # Jump using label string
        res_str = on_jump_to_problem(session, "Chunk 2 (c002) — Generation Error")
        self.assertEqual(res_str[0], 2)
        self.assertEqual(res_str[1], "Second problem chunk")

        # Scorecard update function
        scorecard_md, dropdown, full_md, btn = update_validation_scorecard(session)
        self.assertIn("Validation Health Scorecard", scorecard_md)
        self.assertIn("Detailed Verification Manifest", full_md)


if __name__ == "__main__":
    unittest.main()

