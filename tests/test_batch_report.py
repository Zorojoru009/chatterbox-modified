import tempfile
import unittest
from pathlib import Path

from batch_report import failed_ids, plan_chunks, record_chunk, save_report, load_report, text_sha


def manifest(tmp: Path, texts):
    items = []
    for i, text in enumerate(texts, start=1):
        items.append({"index": i, "id": f"b{i:03d}", "text": text, "audio_path": str(tmp / f"{i:04d}_b{i:03d}.wav")})
    return items


class BatchReportTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.items = manifest(self.tmp, ["one", "two", "three"])
        self.report = {"chunks": {}}
        for item in self.items:
            Path(item["audio_path"]).write_bytes(b"x")
            record_chunk(self.report, item, {"passed": item["id"] != "b002", "score": 0.5, "attempts": [{}, {}]})

    def test_full_run_selects_everything(self):
        self.assertEqual(plan_chunks(self.items, None, {"chunks": {}}), {1, 2, 3})

    def test_failed_reruns_only_failures(self):
        self.assertEqual(failed_ids(self.report), ["b002"])
        self.assertEqual(plan_chunks(self.items, "failed", self.report), {2})

    def test_ids_and_indices(self):
        self.assertEqual(plan_chunks(self.items, "b001, 3", self.report), {1, 3})
        with self.assertRaises(ValueError):
            plan_chunks(self.items, "b009", self.report)

    def test_changed_text_must_be_rerun(self):
        items = manifest(self.tmp, ["one", "two edited", "three"])
        with self.assertRaises(ValueError):
            plan_chunks(items, "b001", self.report)
        self.assertEqual(plan_chunks(items, "b002", self.report), {2})

    def test_missing_wav_must_be_rerun(self):
        Path(self.items[2]["audio_path"]).unlink()
        with self.assertRaises(ValueError):
            plan_chunks(self.items, "b002", self.report)

    def test_save_and_summary(self):
        path = self.tmp / "validation_report.json"
        save_report(self.report, path)
        loaded = load_report(path)
        self.assertEqual(loaded["failed"], ["b002"])
        self.assertEqual(loaded["passed"], 2)
        self.assertEqual(loaded["regenerated"], 3)
        self.assertEqual(loaded["chunks"]["1"]["text_sha"], text_sha("one"))


if __name__ == "__main__":
    unittest.main()
