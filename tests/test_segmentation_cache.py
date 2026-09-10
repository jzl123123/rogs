"""CPU-only regression coverage for segmentation resume without a GPU runtime."""
from contextlib import ExitStack, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rogs import segmentation as seg


class SegmentationCacheTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory()))
        self.cfg = {"device": "cuda", "dataset": {"base_dir": str(self.root),
                    "image_dir": str(self.root / "labels"), "clip_list": ["scene-test"]}}
        self.profile = {"config_file": "config.yaml", "checkpoint": "model.pkl"}
        self.stack.enter_context(redirect_stdout(io.StringIO()))
        self.stack.enter_context(patch.object(seg, "profile_paths", return_value=self.profile))
        self.stack.enter_context(patch.object(seg, "sha256_file", return_value="test-sha"))
        self.stack.enter_context(patch.object(seg, "audit", return_value={"missing_count": 0}))
        self.index = self.stack.enter_context(patch.object(seg, "image_records", return_value=[]))
        self.predictor = self.stack.enter_context(patch.object(seg, "build_predictor", side_effect=AssertionError("Must not initialize GPU")))
        seg.segment(self.cfg, "profile.yaml")
        self.row = {"filename": "samples/a.jpg", "label_filename": "samples/seg_a.png"}
        self.output = self.root / "labels" / self.row["label_filename"]
        self.output.parent.mkdir(parents=True)
        self.output.touch()
        self.index.return_value = [self.row]

    def test_fully_cached_run_never_loads_model(self):
        with patch.object(seg, "validate_cached_label") as validate:
            record = seg.segment(self.cfg, "profile.yaml")
        self.predictor.assert_not_called()
        validate.assert_called_once_with(self.root / "samples/a.jpg", self.output)
        self.assertEqual(record["status"], "complete")
        self.assertEqual(record["last_run"]["cached_images"], 1)
        self.assertEqual(record["last_run"]["processed_images"], 0)
        self.assertEqual(record["last_run"]["model_setup_seconds"], 0)
        self.assertEqual(json.loads((self.root / "labels/segmentation_manifest.json").read_text()), record)

    def test_invalid_cache_blocks_before_model_loading(self):
        with patch.object(seg, "validate_cached_label", side_effect=ValueError("Invalid existing label")):
            with self.assertRaisesRegex(ValueError, "Invalid existing label"):
                seg.segment(self.cfg, "profile.yaml")
        self.predictor.assert_not_called()

    def test_changed_model_cannot_reuse_cache(self):
        self.profile["checkpoint"] = "different-model.pkl"
        with self.assertRaisesRegex(ValueError, "different model manifest"):
            seg.segment(self.cfg, "profile.yaml")
        self.predictor.assert_not_called()

    def test_resume_selects_only_missing_frames(self):
        missing = {"filename": "samples/b.jpg", "label_filename": "samples/seg_b.png"}
        with patch.object(seg, "validate_cached_label") as validate:
            pending = seg.pending_records([self.row, missing], self.root, self.root / "labels")
        self.assertEqual(pending, [missing])
        validate.assert_called_once()


if __name__ == "__main__":
    unittest.main()
