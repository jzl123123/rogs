import ast
import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from rogs.config import load_config, resolve_path
from rogs.nusc_index import image_records, audit, label_path
from verify_algorithm import normalized, verify


class EngineeringTests(unittest.TestCase):
    def test_algorithm_ast_matches_upstream(self):
        self.assertEqual(verify(), 19)

    def test_algorithm_guard_rejects_math_change(self):
        self.assertEqual(normalized("from utils.x import y\na = y * 2"), normalized("from rogs.utils.x import y\na = y * 2"))
        self.assertNotEqual(normalized("a = y * 2"), normalized("a = y * 3"))

    def test_reference_hyperparameters_are_preserved(self):
        reference = yaml.safe_load((ROOT / "configs/reference/local_nusc.yaml").read_text())
        packaged = yaml.safe_load((ROOT / "configs/nuscenes.yaml").read_text())
        for key in ("model", "optimization", "pipeline", "seed"):
            self.assertEqual(reference[key], packaged[key], key)
        for key in ("image_width", "image_height", "camera_names", "min_distance"):
            self.assertEqual(reference["dataset"][key], packaged["dataset"][key])

    def test_environment_and_relative_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            with patch.dict(os.environ, {"ROGS_TEST_DATA": "relative data"}):
                self.assertEqual(resolve_path("${ROGS_TEST_DATA}/images", base), str((base / "relative data/images").resolve()))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaisesRegex(ValueError, "Unresolved"):
                    resolve_path("${UNKNOWN_ROGS_ROOT}/images", base)

    def test_config_does_not_mutate_defaults(self):
        with patch.dict(os.environ, {"NUSC_ROOT": "/data/nusc", "NUSC_SEG_ROOT": "/data/seg", "NUSC_GROUND_ROOT": "/data/nuScenes_road_gt"}):
            config = load_config(ROOT / "configs/nuscenes.yaml")
            self.assertEqual(config["dataset"]["base_dir"], "/data/nusc")
            self.assertEqual(config["optimization"]["z_weight"], 0)
            self.assertEqual(config["optimization"]["epochs"], 2)

    def fixture(self, directory):
        directory = Path(directory)
        meta = directory / "v1.0-trainval"
        meta.mkdir()
        files = {
            "scene": [{"token": "scene-a", "name": "scene-0655"}],
            "sample": [{"token": "a", "scene_token": "scene-a", "timestamp": 0},
                       {"token": "b", "scene_token": "scene-a", "timestamp": 500}],
            "sensor": [{"token": "sensor", "channel": "CAM_FRONT"}],
            "calibrated_sensor": [{"token": "calibration", "sensor_token": "sensor"}],
            "sample_data": [
                {"token": "a0", "sample_token": "a", "is_key_frame": True, "next": "a1", "timestamp": 0, "filename": "samples/CAM_FRONT/a.jpg", "calibrated_sensor_token": "calibration"},
                {"token": "a1", "sample_token": "b", "is_key_frame": False, "next": "b0", "timestamp": 200, "filename": "sweeps/CAM_FRONT/aa.jpg", "calibrated_sensor_token": "calibration"},
                {"token": "b0", "sample_token": "b", "is_key_frame": True, "next": "", "timestamp": 500, "filename": "samples/CAM_FRONT/b.jpg", "calibrated_sensor_token": "calibration"}],
        }
        for name, value in files.items():
            (meta / (name + ".json")).write_text(json.dumps(value))
        return {"base_dir": str(directory), "image_dir": str(directory / "seg"),
                "road_gt_dir": str(directory / "ground"), "version": "trainval",
                "clip_list": ["scene-0655"], "camera_names": ["CAM_FRONT"]}

    def test_index_includes_sweeps_without_duplicate_keyframes(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = image_records(self.fixture(directory))
            self.assertEqual([r["token"] for r in rows], ["a0", "a1", "b0"])
            self.assertEqual(rows[1]["label_filename"], "sweeps/seg_CAM_FRONT/aa.png")

    def test_missing_input_is_reported_not_silently_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.fixture(directory)
            result = audit(dataset)
            self.assertEqual(result["images"], 3)
            self.assertEqual(result["missing_count"], 7)
            self.assertEqual(audit(dataset, False, False)["missing_count"], 3)

    def test_scene_selection_is_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.fixture(directory)
            dataset["clip_list"] = ["scene-065"]
            with self.assertRaisesRegex(ValueError, "Unknown scenes"):
                image_records(dataset)

    def test_output_path_cannot_escape_dataset(self):
        for invalid in ("../CAM_FRONT/a.jpg", "/CAM_FRONT/a.jpg", "samples/../CAM_FRONT/a.jpg"):
            with self.assertRaises(ValueError):
                label_path(invalid)

    def test_cyclic_sweep_chain_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            dataset = self.fixture(directory)
            path = Path(directory) / "v1.0-trainval/sample_data.json"
            records = json.loads(path.read_text())
            records[1]["next"] = "a1"
            path.write_text(json.dumps(records))
            with self.assertRaisesRegex(ValueError, "Cyclic"):
                image_records(dataset)

    def test_generated_semantic_variant_preserves_cuda_code(self):
        import build_extensions
        from fetch_thirdparty import tree_hash
        before = tree_hash(ROOT / "thirdparty/diff-gs-depth-alpha")
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(build_extensions, "BUILD", Path(directory)):
                rgb, semantic = build_extensions.prepare_gs(7)
            self.assertIn("#define NUM_CHANNELS 3", (rgb / "cuda_rasterizer/config.h").read_text())
            self.assertIn("#define NUM_CHANNELS 7", (semantic / "cuda_rasterizer/config.h").read_text())
            self.assertTrue((semantic / "diff_gs_label/__init__.py").is_file())
            self.assertTrue((semantic / "third_party/glm/glm/glm.hpp").is_file())
            upstream = ROOT / "thirdparty/diff-gs-depth-alpha"
            for file in upstream.rglob("*"):
                if file.suffix in (".cu", ".cpp", ".cuh"):
                    self.assertEqual(file.read_bytes(), (semantic / file.relative_to(upstream)).read_bytes())
        self.assertEqual(before, tree_hash(ROOT / "thirdparty/diff-gs-depth-alpha"))

    def test_segmentation_source_label_contract(self):
        path = ROOT / "thirdparty/Mask2Former/mask2former/data/datasets/register_mapillary_vistas.py"
        for node in ast.parse(path.read_text()).body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MAPILLARY_VISTAS_SEM_SEG_CATEGORIES" for t in node.targets):
                labels = ast.literal_eval(node.value)
                break
        # 65 train classes plus the ignored/unlabeled category at index 65.
        self.assertEqual(len(labels), 66)
        expected = {2: "Curb", 13: "Road", 23: "Lane Marking - Crosswalk", 24: "Lane Marking - General", 64: "Ego Vehicle"}
        self.assertEqual({i: labels[i]["readable"] for i in expected}, expected)


if __name__ == "__main__":
    unittest.main()
