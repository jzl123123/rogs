"""Engineering configuration; no algorithm defaults are injected."""
import copy
import os
from pathlib import Path
import re

import yaml


def project_root():
    override = os.environ.get("ROGS_PROJECT_ROOT")
    root = Path(override).expanduser().resolve() if override else Path(__file__).resolve().parents[2]
    if not (root / "thirdparty/sources.lock.json").is_file():
        raise ValueError("Use an editable project install or set ROGS_PROJECT_ROOT to this checkout")
    return root


def resolve_path(value, base):
    expanded = os.path.expandvars(os.path.expanduser(str(value)))
    if re.search(r"\$\{[^}]+\}|\$[A-Za-z_][A-Za-z_0-9]*", expanded):
        raise ValueError("Unresolved environment variable in path: " + expanded)
    path = Path(expanded)
    return str((base / path).resolve() if not path.is_absolute() else path.resolve())


def load_config(filename):
    path = Path(filename).expanduser().resolve()
    with path.open() as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration must be a mapping")
    required = ("seed", "device", "output", "dataset", "model", "optimization", "train", "pipeline")
    for name in required:
        if name not in cfg:
            raise ValueError("Missing configuration field: " + name)
    cfg = copy.deepcopy(cfg)
    cfg["file"] = str(path)
    # Data paths are config-relative. Upstream snapshots are kept separately.
    for name in ("base_dir", "image_dir", "road_gt_dir", "pose_dir"):
        if name in cfg["dataset"]:
            cfg["dataset"][name] = resolve_path(cfg["dataset"][name], path.parent)
    cfg["output"] = resolve_path(cfg["output"], path.parent)
    if cfg["train"].get("start_checkpoint"):
        cfg["train"]["start_checkpoint"] = resolve_path(cfg["train"]["start_checkpoint"], path.parent)
    validate_reference(cfg)
    return cfg


def validate_reference(cfg):
    ds = cfg["dataset"]
    if ds.get("dataset") != "NuscDataset":
        raise ValueError("Engineering CLI currently supports nuScenes; upstream KITTI remains in the snapshot")
    if not ds.get("clip_list") or not ds.get("camera_names"):
        raise ValueError("clip_list and camera_names must not be empty")
    if len(set(ds["camera_names"])) != len(ds["camera_names"]):
        raise ValueError("Duplicate camera names")
    if cfg["model"]["bev_resolution"] <= 0 or cfg["optimization"]["epochs"] < 1:
        raise ValueError("Positive resolution and epochs are required")
    if cfg["optimization"]["seg_loss_weight"] <= 0:
        raise ValueError("Reference code requires segmentation masks and labels; use seg_loss_weight > 0")
    if cfg["train"].get("start_checkpoint"):
        raise ValueError("Upstream checkpoint resume format is inconsistent; fresh runs only in reference mode")

