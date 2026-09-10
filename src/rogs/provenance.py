import datetime
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import uuid

import yaml

from rogs.config import project_root


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run_bundle(cfg):
    run_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8]
    destination = Path(cfg["output"]) / run_id
    destination.mkdir(parents=True, exist_ok=False)
    cfg = dict(cfg)
    cfg["output"] = str(destination)
    original = cfg.pop("file")
    resolved = destination / "config.resolved.yaml"
    resolved.write_text(yaml.safe_dump(cfg, sort_keys=False))
    cfg["file"] = str(resolved)
    versions = {}
    for name in ("torch", "torchvision", "numpy", "pytorch3d", "nuscenes-devkit", "detectron2"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    root = project_root()
    manifest = {"run_id": run_id, "original_config": original,
                "original_config_sha256": sha256_file(original),
                "resolved_config_sha256": sha256_file(resolved),
                "python": platform.python_version(), "platform": platform.platform(),
                "packages": versions, "sources": json.loads((root / "thirdparty/sources.lock.json").read_text()),
                "algorithm_sources": {p.relative_to(root / "src").as_posix(): sha256_file(p)
                                      for p in sorted((root / "src/rogs").rglob("*.py"))}}
    seg_manifest = Path(cfg["dataset"]["image_dir"]) / "segmentation_manifest.json"
    if seg_manifest.is_file():
        manifest["segmentation_manifest"] = json.loads(seg_manifest.read_text())
    else:
        manifest["segmentation_manifest"] = {"source": "externally_provided", "checkpoint_identity": "unverified"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return cfg, destination

