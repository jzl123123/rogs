#!/usr/bin/env python3
"""Download an explicitly selected official model; this is not RoMe checkpoint identification."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "mapillary-swin-l-semantic": {
        "url": "https://dl.fbaipublicfiles.com/maskformer/mask2former/mapillary_vistas/semantic/maskformer2_swin_large_IN21k_384_bs16_300k/model_final_90ee2d.pkl",
        "md5_prefix": "90ee2d", "filename": "mapillary_swin_l_semantic_90ee2d.pkl"},
}


def digest(path, algorithm):
    h = hashlib.new(algorithm)
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, choices=MODELS)
    args = parser.parse_args()
    model = MODELS[args.model]
    directory = ROOT / "thirdparty/weights"
    directory.mkdir(exist_ok=True)
    target = directory / model["filename"]
    if not target.exists():
        temporary = target.with_suffix(".partial")
        subprocess.run(["curl", "--fail", "--location", "--retry", "3", "--connect-timeout", "20",
                        "--output", str(temporary), model["url"]], check=True)
        if not digest(temporary, "md5").startswith(model["md5_prefix"]):
            raise ValueError("Official model MD5 prefix mismatch; partial download preserved")
        temporary.replace(target)
    if not digest(target, "md5").startswith(model["md5_prefix"]):
        raise ValueError("Existing model MD5 prefix mismatch")
    record = dict(model, sha256=digest(target, "sha256"), paper_checkpoint_equivalence="not established")
    target.with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    print(target)
    print("SHA256", record["sha256"])


if __name__ == "__main__":
    main()

