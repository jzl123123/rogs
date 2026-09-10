#!/usr/bin/env python3
"""Stage and build vendored extensions without modifying pinned source trees."""
import argparse
import json
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys

from fetch_thirdparty import tree_hash

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / "thirdparty"
BUILD = VENDOR / "_build"


def verify(name):
    lock = json.loads((VENDOR / "sources.lock.json").read_text())["sources"]
    if name not in lock or tree_hash(VENDOR / name) != lock[name]["tree_sha256"]:
        raise ValueError("Missing or modified pinned source: " + name)


def stage(name, target):
    verify(name)
    destination = BUILD / target
    # Delete only generated directories immediately under the dedicated build root.
    if destination.exists():
        if destination.is_symlink():
            raise ValueError("Build destination must not be a symlink")
        shutil.rmtree(destination)
    shutil.copytree(VENDOR / name, destination, ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    return destination


def prepare_gs(channels=7):
    if channels != 7:
        raise ValueError("Current engineering workflow builds the nuScenes 7-channel semantic renderer")
    verify("glm")
    destinations = []
    for package, count in (("diff_gaussian_rasterization", 3),
                           ("diff_gs_label", channels)):
        directory = stage("diff-gs-depth-alpha", package)
        glm = directory / "third_party/glm"
        if glm.exists():
            shutil.rmtree(glm)
        shutil.copytree(VENDOR / "glm", glm)
        if package != "diff_gaussian_rasterization":
            (directory / "diff_gaussian_rasterization").rename(directory / package)
            setup = directory / "setup.py"
            setup.write_text(setup.read_text().replace("diff_gaussian_rasterization", package))
        header = directory / "cuda_rasterizer/config.h"
        content, changes = re.subn(r"(#define\s+NUM_CHANNELS\s+)3\b", lambda m: m[1] + str(count), header.read_text())
        if changes != 1:
            raise ValueError("Unexpected NUM_CHANNELS declaration")
        header.write_text(content)
        destinations.append(directory)
    return destinations


def install(path):
    subprocess.run([sys.executable, "-m", "pip", "install", "--no-build-isolation", "--no-deps", "."], cwd=path, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true", help="Generate GS build trees; no CUDA needed")
    parser.add_argument("--segmentation", action="store_true")
    args = parser.parse_args()
    if not args.prepare_only:
        if platform.system() != "Linux":
            parser.error("Build the reference stack on the Ubuntu NVIDIA host")
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        if CUDA_HOME is None or shutil.which("nvcc") is None:
            parser.error("CUDA Toolkit and nvcc are required; a CUDA wheel alone is insufficient")
        if not torch.cuda.is_available():
            parser.error("Run this build/validation workflow on a visible NVIDIA GPU")
    paths = prepare_gs()
    if args.prepare_only:
        print("\n".join(map(str, paths)))
        return
    install(stage("pytorch3d", "pytorch3d"))
    for path in paths:
        install(path)
    if args.segmentation:
        install(stage("detectron2", "detectron2"))
        ops = stage("Mask2Former", "Mask2Former") / "mask2former/modeling/pixel_decoder/ops"
        install(ops)
    subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
    command = [sys.executable, "-m", "rogs", "doctor", "--smoke"]
    if args.segmentation:
        command.append("--segmentation")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
