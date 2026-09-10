#!/usr/bin/env python3
"""Install into the currently activated environment on the Ubuntu NVIDIA host."""
import argparse
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PROFILES = {
    "upstream-cu117": ("1.13.1+cu117", "0.14.1+cu117", "https://download.pytorch.org/whl/cu117"),
    "supported-cu118": ("2.1.2+cu118", "0.16.2+cu118", "https://download.pytorch.org/whl/cu118"),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--torch-profile", choices=PROFILES, required=True)
    parser.add_argument("--segmentation", action="store_true")
    args = parser.parse_args()
    if platform.system() != "Linux":
        parser.error("Run the full installation on Ubuntu, not on the Mac")
    pip = [sys.executable, "-m", "pip"]
    torch, vision, index = PROFILES[args.torch_profile]
    subprocess.run(pip + ["install", "setuptools==69.5.1", "wheel==0.44.0"], check=True)
    subprocess.run(pip + ["install", "torch==" + torch, "torchvision==" + vision, "--extra-index-url", index], check=True)
    subprocess.run(pip + ["install", "-r", str(ROOT / "requirements/runtime.txt")], check=True)
    subprocess.run(pip + ["install", "--no-deps", "-e", str(ROOT)], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/fetch_thirdparty.py"), "--verify"], check=True)
    subprocess.run([sys.executable, str(ROOT / "tools/verify_algorithm.py")], check=True)
    command = [sys.executable, str(ROOT / "tools/build_extensions.py")]
    if args.segmentation:
        command.append("--segmentation")
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()

