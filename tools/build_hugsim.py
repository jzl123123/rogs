#!/usr/bin/env python3
"""Build HUGSIM's pinned gsplat fork in a separate CUDA environment."""
import argparse
from pathlib import Path
import platform
import shutil
import subprocess
import sys
from build_extensions import stage, verify

ROOT=Path(__file__).resolve().parents[1]

def prepare():
    verify('hugsim_glm')
    target=stage('HUGSIM_splat','HUGSIM_splat')
    glm=target/'gsplat/cuda/csrc/third_party/glm'
    if glm.exists():shutil.rmtree(glm)
    shutil.copytree(ROOT/'thirdparty/hugsim_glm',glm)
    return target


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if not args.prepare_only:
        if platform.system()!='Linux':parser.error('Compile HUGSIM on Ubuntu + NVIDIA')
        import torch
        from torch.utils.cpp_extension import CUDA_HOME
        if not torch.cuda.is_available() or CUDA_HOME is None or shutil.which('nvcc') is None:
            parser.error('A visible NVIDIA GPU and CUDA Toolkit/nvcc are required')
    target=prepare()
    if args.prepare_only:
        print(target);return
    subprocess.run([sys.executable,'-m','pip','install','--no-build-isolation','--no-deps',str(target)],check=True)
    subprocess.run([sys.executable,'-m','rogs.hugsim.runner','--smoke'],check=True)

if __name__=='__main__':main()
