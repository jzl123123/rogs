#!/usr/bin/env python3
"""Install only HUGSIM ground dependencies, without the closed-loop simulator."""
import importlib.metadata
from pathlib import Path
import platform
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]

def main():
    if platform.system()!='Linux' or sys.version_info[:2]!=(3,11):
        raise SystemExit('Use a separate Python 3.11 environment on Ubuntu + NVIDIA')
    for name in ('diff-gaussian-rasterization','diff-gs-label'):
        try:importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:continue
        raise SystemExit('RoGS CUDA stack detected. Create a separate HUGSIM environment to preserve the reference stack.')
    pip=[sys.executable,'-m','pip']
    subprocess.run(pip+['install','setuptools==78.1.0','wheel==0.45.1'],check=True)
    subprocess.run(pip+['install','torch==2.4.1','torchvision==0.19.1','--index-url','https://download.pytorch.org/whl/cu118'],check=True)
    subprocess.run(pip+['install','-r',str(ROOT/'requirements/hugsim.txt')],check=True)
    subprocess.run(pip+['install','--no-build-isolation','--no-deps','-e',str(ROOT)],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/fetch_thirdparty.py'),'--verify','--only','HUGSIM','HUGSIM_splat','hugsim_glm'],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/verify_hugsim.py')],check=True)
    subprocess.run([sys.executable,str(ROOT/'tools/build_hugsim.py')],check=True)
    subprocess.run(pip+['check'],check=True)

if __name__=='__main__':main()
