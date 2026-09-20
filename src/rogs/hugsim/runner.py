"""Isolated HUGSIM CUDA worker, sharing only explicit data and provenance."""
import argparse
import inspect
import json
from pathlib import Path
import subprocess
import sys
import yaml

from rogs.config import project_root
from rogs.hugsim.config import source_directory, training_config
from rogs.hugsim.data import validate_source
from rogs.provenance import run_bundle, sha256_file


def launch(cfg, profile, dry_run=False):
    source = source_directory(cfg, profile)
    shared = profile['source_mode'] == 'shared_rogs'
    if shared != (source/'adapter_manifest.json').is_file():
        raise ValueError('Source mode does not match prepared-data provenance')
    counts = validate_source(source, shared)
    resolved = training_config(profile, source, Path(cfg['output'])/'<run>')
    if dry_run:
        print(yaml.safe_dump({'source':str(source), 'split':counts, 'hugsim':resolved},sort_keys=False))
        return
    bundled, directory = run_bundle(cfg)
    resolved = training_config(profile, source, directory)
    resolved.update(seed=cfg['seed'], device=cfg['device'])
    file = directory/'hugsim.resolved.yaml'
    file.write_text(yaml.safe_dump(resolved, sort_keys=False))
    (directory/'hugsim.profile.yaml').write_text(yaml.safe_dump(profile,sort_keys=False))
    manifest_file = directory/'manifest.json'
    manifest = json.loads(manifest_file.read_text())
    manifest.update(backend='hugsim_ground', status='running', hugsim_profile=profile,
                    hugsim_config_sha256=sha256_file(file), camera_split=counts,
                    input_metadata_sha256=sha256_file(source/'meta_data.json'),
                    initial_ground_sha256=sha256_file(source/'ground_points3d.ply'))
    adapter = source/'adapter_manifest.json'
    if adapter.exists():manifest['hugsim_adapter']=json.loads(adapter.read_text())
    manifest_file.write_text(json.dumps(manifest,indent=2)+'\n')
    print('HUGSIM run bundle:',directory,flush=True)
    try:
        subprocess.run([sys.executable,'-m','rogs.hugsim.runner','--worker',str(file)],check=True)
        manifest['status']='complete'
    except BaseException as e:
        manifest.update(status='failed', error=f'{type(e).__name__}: {e}')
        raise
    finally:
        manifest_file.write_text(json.dumps(manifest,indent=2)+'\n')
    return directory


def check_runtime(smoke=False):
    import torch
    if not torch.cuda.is_available():
        raise ValueError('HUGSIM needs an NVIDIA CUDA runtime')
    from gsplat.rendering import rasterization
    parameters = inspect.signature(rasterization).parameters
    if not {'smts', 'ortho'} <= set(parameters):
        raise ValueError('Wrong gsplat: install thirdparty/HUGSIM_splat, not stock gsplat')
    if smoke:
        from types import SimpleNamespace
        import numpy as np
        from rogs.hugsim.ground_model import GroundModel
        from rogs.hugsim.graphics_utils import BasicPointCloud
        from rogs.hugsim.render import render_ground
        pcd = BasicPointCloud(points=np.array([[-.2,1,3],[.2,1,3],[0,1,4]],dtype=np.float32),
                              colors=np.ones((3,3),dtype=np.float32)*.5,normals=np.zeros((3,3)))
        model = GroundModel(0, pcd)
        cam = SimpleNamespace(K=torch.tensor([[40,0,32],[0,40,32],[0,0,1.]],device='cuda'),
                              c2w=torch.eye(4,device='cuda'),width=64,height=64)
        output = render_ground(cam, model, torch.zeros(3,device='cuda'))
        assert output['render'].shape==(3,64,64) and output['feats'].shape==(20,64,64)
        loss = output['render'].sum()+output['feats'].sum()
        loss.backward()
        if model._xyz.grad is None or not torch.isfinite(model._xyz.grad).all():
            raise ValueError('HUGSIM renderer backward failed')
    print('HUGSIM runtime OK' + ('; RGB/20-class forward and backward passed' if smoke else ''))


def worker(file):
    import random
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    cfg = OmegaConf.load(file)
    if not str(cfg.device).startswith('cuda'):
        raise ValueError('HUGSIM requires a CUDA device')
    torch.cuda.set_device(torch.device(cfg.device))
    random.seed(cfg.seed);np.random.seed(cfg.seed);torch.manual_seed(cfg.seed);torch.cuda.manual_seed_all(cfg.seed)
    check_runtime()
    from rogs.hugsim.train import training
    training(cfg)
    # Preserve native checkpoint; additionally expose a complete Gaussian PLY.
    from rogs.hugsim.ground_model import GroundModel
    state, iteration = torch.load(Path(cfg.model_path)/'ckpts'/f'ground_chkpnt{cfg.train.iterations}.pth', map_location='cuda', weights_only=False)
    model = GroundModel(cfg.model.sh_degree, model_args=state)
    model.save_ply(str(Path(cfg.model_path)/'ground_gaussians.ply'))
    (Path(cfg.model_path)/'ground_export.json').write_text(json.dumps({
        'iteration':iteration,'coordinate_frame':'HUGSIM source frame',
        'source_metadata':str(Path(cfg.source_path)/'meta_data.json'),
        'format':'HUGSIM 3D Gaussian PLY, 20 semantic channels; not RoGS checkpoint format'
    },indent=2)+'\n')


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--worker')
    parser.add_argument('--smoke',action='store_true')
    args=parser.parse_args()
    if args.worker:worker(args.worker)
    else:check_runtime(args.smoke)
