"""Explicit data protocol and independent HUGSIM training configuration."""
import copy
from pathlib import Path
import yaml
from rogs.config import project_root, resolve_path


def load_profile(filename):
    filename = Path(filename).resolve()
    p = yaml.safe_load(filename.read_text())
    if p.get('source_mode') not in ('shared_rogs', 'native_hugsim'):
        raise ValueError('HUGSIM source_mode must be shared_rogs or native_hugsim')
    p = copy.deepcopy(p)
    p['file'] = str(filename)
    for key in ('prepared_root', 'source_path'):
        if p.get(key):
            p[key] = resolve_path(p[key], filename.parent)
    if p['source_mode'] == 'shared_rogs':
        if p.get('mask_policy') not in ('rogs', 'road_sidewalk'):
            raise ValueError('Unknown HUGSIM mask_policy')
        if p.get('split') != 'every_fifth_per_camera':
            raise ValueError('Shared HUGSIM requires an explicit every_fifth_per_camera holdout')
        if p.get('initialization', {}).get('mode') != 'rogs_lidar_projected':
            raise ValueError('Shared initialization must be rogs_lidar_projected; use native mode for original initialization')
        if not p.get('prepared_root'):
            raise ValueError('prepared_root is required')
        count = p['initialization'].get('max_points', 0)
        if not isinstance(count, int) or count < 2:
            raise ValueError('initialization.max_points must be at least 2')
    elif not p.get('source_path'):
        raise ValueError('native_hugsim requires source_path')
    return p


def source_directory(cfg, profile):
    if profile['source_mode'] == 'native_hugsim':
        return Path(profile['source_path'])
    clips = cfg['dataset']['clip_list']
    if len(clips) != 1:
        raise ValueError('HUGSIM trains one scene per run; select exactly one clip')
    return Path(profile['prepared_root']) / clips[0]


def training_config(profile, source, output):
    root = project_root() / 'thirdparty/HUGSIM/configs'
    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = copy.deepcopy(v)
        return a
    cfg = merge(yaml.safe_load((root/'gs_base.yaml').read_text()), yaml.safe_load((root/'nusc.yaml').read_text()))
    overrides = profile.get('overrides') or {}
    if set(overrides) - {'model', 'ground', 'opt', 'train', 'semantic'}:
        raise ValueError('Only HUGSIM model/ground/opt/train/semantic overrides are allowed')
    merge(cfg, overrides)
    cfg.update(source_path=str(source), model_path=str(output))
    iterations = cfg['train']['iterations']
    checkpoints = cfg['train']['checkpoint_iterations']
    if iterations < 1 or not checkpoints or iterations not in checkpoints or any(x < 1 or x > iterations for x in checkpoints):
        raise ValueError('HUGSIM checkpoints must be within iterations and include the final iteration')
    return cfg
