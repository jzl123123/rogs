"""Lazy camera loading for the unchanged HUGSIM training loop."""
import json
from pathlib import Path


class LazyCamera:
    def __init__(self, frame, root, data_device):
        import torch
        self.frame = frame
        self.root = Path(root)
        self.width, self.height = frame['width'], frame['height']
        self.image_width, self.image_height = self.width, self.height
        self.image_name = '_'.join(Path(frame['rgb_path']).parts[-2:]).rsplit('.', 1)[0]
        self.timestamp = frame.get('timestamp', -1)
        self.K = torch.tensor(frame['intrinsics'], dtype=torch.float32, device='cuda')
        self.c2w = torch.tensor(frame['camtoworld'], dtype=torch.float32, device='cuda')
        self.data_device = data_device
        self.optical_gt = self.depth = self.mask = None
        self.dynamics = {}

    @property
    def original_image(self):
        import numpy as np
        import torch
        from PIL import Image
        with Image.open(self.root/self.frame['rgb_path']) as im:
            image = np.array(im.convert('RGB'), copy=True)
        return torch.from_numpy(image).permute(2, 0, 1).float().div(255).clamp(0, 1).to(self.data_device)

    @property
    def semantic2d(self):
        import numpy as np
        import torch
        rel = self.frame['rgb_path'].replace('images', 'semantics').replace('.png', '.npy').replace('.jpg', '.npy')
        labels = np.load(self.root/rel, allow_pickle=False)
        labels[(labels == 14) | (labels == 15)] = 13  # upstream native reader convention
        return torch.from_numpy(labels.astype(np.int64))[None].to(self.data_device)


def load_cameras(cfg, data_type, ignore_dynamic=False):
    root = Path(cfg.source_path)
    meta = json.loads((root/'meta_data.json').read_text())
    shared = (root/'adapter_manifest.json').is_file()
    train, test = [], []
    for idx, frame in enumerate(meta['frames']):
        split = frame['split'] if shared else ('test' if idx % 30 >= 24 else 'train')
        camera = LazyCamera(frame, root, cfg.model.data_device)
        (test if split == 'test' else train).append(camera)
    if not train or not test:
        raise ValueError('HUGSIM needs both train and test cameras')
    return train, test, None
