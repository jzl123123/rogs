"""nuScenes/RoGS-to-HUGSIM data adapter. Training algorithms live separately."""
import hashlib
import json
from pathlib import Path
import shutil
import uuid

from rogs.nusc_index import audit, image_records, safe_relative
from rogs.provenance import sha256_file
from rogs.hugsim.config import source_directory


def pose_matrix(rotation, translation):
    import numpy as np
    q = np.asarray(rotation, dtype=float)
    if q.shape != (4,) or not np.isfinite(q).all() or np.linalg.norm(q) == 0:
        raise ValueError('Invalid nuScenes wxyz quaternion')
    w, x, y, z = q / np.linalg.norm(q)
    t = np.eye(4)
    t[:3, :3] = [[1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
                 [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
                 [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]]
    t[:3, 3] = translation
    return t


def camera_geometry(k, c2w, origin_inverse, original_size, output_size, crop_top):
    import numpy as np
    k = np.array(k, dtype=float, copy=True)
    k[0] *= output_size[0] / original_size[0]
    k[1] *= output_size[1] / original_size[1]
    k[1, 2] -= crop_top
    return k, origin_inverse @ c2w


def convert_labels(label, camera, policy, crop_top):
    import numpy as np
    from rogs.hugsim.shared_mask import label2mask
    if label.ndim != 2 or label.size == 0 or label.min() < 0 or label.max() > 64:
        raise ValueError('Expected raw Mapillary IDs 0..64, not an already remapped RoGS label')
    original = label.copy()
    if policy == 'rogs':
        mask, _ = label2mask(label.copy())
        if camera == 'CAM_BACK':
            mask[int(0.83 * mask.shape[0]):] = 0
        valid = mask.astype(bool)
    elif policy == 'road_sidewalk':
        valid = np.isin(label, [7, 8, 13, 14, 15, 23, 24, 41])
    else:
        raise ValueError('Unknown mask policy')
    result = np.full(label.shape, 19, dtype=np.uint8)
    result[valid] = 0
    result[valid & (original == 15)] = 1
    return result[crop_top:]


def project_ground(points, front_poses, camera_height):
    """Same nearest-camera plane projection as HUGSIM, with bounded CPU memory."""
    import numpy as np
    if len(front_poses) < 2 or len(points) < 2:
        raise ValueError('At least two front poses and ground points are required')
    # Upstream excludes the last front camera in its nearest-neighbor search.
    # Chunk the original distance/argmin calculation; retain its first-index tie rule.
    idx = np.concatenate([np.argmin(np.sqrt(np.sum(
        (points[start:start+512, None, :] - front_poses[None, :-1, :3, 3]) ** 2,
        axis=-1)), axis=1) for start in range(0, len(points), 512)])
    c2w = front_poses[idx]
    w2c = np.linalg.inv(front_poses)[idx]
    local = np.einsum('nij,nj->ni', w2c[:, :3, :3], points) + w2c[:, :3, 3]
    local[:, 1] = camera_height
    return np.einsum('nij,nj->ni', c2w[:, :3, :3], local) + c2w[:, :3, 3]


def metadata_tables(dataset):
    root = Path(dataset['base_dir']) / ('v1.0-' + dataset['version'])
    return {name: {r['token']: r for r in json.loads((root/(name+'.json')).read_text())}
            for name in ('sample_data', 'calibrated_sensor', 'ego_pose', 'sensor')}


def global_camera(row, tables):
    sample = tables['sample_data'][row['token']]
    cal = tables['calibrated_sensor'][sample['calibrated_sensor_token']]
    ego = tables['ego_pose'][sample['ego_pose_token']]
    return cal['camera_intrinsic'], pose_matrix(ego['rotation'], ego['translation']) @ pose_matrix(cal['rotation'], cal['translation'])


def front_height(cfg, first_front, tables):
    """HUGSIM nuScenes loader's first-LiDAR RANSAC camera-height calculation."""
    import numpy as np
    import open3d as o3d
    sd = tables['sample_data'][first_front['token']]
    candidates = []
    for sample in tables['sample_data'].values():
        cal = tables['calibrated_sensor'][sample['calibrated_sensor_token']]
        sensor = tables['sensor'][cal['sensor_token']]
        if sample['sample_token'] == sd['sample_token'] and sample['is_key_frame'] and sensor['channel'] == 'LIDAR_TOP':
            candidates.append(sample)
    if len(candidates) != 1:
        raise ValueError('Cannot locate first-frame LiDAR for camera-height estimation')
    lidar = candidates[0]
    safe_relative(lidar['filename'])
    points = np.fromfile(Path(cfg['dataset']['base_dir'])/lidar['filename'], dtype=np.float32).reshape(-1, 5)[:, :3]
    near = (np.abs(points[:, 0]) < 3) & (np.abs(points[:, 1]) < 6)
    ego = (np.abs(points[:, 0]) < 1.5) & (np.abs(points[:, 1]) < 2.5)
    points = points[near & ~ego]
    cal = tables['calibrated_sensor'][lidar['calibrated_sensor_token']]
    lidar2ego = pose_matrix(cal['rotation'], cal['translation'])
    points = points @ lidar2ego[:3, :3].T + lidar2ego[:3, 3]
    if len(points) < 3:
        raise ValueError('Too few LiDAR points for camera-height RANSAC')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    o3d.utility.random.seed(cfg['seed'])
    plane, _ = pcd.segment_plane(distance_threshold=0.01, ransac_n=3, num_iterations=1000)
    a, b, c, d = plane
    t = tables['calibrated_sensor'][sd['calibrated_sensor_token']]['translation']
    if abs(c) < 1e-6:
        raise ValueError('Estimated ground plane is vertical')
    height = t[2] + (a*t[0]+b*t[1]+d)/c
    if not np.isfinite(height) or height <= 0:
        raise ValueError('Invalid estimated front camera height')
    return float(height)


def validate_source(source, shared=False):
    import numpy as np
    from PIL import Image
    source = Path(source)
    metadata = json.loads((source/'meta_data.json').read_text())
    frames = metadata.get('frames', [])
    if not frames:
        raise ValueError('Empty HUGSIM camera manifest')
    counts = {'train': 0, 'test': 0}
    for idx, frame in enumerate(frames):
        rel = str(safe_relative(frame['rgb_path']))
        image_path = source/rel
        semantic = source/rel.replace('images', 'semantics').replace('.jpg', '.npy').replace('.png', '.npy')
        with Image.open(image_path) as image:
            size = image.size
        labels = np.load(semantic, allow_pickle=False)
        if labels.shape != (size[1], size[0]) or not np.issubdtype(labels.dtype, np.integer) or labels.min() < 0 or labels.max() > 19:
            raise ValueError('Invalid HUGSIM semantic array: ' + str(semantic))
        if not np.any(labels <= 1):
            raise ValueError('No ground supervision in '+str(semantic)+'; upstream semantic loss would be undefined')
        k = np.asarray(frame['intrinsics']); pose = np.asarray(frame['camtoworld'])
        if k.shape not in ((3, 3), (4, 4)) or pose.shape != (4, 4) or not np.isfinite(k).all() or not np.isfinite(pose).all():
            raise ValueError('Invalid camera geometry')
        if (frame['width'], frame['height']) != size:
            raise ValueError('Camera/image dimensions disagree')
        split = frame.get('split') if shared else ('test' if idx % 30 >= 24 else 'train')
        if split not in counts:
            raise ValueError('Unknown camera split')
        counts[split] += 1
    if min(counts.values()) == 0:
        raise ValueError('Both training and holdout cameras are required for upstream validation')
    ply = source/'ground_points3d.ply'
    if not ply.is_file() or ply.stat().st_size == 0:
        raise FileNotFoundError(ply)
    return counts


def prepare(cfg, profile):
    import cv2
    import numpy as np
    from plyfile import PlyData, PlyElement
    from tqdm import tqdm
    source = source_directory(cfg, profile)
    if profile['source_mode'] == 'native_hugsim':
        print(json.dumps(validate_source(source), indent=2))
        return source
    ds = cfg['dataset']
    result = audit(ds, require_labels=True, require_ground=True)
    if result['missing_count']:
        raise ValueError('Shared data audit failed: ' + json.dumps(result))
    rows = image_records(ds)
    fronts = sorted((r for r in rows if r['camera'] == 'CAM_FRONT'), key=lambda r:r['timestamp'])
    if len(fronts) < 5:
        raise ValueError('At least five front frames are required')
    # Preserve every RoGS-selected frame. Order by per-camera temporal rank for holdout.
    ranked = []
    for cam_index, camera in enumerate(ds['camera_names']):
        camera_rows = sorted((r for r in rows if r['camera'] == camera), key=lambda r:r['timestamp'])
        ranked.extend((rank, cam_index, row) for rank, row in enumerate(camera_rows))
    ranked.sort(key=lambda x:x[:2])
    inputs = [Path(ds['road_gt_dir'])/(ds['clip_list'][0]+'.ply')]
    for row in rows:
        inputs.extend([Path(ds['base_dir'])/row['filename'], Path(ds['image_dir'])/row['label_filename']])
    meta = Path(ds['base_dir'])/('v1.0-'+ds['version'])
    inputs.extend(meta/(name+'.json') for name in ('sample_data','calibrated_sensor','ego_pose','sensor'))
    identity = {'dataset': ds, 'profile': profile, 'seed': cfg['seed'],
                'adapter_sha256': sha256_file(__file__),
                'shared_mask_sha256': sha256_file(Path(__file__).with_name('shared_mask.py')),
                'inputs': [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in inputs]}
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    if source.exists():
        manifest = source/'adapter_manifest.json'
        if not manifest.is_file() or json.loads(manifest.read_text()).get('identity_sha256') != digest:
            raise ValueError('Prepared directory differs from current inputs/protocol; choose a new prepared_root')
        validate_source(source, shared=True)
        print('Reusing HUGSIM prepared data:', source)
        return source
    tables = metadata_tables(ds)
    _, first_pose = global_camera(fronts[0], tables)
    inverse = np.linalg.inv(first_pose)
    front_poses = np.stack([inverse @ global_camera(row, tables)[1] for row in fronts])
    height = profile['initialization'].get('camera_height', 'auto_lidar')
    if height == 'auto_lidar':
        height = front_height(cfg, fronts[0], tables)
    height = float(height)
    if not np.isfinite(height) or height <= 0:
        raise ValueError('camera_height must be positive')
    source.parent.mkdir(parents=True, exist_ok=True)
    staged = source.with_name('.'+source.name+'-'+uuid.uuid4().hex[:8]+'.partial')
    staged.mkdir()
    try:
        frames = []
        size = (ds['image_width'], ds['image_height'])
        crop = size[1]//2 if profile.get('crop_top_half', True) else 0
        for rank, _, row in tqdm(ranked, desc='Preparing HUGSIM'):
            image = cv2.imread(str(Path(ds['base_dir'])/row['filename']))
            label = cv2.imread(str(Path(ds['image_dir'])/row['label_filename']), cv2.IMREAD_UNCHANGED)
            if image is None or label is None or label.shape != image.shape[:2]:
                raise ValueError('Unreadable image or mismatched raw label: '+row['filename'])
            k, pose = global_camera(row, tables)
            k, pose = camera_geometry(k, pose, inverse, (image.shape[1], image.shape[0]), size, crop)
            resized = cv2.resize(image, size, interpolation=cv2.INTER_LINEAR)[crop:]
            labels = convert_labels(cv2.resize(label, size, interpolation=cv2.INTER_NEAREST), row['camera'], profile['mask_policy'], crop)
            rel = Path('images')/row['camera']/(row['token']+'.png')
            safe_relative(rel.as_posix())
            target = staged/rel; target.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(target), resized):
                raise OSError('Could not write '+str(target))
            semantic = staged/'semantics'/row['camera']/(row['token']+'.npy')
            semantic.parent.mkdir(parents=True, exist_ok=True); np.save(semantic, labels)
            frames.append({'rgb_path': rel.as_posix(), 'camtoworld': pose.tolist(), 'intrinsics': k.tolist(),
                           'width': size[0], 'height': size[1]-crop, 'timestamp': (row['timestamp']-fronts[0]['timestamp'])/1e6,
                           'dynamics': {}, 'split': 'test' if rank % 5 == 4 else 'train', 'source': row})
        vertices = PlyData.read(inputs[0])['vertex']
        xyz = np.column_stack([vertices[k] for k in ('x','y','z')])
        rgb = np.column_stack([vertices[k] for k in ('r','g','b')])
        if not np.isfinite(xyz).all() or not np.isfinite(rgb).all() or len(xyz) < 2:
            raise ValueError('Invalid shared road point cloud')
        rng = np.random.default_rng(cfg['seed'])
        selected = rng.choice(len(xyz), min(len(xyz), profile['initialization']['max_points']), replace=False)
        xyz = xyz[selected] @ inverse[:3,:3].T + inverse[:3,3]
        xyz = project_ground(xyz, front_poses, height)
        cloud = np.empty(len(xyz), dtype=[('x','f4'),('y','f4'),('z','f4'),('red','u1'),('green','u1'),('blue','u1')])
        for j,key in enumerate(('x','y','z')):cloud[key]=xyz[:,j]
        rgb = (rgb[selected].clip(0,1)*255).astype(np.uint8)
        for j,key in enumerate(('red','green','blue')):cloud[key]=rgb[:,j]
        PlyData([PlyElement.describe(cloud,'vertex')]).write(staged/'ground_points3d.ply')
        (staged/'meta_data.json').write_text(json.dumps({'camera_model':'OPENCV','inv_pose':inverse.tolist(),'verts':{},'frames':frames},indent=2)+'\n')
        manifest = {'identity_sha256': digest, 'identity':identity, 'source_mode':'shared_rogs',
                    'world_to_hugsim':inverse.tolist(), 'camera_height':height, 'ground_points':len(xyz),
                    'initialization':'RoGS fused LiDAR points projected onto HUGSIM nearest-camera planes; differs from original depth initialization',
                    'semantic_contract':'20 channels retained; 0=shared ground except sidewalk, 1=sidewalk, 19=masked',
                    'segmentation_manifest':None}
        seg = Path(ds['image_dir'])/'segmentation_manifest.json'
        if seg.is_file():manifest['segmentation_manifest']=json.loads(seg.read_text())
        (staged/'adapter_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
        validate_source(staged, shared=True)
        staged.rename(source)
    except BaseException:
        shutil.rmtree(staged)
        raise
    print('Prepared HUGSIM data:',source)
    return source
