"""Export an orthographic ground map using HUGSIM's own rasterizer."""
import argparse
import json
import math
from pathlib import Path
import numpy as np


def bev_geometry(points_world, resolution, margin=0.5, max_pixels=16000000):
    if not math.isfinite(resolution) or resolution <= 0:
        raise ValueError('BEV resolution must be positive')
    low=points_world.min(axis=0);high=points_world.max(axis=0)
    xmin,ymin=low[:2]-margin;xmax,ymax=high[:2]+margin
    width=math.ceil((xmax-xmin)/resolution);height=math.ceil((ymax-ymin)/resolution)
    if width*height>max_pixels:
        raise ValueError(f'BEV has {width*height} pixels; increase --resolution (limit {max_pixels})')
    pose=np.array([[1,0,0,xmin],[0,-1,0,ymax],[0,0,-1,high[2]+5],[0,0,0,1]],dtype=float)
    k=np.array([[1/resolution,0,0],[0,1/resolution,0],[0,0,1]],dtype=float)
    return pose,k,width,height,{'xmin':float(xmin),'ymax':float(ymax),'resolution':resolution,
                             'width':width,'height':height,'camera_z':float(high[2]+5),
                             'far_plane':float(high[2]-low[2]+10)}


def export(run, resolution=0.05):
    import torch
    from PIL import Image
    import yaml
    from gsplat.rendering import rasterization
    from rogs.hugsim.ground_model import GroundModel
    from rogs.hugsim.runner import check_runtime
    run=Path(run)
    cfg=yaml.safe_load((run/'hugsim.resolved.yaml').read_text())
    torch.cuda.set_device(torch.device(cfg['device']))
    check_runtime()
    metadata=json.loads((Path(cfg['source_path'])/'meta_data.json').read_text())
    if 'inv_pose' not in metadata:
        raise ValueError('World-aligned BEV requires meta_data.json inv_pose (world to HUGSIM frame)')
    world_to_hugsim=np.asarray(metadata['inv_pose'],dtype=float)
    state,iteration=torch.load(run/'ckpts'/f"ground_chkpnt{cfg['train']['iterations']}.pth",map_location='cuda',weights_only=False)
    model=GroundModel(cfg['model']['sh_degree'],model_args=state)
    h2w=np.linalg.inv(world_to_hugsim)
    points=model.get_xyz.detach().cpu().numpy().astype(float)
    world=points@h2w[:3,:3].T+h2w[:3,3]
    pose,k,width,height,geometry=bev_geometry(world,resolution)
    view=np.linalg.inv(world_to_hugsim@pose)
    with torch.no_grad():
        values,alpha,_=rasterization(means=model.get_xyz,quats=model.get_rotation,scales=model.get_scaling,
            opacities=model.get_opacity[:,0],colors=model.get_features,
            viewmats=torch.tensor(view,dtype=torch.float32,device='cuda')[None],
            Ks=torch.tensor(k,dtype=torch.float32,device='cuda')[None],width=width,height=height,
            smts=model.get_3D_features[None],render_mode='RGB+ED+S',sh_degree=model.active_sh_degree,
            ortho=True,near_plane=0.01,far_plane=geometry['far_plane'],packed=False,
            backgrounds=torch.zeros((1,3),device='cuda'))
        values=values[0].cpu().numpy();alpha=alpha[0,...,0].cpu().numpy()
    output=run/'bev';output.mkdir(exist_ok=True)
    valid=alpha>0.5
    Image.fromarray((np.clip(values[:,:,:3],0,1)*255).astype(np.uint8)).save(output/'rgb.png')
    semantic=values[:,:,4:24].argmax(axis=-1).astype(np.uint8);semantic[~valid]=19
    Image.fromarray(semantic).save(output/'semantic.png')
    Image.fromarray(valid.astype(np.uint8)*255).save(output/'mask.png')
    elevation=geometry['camera_z']-values[:,:,3];elevation[~valid]=np.nan
    np.save(output/'height.npy',elevation)
    geometry.update(iteration=iteration,frame='nuScenes global XYZ (or native inv_pose source world)',
        pixel_center='x=xmin+(column+0.5)*resolution; y=ymax-(row+0.5)*resolution',
        semantic='HUGSIM 20 channels; shared mode supervises 0=ground except sidewalk, 1=sidewalk; 19=invalid',
        alpha_threshold=0.5,rendering='HUGSIM_splat orthographic; export-only, no training changes')
    (output/'map.json').write_text(json.dumps(geometry,indent=2)+'\n')
    print('Exported HUGSIM BEV:',output)
    return output


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',required=True)
    parser.add_argument('--resolution',type=float,default=.05)
    args=parser.parse_args();export(args.run,args.resolution)
