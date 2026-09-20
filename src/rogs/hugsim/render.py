import torch
from gsplat.rendering import rasterization
from rogs.hugsim.ground_model import GroundModel
from rogs.hugsim.cameras import Camera
from torch import Tensor

def render_ground(viewpoint:Camera, pc:GroundModel, bg_color:Tensor):
    xyz, opacities, scales = pc.get_xyz, pc.get_opacity, pc.get_scaling
    rotations, shs, feats3D = pc.get_rotation, pc.get_features, pc.get_3D_features

    K = viewpoint.K[None, :3, :3]
    renders, render_alphas, info = rasterization(
        means=xyz,
        quats=rotations,
        scales=scales,
        opacities=opacities[:, 0],
        colors=shs,
        viewmats=torch.linalg.inv(viewpoint.c2w)[None, ...],  # [C, 4, 4]
        Ks=K,  # [C, 3, 3]
        width=viewpoint.width,
        height=viewpoint.height,
        smts=feats3D[None, ...],
        render_mode='RGB+ED+S',
        sh_degree=pc.active_sh_degree,
        near_plane=0.01,
        far_plane=500,
        packed=False,
        backgrounds=bg_color[None, :],
    )

    renders = renders[0]
    rendered_image = renders[..., :3].permute(2,0,1)
    depth = renders[..., 3][None, ...]
    smt = renders[..., 4:(4+feats3D.shape[-1])].permute(2,0,1)

    return {"render": rendered_image,
            "feats": smt,
            "depth": depth,
            "opticalflow": None,
            "alphas": render_alphas,
            "viewspace_points": info["means2d"],
            "info": info,
            }
