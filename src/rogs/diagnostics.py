import importlib
import json
import platform
import shutil


def diagnose(smoke=False, segmentation=False):
    result = {"platform": platform.platform(), "nvcc": shutil.which("nvcc"), "modules": {}}
    names = ["torch", "pytorch3d", "diff_gaussian_rasterization", "diff_gs_label", "nuscenes"]
    if segmentation:
        names += ["detectron2", "MultiScaleDeformableAttention"]
    failed = False
    for name in names:
        try:
            module = importlib.import_module(name)
            result["modules"][name] = {"file": module.__file__, "version": getattr(module, "__version__", None)}
        except Exception as error:
            failed = True
            result["modules"][name] = {"error": str(error)}
    if not failed:
        import torch
        result["cuda_available"] = torch.cuda.is_available()
        result["torch_cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            result["gpu"] = torch.cuda.get_device_name()
        if segmentation:
            from rogs.segmentation import load_local_mask2former
            result["mask2former"] = load_local_mask2former().__file__
    print(json.dumps(result, indent=2))
    if failed:
        raise SystemExit(2)
    if smoke:
        cuda_smoke()


def cuda_smoke():
    import torch
    from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
    from diff_gs_label import GaussianRasterizer as LabelRasterizer
    from diff_gaussian_rasterization.scene.cameras import PerspectiveCamera, OrthographicCamera
    if not torch.cuda.is_available():
        raise SystemExit("CUDA smoke check requires NVIDIA GPU")
    device = "cuda"
    k = torch.tensor([[32., 0, 16], [0, 32., 16], [0, 0, 1]], device=device)
    r, t = torch.eye(3, device=device), torch.zeros(3, device=device)
    cameras = [PerspectiveCamera(r, t, k, 32, 32, 0.1, 10., device),
               OrthographicCamera(r, t, 32, 32, 0.1, 10., -1., 1., 1., -1., device)]
    for camera_type, camera in enumerate(cameras):
        for channels, rasterizer_class in ((3, GaussianRasterizer), (7, LabelRasterizer)):
            settings = GaussianRasterizationSettings(
                image_height=32, image_width=32, tanfovx=0.5 if camera_type == 0 else 0.,
                tanfovy=0.5 if camera_type == 0 else 0., bg=torch.zeros(channels, device=device),
                scale_modifier=1., viewmatrix=camera.world_view_transform,
                projmatrix=camera.full_proj_transform, sh_degree=0, campos=camera.camera_center,
                prefiltered=False, camera_type=camera_type, debug=False)
            xyz = torch.tensor([[0., 0., 3.]], device=device, requires_grad=True)
            feature = torch.full((1, channels), 0.5, device=device, requires_grad=True)
            output, radii, depth, alpha = rasterizer_class(raster_settings=settings)(
                means3D=xyz, means2D=torch.zeros_like(xyz, requires_grad=True), shs=None,
                colors_precomp=feature, opacities=torch.full((1, 1), 0.8, device=device),
                scales=torch.tensor([[0.2, 0.2, 0.]], device=device),
                rotations=torch.tensor([[1., 0., 0., 0.]], device=device), cov3D_precomp=None)
            assert output.shape == (channels, 32, 32) and (radii > 0).any()
            assert torch.isfinite(output).all() and torch.isfinite(depth).all()
            output.sum().backward()
            assert feature.grad is not None and torch.isfinite(feature.grad).all() and feature.grad.abs().sum() > 0
            assert xyz.grad is not None and torch.isfinite(xyz.grad).all()
            torch.cuda.synchronize()
            print("OK CUDA render + backward:", "perspective" if camera_type == 0 else "orthographic", channels, "channels")

