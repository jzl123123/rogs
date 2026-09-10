"""Single entry point; heavy GPU imports are deferred until an operation needs them."""
import argparse
import json
from pathlib import Path
import runpy
import subprocess
import sys

from rogs.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(prog="rogs", description="Reference-aligned RoGS workflow")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("train", "baseline", "preprocess", "segment", "check"):
        item = sub.add_parser(command)
        item.add_argument("--config", required=True)
        if command == "segment":
            item.add_argument("--model-profile", required=True)
        if command == "check":
            item.add_argument("--stage", choices=("images", "labels", "train"), default="train")
    doctor = sub.add_parser("doctor")
    doctor.add_argument("--smoke", action="store_true", help="Exercise CUDA RGB/semantic forward and backward")
    doctor.add_argument("--segmentation", action="store_true", help="Also load the local Mask2Former stack")
    args = parser.parse_args(argv)
    try:
        if args.command == "doctor":
            from rogs.diagnostics import diagnose
            return diagnose(args.smoke, args.segmentation)
        cfg = load_config(args.config)
        if args.command == "segment":
            from rogs.segmentation import segment
            segment(cfg, args.model_profile)
            return
        from rogs.nusc_index import audit
        stage = args.stage if args.command == "check" else ("labels" if args.command == "preprocess" else "train")
        result = audit(cfg["dataset"], require_labels=stage != "images", require_ground=stage == "train")
        print(json.dumps(result, indent=2))
        if result["missing_count"]:
            raise ValueError("Required data missing. No frames will be silently dropped by this entry point.")
        if args.command == "check":
            return
        if args.command == "preprocess":
            ds = cfg["dataset"]
            ground = Path(ds["road_gt_dir"])
            if ground.name != "nuScenes_road_gt":
                raise ValueError("Unmodified preprocessor requires road_gt_dir ending in nuScenes_road_gt")
            old_argv = sys.argv
            sys.argv = ["rogs.preprocess.process_nusc", "--nusc_root", ds["base_dir"],
                        "--seg_root", ds["image_dir"], "--save_root", str(ground.parent),
                        "-v", ds["version"], "--scene_names"] + ds["clip_list"]
            try:
                runpy.run_module("rogs.preprocess.process_nusc", run_name="__main__")
            finally:
                sys.argv = old_argv
            return
        if args.command in ("train", "baseline"):
            import torch
            if not torch.cuda.is_available():
                raise ValueError("RoGS training requires an NVIDIA CUDA runtime")
            from rogs.provenance import run_bundle
            config, directory = run_bundle(cfg)
            print("Run bundle:", directory)
            if args.command == "baseline":
                from rogs.config import project_root
                upstream = project_root() / "thirdparty/rogs_upstream"
                subprocess.run([sys.executable, str(upstream / "train.py"), "--config", config["file"]],
                               cwd=upstream, check=True)
                return
            import addict
            from rogs.train import train
            train(addict.Dict(config))
    except (ValueError, FileNotFoundError, KeyError, ImportError) as error:
        parser.exit(2, "rogs: {}\n".format(error))


if __name__ == "__main__":
    main()
