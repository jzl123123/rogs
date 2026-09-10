"""Local Mask2Former inference exporting the original 65 Mapillary class IDs."""
import json
from pathlib import Path
import sys
from time import perf_counter

import yaml

from rogs.config import project_root, resolve_path
from rogs.nusc_index import audit, image_records
from rogs.provenance import sha256_file


def load_local_mask2former():
    source = project_root() / "thirdparty/Mask2Former"
    sys.path.insert(0, str(source))
    import mask2former
    try:
        Path(mask2former.__file__).resolve().relative_to(source.resolve())
    except ValueError:
        raise ValueError("Mask2Former was loaded from a different checkout")
    return mask2former


def profile_paths(filename):
    filename = Path(filename).resolve()
    profile = yaml.safe_load(filename.read_text())
    if profile.get("label_space") != "mapillary_vistas_65":
        raise ValueError("RoGS reference remapping expects mapillary_vistas_65")
    for name in ("config_file", "checkpoint"):
        profile[name] = resolve_path(profile[name], filename.parent)
        if not Path(profile[name]).is_file():
            raise FileNotFoundError(profile[name])
    digest = sha256_file(profile["checkpoint"])
    expected = profile.get("checkpoint_sha256")
    if expected and expected != digest:
        raise ValueError("Segmentation checkpoint SHA256 mismatch")
    profile["checkpoint_sha256"] = digest
    return profile


def build_predictor(cfg, profile):
    module = load_local_mask2former()
    from detectron2.config import get_cfg
    from detectron2.engine import DefaultPredictor
    from detectron2.projects.deeplab import add_deeplab_config
    model_cfg = get_cfg()
    add_deeplab_config(model_cfg)
    module.add_maskformer2_config(model_cfg)
    model_cfg.merge_from_file(profile["config_file"])
    if model_cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES != 65:
        raise ValueError("Expected 65 source semantic classes")
    model_cfg.MODEL.WEIGHTS = profile["checkpoint"]
    model_cfg.MODEL.DEVICE = cfg["device"]
    model_cfg.MODEL.MASK_FORMER.TEST.SEMANTIC_ON = True
    model_cfg.MODEL.MASK_FORMER.TEST.INSTANCE_ON = False
    model_cfg.MODEL.MASK_FORMER.TEST.PANOPTIC_ON = False
    model_cfg.freeze()
    return DefaultPredictor(model_cfg), model_cfg.dump()


def validate_cached_label(image_path, output):
    # Read only the JPEG header; fully decode the much smaller label PNG.
    from PIL import Image
    with Image.open(image_path) as image:
        size = image.size
    with Image.open(output) as label:
        if label.mode != "L" or label.size != size or label.getextrema()[1] > 64:
            raise ValueError("Invalid existing label: " + str(output))


def pending_records(records, base, destination):
    pending = []
    for row in records:
        output = destination / row["label_filename"]
        if output.exists():
            validate_cached_label(base / row["filename"], output)
        else:
            pending.append(row)
    return pending


def write_manifest(path, record):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n")
    temporary.replace(path)


def segment(cfg, profile_file):
    started = perf_counter()
    profile = profile_paths(profile_file)
    result = audit(cfg["dataset"], require_labels=False, require_ground=False)
    if result["missing_count"]:
        raise ValueError("Image audit failed: " + json.dumps(result))
    records = image_records(cfg["dataset"])
    base = Path(cfg["dataset"]["base_dir"])
    destination = Path(cfg["dataset"]["image_dir"])
    manifest_file = destination / "segmentation_manifest.json"
    identity = {"profile": profile, "config_sha256": sha256_file(profile["config_file"]),
                "inference": "DefaultPredictor sem_seg argmax, single scale, no TTA",
                "paper_checkpoint_equivalence": "not established",
                "mask2former_commit": json.loads((project_root()/"thirdparty/sources.lock.json").read_text())["sources"]["Mask2Former"]["commit"]}
    previous = {}
    if manifest_file.exists():
        previous = json.loads(manifest_file.read_text())
        if previous.get("identity") != identity:
            raise ValueError("Segmentation output directory contains a different model manifest; select a separate directory")
    elif any((destination / r["label_filename"]).exists() for r in records):
        raise ValueError("Refusing to mix generated labels with untracked existing labels; select a separate image_dir")
    pending = pending_records(records, base, destination)
    stats = {"cached_images": len(records) - len(pending), "processed_images": 0,
             "preflight_seconds": perf_counter() - started, "model_setup_seconds": 0.0,
             "image_read_seconds": 0.0, "prediction_seconds": 0.0, "label_write_seconds": 0.0}
    record = {"identity": identity, "status": "running", "selected_images": len(records),
              "scenes": cfg["dataset"]["clip_list"], "last_run": stats}
    if "resolved_model_config" in previous:
        record["resolved_model_config"] = previous["resolved_model_config"]
    print(f"Segmentation: {len(records)} selected, {stats['cached_images']} cached, {len(pending)} pending", flush=True)
    write_manifest(manifest_file, record)
    try:
        if pending:
            setup_started = perf_counter()
            import cv2
            import numpy as np
            from tqdm import tqdm
            predictor, resolved = build_predictor(cfg, profile)
            stats["model_setup_seconds"] = perf_counter() - setup_started
            record["resolved_model_config"] = resolved
            write_manifest(manifest_file, record)
            for row in tqdm(pending, desc="Mask2Former"):
                tick = perf_counter()
                image = cv2.imread(str(base / row["filename"]))
                if image is None:
                    raise ValueError("Unreadable image: " + row["filename"])
                stats["image_read_seconds"] += perf_counter() - tick
                tick = perf_counter()
                # CPU transfer completes the GPU work before this timer stops.
                label = predictor(image)["sem_seg"].argmax(dim=0).to("cpu").numpy().astype(np.uint8)
                stats["prediction_seconds"] += perf_counter() - tick
                if label.shape != image.shape[:2] or label.max() > 64:
                    raise ValueError("Unexpected model label output")
                tick = perf_counter()
                output = destination / row["label_filename"]
                output.parent.mkdir(parents=True, exist_ok=True)
                temporary = output.with_name(output.stem + ".partial.png")
                if not cv2.imwrite(str(temporary), label):
                    raise OSError("Failed to write " + str(temporary))
                temporary.replace(output)
                stats["label_write_seconds"] += perf_counter() - tick
                stats["processed_images"] += 1
        else:
            print("All selected labels are cached; skipped model loading and GPU inference.", flush=True)
        record["status"] = "complete"
    except BaseException as exc:
        record["status"] = "failed"
        record["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        stats["total_seconds"] = perf_counter() - started
        stats["end_to_end_images_per_second"] = stats["processed_images"] / stats["total_seconds"]
        write_manifest(manifest_file, record)
        print("Segmentation timing: " + json.dumps(stats), flush=True)
    return record
