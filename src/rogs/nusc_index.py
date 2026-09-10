"""Lightweight sample/sweep indexing without importing CUDA or the nuScenes SDK."""
import json
from pathlib import Path, PurePosixPath


def safe_relative(value):
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value:
        raise ValueError("Expected safe dataset-relative path: " + value)
    return path


def label_path(image_path):
    safe_relative(image_path)
    if "/CAM" not in image_path or not image_path.endswith(".jpg"):
        raise ValueError("Unexpected nuScenes image path: " + image_path)
    return image_path.replace("/CAM", "/seg_CAM").replace(".jpg", ".png")


def image_records(dataset):
    meta = Path(dataset["base_dir"]) / ("v1.0-" + dataset["version"])
    def read(name):
        return json.loads((meta / (name + ".json")).read_text())
    scenes = {s["token"]: s["name"] for s in read("scene")}
    requested = dataset["clip_list"]
    missing = set(requested) - set(scenes.values())
    if missing:
        raise ValueError("Unknown scenes: " + ", ".join(sorted(missing)))
    # scene names are matched exactly; standard nuScenes names match upstream.
    samples = read("sample")
    sd = {s["token"]: s for s in read("sample_data")}
    calibrated = {s["token"]: s for s in read("calibrated_sensor")}
    sensors = {s["token"]: s for s in read("sensor")}
    # sample.json on disk has no SDK-generated 'data' field.
    keyframes = {}
    for row in sd.values():
        if row["is_key_frame"]:
            channel = sensors[calibrated[row["calibrated_sensor_token"]]["sensor_token"]]["channel"]
            keyframes[(row["sample_token"], channel)] = row["token"]
    output = []
    for scene in requested:
        ordered = sorted((s for s in samples if scenes[s["scene_token"]] == scene), key=lambda s: s["timestamp"])
        for sample in ordered:
            for camera in dataset["camera_names"]:
                row = sd[keyframes[(sample["token"], camera)]]
                visited = set()
                while True:
                    if row["token"] in visited:
                        raise ValueError("Cyclic sample_data chain")
                    visited.add(row["token"])
                    safe_relative(row["filename"])
                    output.append({"scene": scene, "camera": camera, "token": row["token"],
                                   "timestamp": row["timestamp"], "filename": row["filename"],
                                   "label_filename": label_path(row["filename"])})
                    if not row["next"]:
                        break
                    row = sd[row["next"]]
                    if row["is_key_frame"]:
                        break
    if not output:
        raise ValueError("No images selected")
    return output


def audit(dataset, require_labels=True, require_ground=True):
    records = image_records(dataset)
    missing = []
    counts = {}
    for row in records:
        counts[row["camera"]] = counts.get(row["camera"], 0) + 1
        candidates = [Path(dataset["base_dir"]) / row["filename"]]
        if require_labels:
            candidates.append(Path(dataset["image_dir"]) / row["label_filename"])
        for path in candidates:
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    if require_ground:
        for scene in dataset["clip_list"]:
            path = Path(dataset["road_gt_dir"]) / (scene + ".ply")
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    return {"images": len(records), "per_camera": counts, "missing_count": len(missing),
            "missing_examples": missing[:20]}

