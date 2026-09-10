#!/usr/bin/env python3
"""Compare outputs from two trusted local RoGS run bundles on the same scene/environment."""
import argparse
import json
from pathlib import Path


def compare(left, right, atol, rtol, prefix="checkpoint"):
    import numpy as np
    import torch
    if torch.is_tensor(left):
        if not torch.is_tensor(right) or left.shape != right.shape:
            return [{"path": prefix, "error": "tensor type/shape mismatch"}]
        a, b = left.detach().cpu(), right.detach().cpu()
        # Infinite logits exist in the upstream opacity initializer; compare same-signed inf literally.
        ok = torch.allclose(a, b, atol=atol, rtol=rtol, equal_nan=False)
        if ok:
            return []
        finite = torch.isfinite(a) & torch.isfinite(b)
        delta = (a[finite].double() - b[finite].double()).abs()
        return [{"path": prefix, "max_abs_error": delta.max().item() if delta.numel() else None}]
    if isinstance(left, np.ndarray):
        if not isinstance(right, np.ndarray) or left.shape != right.shape or not np.allclose(left, right, atol=atol, rtol=rtol):
            return [{"path": prefix, "error": "array mismatch"}]
        return []
    if isinstance(left, dict):
        if not isinstance(right, dict) or left.keys() != right.keys():
            return [{"path": prefix, "error": "dictionary keys mismatch"}]
        return [e for key in left for e in compare(left[key], right[key], atol, rtol, prefix + "." + str(key))]
    if isinstance(left, (tuple, list)):
        if not isinstance(right, (tuple, list)) or len(left) != len(right):
            return [{"path": prefix, "error": "sequence length mismatch"}]
        return [e for i, pair in enumerate(zip(left, right)) for e in compare(*pair, atol, rtol, prefix + "[{}]".format(i))]
    return [] if left == right else [{"path": prefix, "error": "value mismatch"}]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("packaged", type=Path)
    parser.add_argument("--atol", type=float, default=1e-6)
    parser.add_argument("--rtol", type=float, default=1e-5)
    args = parser.parse_args()
    import torch
    def checkpoint(root):
        paths = list(root.rglob("final.pth"))
        if len(paths) != 1:
            raise ValueError("Expected exactly one final.pth in " + str(root))
        return torch.load(paths[0], map_location="cpu")
    errors = compare(checkpoint(args.baseline), checkpoint(args.packaged), args.atol, args.rtol)
    for name in ("exposure.pth",):
        a, b = list(args.baseline.rglob(name)), list(args.packaged.rglob(name))
        if len(a) != len(b) or len(a) > 1:
            errors.append({"path": name, "error": "file count mismatch"})
        elif a:
            errors.extend(compare(torch.load(a[0], map_location="cpu"), torch.load(b[0], map_location="cpu"), args.atol, args.rtol, name))
    import cv2
    import numpy as np
    for name in ("bev_image.png", "bev_label.png", "bev_mask.png"):
        a = list(args.baseline.glob("**/images/final/" + name))
        b = list(args.packaged.glob("**/images/final/" + name))
        if len(a) != 1 or len(b) != 1:
            errors.append({"path": name, "error": "missing or ambiguous final output"})
        elif not np.array_equal(cv2.imread(str(a[0]), -1), cv2.imread(str(b[0]), -1)):
            errors.append({"path": name, "error": "pixel values differ"})
    print(json.dumps({"passed": not errors, "atol": args.atol, "rtol": args.rtol, "differences": errors}, indent=2))
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
