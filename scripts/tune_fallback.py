"""Development-only check of deterministic classical fallback rules."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggcv.core import OUT, ROOT, classical, key, metrics, read


def main() -> None:
    split = json.loads((OUT / "split.json").read_text())
    cases: dict[str, list[float]] = {name: [] for name in (
        "graphcut", "global", "refined", "empty_to_global", "empty_to_refined",
        "small_to_global", "small_to_refined", "union_global", "union_refined",
    )}
    for relative in split["dev"]:
        path = ROOT / relative
        image = read(path)
        truth = (cv2.imread(str(OUT / "reference" / f"{key(path)}.png"), 0) > 0).astype(np.uint8)
        graphcut = classical(image, "grabcut", 180, 110, .003)
        global_mask = classical(image, "global", 165, 255, .003)
        refined = classical(image, "refined", 180, 100, .003)
        small = np.count_nonzero(graphcut) < .005 * graphcut.size
        masks = {
            "graphcut": graphcut,
            "global": global_mask,
            "refined": refined,
            "empty_to_global": global_mask if not np.any(graphcut) else graphcut,
            "empty_to_refined": refined if not np.any(graphcut) else graphcut,
            "small_to_global": global_mask if small else graphcut,
            "small_to_refined": refined if small else graphcut,
            "union_global": graphcut | global_mask,
            "union_refined": graphcut | refined,
        }
        for name, mask in masks.items():
            cases[name].append(metrics(truth, mask)["iou"])
    print(json.dumps({name: round(float(np.mean(scores)), 6) for name, scores in cases.items()}, indent=2))


if __name__ == "__main__":
    main()
