"""Recheck the frozen 4:1 split for cross-partition duplicate and scene cues."""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggcv.core import OUT, ROOT, images, make_split, phash, read, save_json


def source_id(path: str) -> int | None:
    match = re.search(r"not_damaged_(\d+)", path)
    return int(match.group(1)) if match else None


def main() -> None:
    split = json.loads((OUT / "split.json").read_text())
    assert split == make_split(images()), "Saved split no longer matches split code"
    dev, evaluation = set(split["dev"]), set(split["eval"])
    assert len(dev) == 88 and len(evaluation) == 22 and not dev & evaluation
    for group in split["groups"]:
        assert set(group) <= dev or set(group) <= evaluation

    features = {}
    for relative in dev | evaluation:
        path = ROOT / relative
        image = read(path)
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1], None, [16, 8], [0, 180, 0, 256])
        features[relative] = {
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "phash": phash(image),
            "histogram": cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX),
            "source_id": source_id(relative),
        }
    minimum_distance = 64
    cross_risk_pairs = []
    exact_duplicates = 0
    for a in dev:
        for b in evaluation:
            x, y = features[a], features[b]
            distance = int(np.count_nonzero(x["phash"] != y["phash"]))
            minimum_distance = min(minimum_distance, distance)
            exact_duplicates += x["sha256"] == y["sha256"]
            adjacent = (x["source_id"] is not None and y["source_id"] is not None
                        and abs(x["source_id"] - y["source_id"]) <= 2 and distance <= 20)
            same_color_profile = distance <= 20 and cv2.compareHist(x["histogram"], y["histogram"], cv2.HISTCMP_CORREL) >= .92
            if distance <= 6 or adjacent or same_color_profile:
                cross_risk_pairs.append([a, b, distance])
    assert not exact_duplicates and not cross_risk_pairs
    result = {
        "dev_count": len(dev), "eval_count": len(evaluation),
        "ratio": f"{len(dev)}:{len(evaluation)} = 4:1",
        "scene_groups": split["group_count"],
        "cross_exact_duplicate_files": exact_duplicates,
        "cross_split_min_phash_distance": minimum_distance,
        "cross_split_pairs_meeting_group_rules": len(cross_risk_pairs),
        "reviewed_example_151_152_grouped": any(
            any("not_damaged_151_jpg" in p for p in group) and any("not_damaged_152_jpg" in p for p in group)
            for group in split["groups"]
        ),
        "reviewed_example_113_135_grouped": any(
            any("not_damaged_113_jpg" in p for p in group) and any("not_damaged_135_jpg" in p for p in group)
            for group in split["groups"]
        ),
        "limitation": "Visual similarity rules cannot guarantee independent capture sessions; manual scene IDs and new human masks would be stronger.",
    }
    save_json(OUT / "split_audit.json", result)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
