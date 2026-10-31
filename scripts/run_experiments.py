"""Tune only on dev, then evaluate the frozen winner once on eval."""
from __future__ import annotations

import argparse
import csv
import html
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggcv.core import OUT, ROOT, classical, components, images, key, make_split, metrics, overlay, read, save_json


TRIALS = [
    {"name": "T1 global bright", "method": "global", "light": 165, "sat": 255, "min_area": 0.003, "reason": "Simple light-egg baseline"},
    {"name": "T2 Otsu", "method": "otsu", "light": 0, "sat": 255, "min_area": 0.003, "reason": "Adaptive global threshold for varying exposure"},
    {"name": "T3 color and shape", "method": "refined", "light": 145, "sat": 100, "min_area": 0.003, "reason": "Reject colored background and non-elliptical regions"},
    {"name": "T4 stricter color", "method": "refined", "light": 160, "sat": 75, "min_area": 0.003, "reason": "Reduce bright clutter false positives"},
    {"name": "T5 small components", "method": "refined", "light": 145, "sat": 100, "min_area": 0.001, "reason": "Recover distant eggs"},
    {"name": "T6 high brightness", "method": "refined", "light": 180, "sat": 100, "min_area": 0.003, "reason": "Test stronger foreground contrast"},
    {"name": "T7 graph cut 160", "method": "grabcut", "light": 160, "sat": 110, "min_area": 0.003, "reason": "Grow bright egg cores to full boundaries using local color models"},
    {"name": "T8 graph cut 180", "method": "grabcut", "light": 180, "sat": 110, "min_area": 0.003, "reason": "Stricter seeds before local graph cut"},
    {"name": "T9 graph cut with empty-mask fallback", "method": "grabcut_fallback", "light": 180, "sat": 110, "min_area": 0.003, "reason": "Use a conservative global mask only when graph cut finds no egg"},
]


def instance_scores(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    t = components(truth, 250)
    p = components(pred, 250)
    if not t and not p:
        return {"object_precision": 1.0, "object_recall": 1.0, "count_error": 0.0}
    if not t or not p:
        return {"object_precision": 0.0, "object_recall": 0.0, "count_error": float(abs(len(t)-len(p)))}
    overlap = np.zeros((len(t), len(p)))
    for i, a in enumerate(t):
        for j, b in enumerate(p):
            intersection = np.count_nonzero(a & b)
            overlap[i, j] = intersection / max(np.count_nonzero(a | b), 1)
    rows, cols = linear_sum_assignment(-overlap)
    matches = sum(overlap[i, j] >= .5 for i, j in zip(rows, cols))
    return {"object_precision": matches / len(p), "object_recall": matches / len(t), "count_error": float(abs(len(t)-len(p)))}


def assess(paths: list[Path], trial: dict, save: bool = False) -> list[dict]:
    rows = []
    for path in paths:
        mask_path = OUT / "reference" / f"{key(path)}.png"
        if not mask_path.exists():
            raise FileNotFoundError(mask_path)
        truth = (cv2.imread(str(mask_path), 0) > 0).astype(np.uint8)
        image = read(path)
        pred = classical(image, trial["method"], trial["light"], trial["sat"], trial["min_area"])
        row = {"image": str(path.relative_to(ROOT)), "key": key(path), **metrics(truth, pred), **instance_scores(truth, pred), "reference_pixels": int(truth.sum()), "predicted_pixels": int(pred.sum())}
        rows.append(row)
        if save:
            cv2.imwrite(str(OUT / "predictions" / f"{key(path)}.png"), pred * 255)
            cv2.imwrite(str(OUT / "explorer" / f"{key(path)}.jpg"), overlay(image, truth, pred))
    return rows


def summary(rows: list[dict]) -> dict[str, float]:
    return {k: float(np.mean([r[k] for r in rows])) for k in ["iou", "dice", "precision", "recall", "pixel_accuracy", "balanced_accuracy", "specificity", "object_precision", "object_recall", "count_error"]}


def bootstrap_ci(rows: list[dict], metric: str, seed: int = 31) -> list[float]:
    values = np.array([r[metric] for r in rows])
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, (5000, len(values)), replace=True).mean(axis=1)
    return [float(np.quantile(draws, .025)), float(np.quantile(draws, .975))]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def explorer(rows: list[dict]) -> None:
    cards = []
    for row in sorted(rows, key=lambda x: x["iou"]):
        img = html.escape(row["key"])
        cards.append(f'<article data-iou="{row["iou"]:.4f}"><img src="{img}.jpg" loading="lazy"><div><b>{img}</b><span>IoU {row["iou"]:.3f} · Dice {row["dice"]:.3f}</span><small>Reference {row["reference_pixels"]:,} px · Prediction {row["predicted_pixels"]:,} px</small></div></article>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Egg segmentation error explorer</title>
<style>body{font:16px system-ui;background:#101827;color:#eee;margin:0;padding:2rem}header{max-width:900px;margin:auto auto 2rem}h1{font-size:2.2rem}p{color:#b8c8dc}input{width:100%;max-width:500px;padding:.8rem;border-radius:8px;border:0}main{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:1rem}article{background:#202d41;border-radius:12px;overflow:hidden}img{width:100%;aspect-ratio:1;object-fit:cover}article div{padding:1rem;display:grid;gap:.4rem}span,small{color:#c4d1e0}</style>
<header><h1>Egg segmentation error explorer</h1><p>Held-out evaluation set, worst IoU first. Green = overlap; red = reference only; blue = classical method only. These are comparisons with MobileSAM pseudo-labels.</p><input id="q" placeholder="Filter image ID"></header><main>''' + ''.join(cards) + '''</main><script>document.querySelector('#q').oninput=e=>document.querySelectorAll('article').forEach(x=>x.hidden=!x.textContent.toLowerCase().includes(e.target.value.toLowerCase()))</script></html>'''
    (OUT / "explorer" / "index.html").write_text(page)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=31)
    args = parser.parse_args()
    paths = images()
    split = make_split(paths, args.seed)
    save_json(OUT / "split.json", split)
    dev = [ROOT / p for p in split["dev"]]
    evaluation = [ROOT / p for p in split["eval"]]
    log = []
    for trial in TRIALS:
        rows = assess(dev, trial)
        result = {**trial, "dev": summary(rows), "decision": "pending"}
        log.append(result)
        print(trial["name"], result["dev"])
    winner = max(log, key=lambda row: row["dev"]["iou"])
    for result in log:
        result["decision"] = "kept: highest dev mean IoU" if result is winner else "discarded: lower dev mean IoU"
    save_json(OUT / "experiment_log.json", log)
    save_json(OUT / "selected_algorithm.json", {k: winner[k] for k in ["name", "method", "light", "sat", "min_area"]})
    (OUT / "predictions").mkdir(parents=True, exist_ok=True)
    (OUT / "explorer").mkdir(parents=True, exist_ok=True)
    for old in (OUT / "predictions").glob("*.png"):
        old.unlink()
    for old in (OUT / "explorer").glob("*.jpg"):
        old.unlink()
    eval_rows = assess(evaluation, winner, save=True)
    write_csv(OUT / "eval_per_image.csv", eval_rows)
    scores = summary(eval_rows)
    scores["iou_95_ci"] = bootstrap_ci(eval_rows, "iou")
    scores["dice_95_ci"] = bootstrap_ci(eval_rows, "dice")
    scores["eval_count"] = len(eval_rows)
    scores["dev_count"] = len(dev)
    save_json(OUT / "eval_summary.json", scores)
    baseline = next(trial for trial in TRIALS if trial["name"] == "T8 graph cut 180")
    save_json(OUT / "eval_baseline_summary.json", summary(assess(evaluation, baseline)))
    explorer(eval_rows)
    print("Winner", winner["name"], "Eval", scores)


if __name__ == "__main__":
    main()
