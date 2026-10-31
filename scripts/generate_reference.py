"""Generate MobileSAM pseudo-labels for all supplied images.

The SAM proposals are filtered without reference to the classical predictions.
Review the QC contact sheet before treating them as reference masks.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import SAM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggcv.core import OUT, classical, images, key, read, save_json, shape_features


def select_proposals(image: np.ndarray, proposals: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    h, w = image.shape[:2]
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    accepted: list[tuple[float, np.ndarray, dict]] = []
    for idx, raw in enumerate(proposals):
        mask = (raw > 0.5).astype(np.uint8)
        f = shape_features(mask)
        ratio = f["area"] / (h * w)
        if not 0.004 <= ratio <= 0.38 or f["border"] or f["solidity"] < 0.80 or f["circularity"] < 0.40 or not 0.35 <= f["aspect"] <= 2.8:
            continue
        brightness = float(lab[:, :, 0][mask.astype(bool)].mean())
        score = f["solidity"] + f["circularity"] + min(brightness / 200, 1.0) + min(ratio / 0.07, 1.0)
        f.update({"proposal": idx, "brightness": brightness, "score": score})
        accepted.append((score, mask, f))
    accepted.sort(key=lambda row: row[0], reverse=True)
    union = np.zeros((h, w), np.uint8)
    selected = []
    for _, mask, f in accepted:
        overlap = np.count_nonzero(mask & union) / max(np.count_nonzero(mask), 1)
        if overlap > 0.35:
            continue
        union |= mask
        selected.append(f)
    return union, selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    model_path = Path(__file__).resolve().parents[1] / "mobile_sam.pt"
    if not model_path.exists():
        raise SystemExit("Download mobile_sam.pt with Ultralytics SAM before running")
    model = SAM(str(model_path))
    paths = images()[:args.limit] if args.limit else images()
    rows = []
    masks_dir = OUT / "reference"
    masks_dir.mkdir(parents=True, exist_ok=True)
    for n, path in enumerate(paths, 1):
        image = read(path)
        destination = masks_dir / f"{key(path)}.png"
        if destination.exists():
            mask = (cv2.imread(str(destination), 0) > 0).astype(np.uint8)
            proposals = []
        else:
            result = model(str(path), device="cpu", imgsz=args.imgsz, verbose=False)[0]
            raw = np.zeros((0, *image.shape[:2]), np.uint8) if result.masks is None else result.masks.data.cpu().numpy()
            mask, proposals = select_proposals(image, raw)
            cv2.imwrite(str(destination), mask * 255)
        rows.append({"image": str(path.relative_to(Path(__file__).resolve().parents[1])), "mask": str(destination.relative_to(Path(__file__).resolve().parents[1])), "mask_pixels": int(mask.sum()), "selected": proposals})
        if n % 10 == 0 or n == len(paths):
            print(f"Reference masks {n}/{len(paths)}", flush=True)
            save_json(OUT / "reference_manifest.json", rows)
    sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
    save_json(OUT / "reference_provenance.json", {"model": "Ultralytics MobileSAM", "checkpoint": "mobile_sam.pt", "sha256": sha, "imgsz": args.imgsz, "selection": "area, solidity, circularity, border, aspect, and overlap filter; see source", "caution": "Pseudo-labels; not human ground truth"})
    sheet = Image.new("RGB", (1200, ((len(paths) + 7) // 8) * 180), "white")
    draw = ImageDraw.Draw(sheet)
    for i, path in enumerate(paths):
        image = read(path)
        mask = cv2.imread(str(masks_dir / f"{key(path)}.png"), 0) > 0
        preview = image.copy()
        preview[mask] = np.uint8(0.55 * preview[mask] + 0.45 * np.array([0, 220, 0]))
        rgb = Image.fromarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        rgb.thumbnail((145, 145))
        x, y = (i % 8) * 150, (i // 8) * 180
        sheet.paste(rgb, (x + (150 - rgb.width) // 2, y))
        draw.text((x + 2, y + 148), f"{i:03} {key(path)[-12:]}", fill="black")
    sheet.save(OUT / "reference_qc.jpg", quality=85)


if __name__ == "__main__":
    main()
