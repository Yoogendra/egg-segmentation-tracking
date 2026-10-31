"""Human-selected positive points for missed/partly missed MobileSAM eggs.

Coordinates are in the 640x640 supplied images. They were selected by visually
reviewing outputs/reference_qc.jpg, not by using classical predictions.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw
from ultralytics import SAM

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggcv.core import OUT, ROOT, key, save_json

POINTS = {
    5: [(305, 305)], 23: [(170, 390)], 25: [(525, 420)],
    32: [(175, 290)], 33: [(175, 305)], 36: [(170, 330)],
    42: [(390, 245)], 43: [(435, 190)], 45: [(465, 370)], 53: [(225, 305)],
    63: [(390, 285)], 70: [(430, 330)], 76: [(465, 305)], 79: [(260, 105)],
    86: [(295, 385)], 92: [(305, 445)], 98: [(155, 285)], 99: [(220, 350)],
    107: [(205, 405)],
}
BOXES = {
    26: [(250, 110, 525, 270)],
    28: [(25, 195, 190, 350)],
    100: [(260, 0, 450, 190), (275, 205, 480, 395)],
    102: [(70, 330, 315, 465), (380, 215, 590, 360)],
}


def main() -> None:
    rows = json.loads((OUT / "reference_manifest.json").read_text())
    model = SAM(str(ROOT / "mobile_sam.pt"))
    log = []
    for index, points in POINTS.items():
        row = rows[index]
        path = ROOT / row["image"]
        target = ROOT / row["mask"]
        original = (cv2.imread(str(target), 0) > 0).astype(np.uint8)
        combined = original.copy()
        for point in points:
            result = model(str(path), points=[list(point)], labels=[1], imgsz=640, device="cpu", verbose=False)[0]
            if result.masks is None:
                log.append({"index": index, "point": point, "accepted": False, "reason": "no mask"})
                continue
            candidate = (result.masks.data[0].cpu().numpy() > .5).astype(np.uint8)
            area = int(candidate.sum())
            if not 400 <= area <= 150000:
                log.append({"index": index, "point": point, "accepted": False, "area": area, "reason": "area outside range"})
                continue
            combined |= candidate
            log.append({"index": index, "point": point, "accepted": True, "area": area})
        cv2.imwrite(str(target), combined * 255)
        row["mask_pixels"] = int(combined.sum())
        row["human_prompted"] = points
        print(index, key(path), int(original.sum()), int(combined.sum()), flush=True)
    for index, boxes in BOXES.items():
        row = rows[index]
        path = ROOT / row["image"]
        target = ROOT / row["mask"]
        combined = (cv2.imread(str(target), 0) > 0).astype(np.uint8)
        for box in boxes:
            result = model(str(path), bboxes=list(box), imgsz=640, device="cpu", verbose=False)[0]
            if result.masks is not None:
                candidate = (result.masks.data[0].cpu().numpy() > .5).astype(np.uint8)
                combined |= candidate
                log.append({"index": index, "box": box, "accepted": True, "area": int(candidate.sum())})
        cv2.imwrite(str(target),combined*255)
        row["mask_pixels"] = int(combined.sum())
        row["human_box_reviewed"] = boxes
    save_json(OUT / "reference_manifest.json", rows)
    save_json(OUT / "reference_repairs.json", log)
    sheet = Image.new("RGB", (1200, ((len(rows)+7)//8)*180), "white")
    draw = ImageDraw.Draw(sheet)
    for i, row in enumerate(rows):
        image = cv2.imread(str(ROOT / row["image"]))
        mask = cv2.imread(str(ROOT / row["mask"]), 0) > 0
        image[mask] = np.uint8(.55*image[mask]+.45*np.array([0,220,0]))
        preview = Image.fromarray(cv2.cvtColor(image,cv2.COLOR_BGR2RGB))
        preview.thumbnail((145,145))
        x,y=(i%8)*150,(i//8)*180
        sheet.paste(preview,(x+(150-preview.width)//2,y))
        draw.text((x+2,y+148),f"{i:03} {key(ROOT/row['image'])[-12:]}",fill="black")
    sheet.save(OUT/"reference_qc.jpg",quality=85)


if __name__ == "__main__":
    main()
