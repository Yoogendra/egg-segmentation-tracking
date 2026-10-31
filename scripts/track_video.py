"""Classical segmentation plus nearest-neighbor multi-object tracking."""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eggcv.core import OUT, classical, components, shape_features


def detections(frame: np.ndarray) -> list[dict]:
    h, w = frame.shape[:2]
    mask = classical(frame, "refined", 190, 110, .001)
    mask[:int(.32*h)] = 0
    mask[int(.87*h):] = 0
    result = []
    for part in components(mask, 700):
        f = shape_features(part)
        if f["area"] > 12000 or f["solidity"] < .80 or f["circularity"] < .48 or not .40 < f["aspect"] < 2.5:
            continue
        contours, _ = cv2.findContours(part, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour = max(contours, key=cv2.contourArea)
        moments = cv2.moments(contour)
        if moments["m00"] == 0:
            continue
        result.append({"x": moments["m10"]/moments["m00"], "y": moments["m01"]/moments["m00"], "area": f["area"], "contour": contour})
    return sorted(result, key=lambda r: r["x"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/video/source.mp4")
    parser.add_argument("--output", default="outputs/tracked_eggs.mp4")
    parser.add_argument("--max-gap", type=int, default=24)
    args = parser.parse_args()
    source = cv2.VideoCapture(args.input)
    if not source.isOpened():
        raise SystemExit(f"Cannot open {args.input}")
    fps = source.get(cv2.CAP_PROP_FPS)
    width, height = int(source.get(3)), int(source.get(4))
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        raise SystemExit("Cannot create video writer")
    tracks: dict[int, dict] = {}
    next_id = 1
    rows = []
    frame_index = 0
    while True:
        ok, frame = source.read()
        if not ok:
            break
        seen = detections(frame)
        active_ids = list(tracks)
        if active_ids and seen:
            cost = np.array([[np.hypot(tracks[i]["x"]-d["x"], tracks[i]["y"]-d["y"]) + min(frame_index-tracks[i]["last"], 120)*1.5 for d in seen] for i in active_ids])
            rr, cc = linear_sum_assignment(cost)
            for r, c in zip(rr, cc):
                tid = active_ids[r]
                gap = frame_index-tracks[tid]["last"]
                distance = np.hypot(tracks[tid]["x"]-seen[c]["x"], tracks[tid]["y"]-seen[c]["y"])
                tracks[tid].update(x=seen[c]["x"], y=seen[c]["y"], last=frame_index)
                seen[c]["id"] = tid
                seen[c]["uncertain"] = gap > args.max_gap or distance > 100
        for d in seen:
            if "id" not in d:
                if next_id > 2:
                    continue  # Keep exactly two physical identities; extra proposals are clutter.
                d["id"] = next_id
                d["uncertain"] = False
                tracks[next_id] = {"x": d["x"], "y": d["y"], "last": frame_index}
                next_id += 1
            if "id" not in d:
                continue
            tid = d["id"]
            color = [(66, 219, 245), (245, 182, 76), (130, 227, 121)][(tid-1)%3]
            cv2.drawContours(frame, [d["contour"]], -1, color, 2)
            x, y = int(d["x"]), int(d["y"])
            cv2.circle(frame, (x,y), 4, color, -1)
            cv2.rectangle(frame, (x-27,y-53),(x+42,y-30),(10,19,30),-1)
            cv2.putText(frame, f"EGG {tid}{' ?' if d.get('uncertain') else ''}", (x-23,y-36), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
            rows.append({"frame": frame_index, "seconds": round(frame_index/fps,3), "id": tid, "x": round(d["x"],2), "y": round(d["y"],2), "area": round(d["area"],1), "uncertain_reid": bool(d.get("uncertain"))})
        cv2.rectangle(frame,(8,8),(258,48),(10,19,30),-1)
        cv2.putText(frame,"EGG TRACKING  |  CLASSICAL CV",(16,32),cv2.FONT_HERSHEY_SIMPLEX,.49,(235,245,250),1,cv2.LINE_AA)
        writer.write(frame)
        frame_index += 1
        if frame_index % 240 == 0:
            print(f"Processed {frame_index} frames", flush=True)
    writer.release()
    source.release()
    with (OUT / "tracks.csv").open("w",newline="") as handle:
        writer_csv = csv.DictWriter(handle,["frame","seconds","id","x","y","area","uncertain_reid"])
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    print(f"Wrote {destination}: {frame_index} frames, {len(rows)} detections, {next_id-1} IDs")


if __name__ == "__main__":
    main()
