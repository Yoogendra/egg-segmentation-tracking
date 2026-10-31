from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "egg.v1i.folder"
OUT = ROOT / "outputs"


def images() -> list[Path]:
    return sorted(DATA.rglob("*.jpg"))


def key(path: Path) -> str:
    return path.stem.split(".rf.")[0]


def read(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Cannot read {path}")
    return image


def save_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def phash(image: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(cv2.resize(image, (32, 32)), cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))[:8, :8]
    return (dct > np.median(dct[1:])).flatten()


def make_split(paths: list[Path], seed: int = 31) -> dict[str, object]:
    """Keep near duplicates and geometrically matching scenes together."""
    hashes = [phash(read(p)) for p in paths]
    orb = cv2.ORB_create(nfeatures=500)
    features = [orb.detectAndCompute(cv2.cvtColor(cv2.resize(read(p),(320,320)),cv2.COLOR_BGR2GRAY),None) for p in paths]
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    parents = list(range(len(paths)))

    def find(x: int) -> int:
        while parents[x] != x:
            parents[x] = parents[parents[x]]
            x = parents[x]
        return x

    for i in range(len(paths)):
        for j in range(i):
            same = np.count_nonzero(hashes[i] != hashes[j]) <= 6
            if not same:
                kp_a, desc_a = features[i]
                kp_b, desc_b = features[j]
                if desc_a is not None and desc_b is not None:
                    good = [a for a,b in matcher.knnMatch(desc_a,desc_b,k=2) if a.distance < .75*b.distance]
                    if len(good) >= 20:
                        src = np.float32([kp_a[m.queryIdx].pt for m in good])
                        dst = np.float32([kp_b[m.trainIdx].pt for m in good])
                        _, inliers = cv2.findHomography(src,dst,cv2.RANSAC,5)
                        same = inliers is not None and int(inliers.sum()) >= 20
            if same:
                parents[find(i)] = find(j)
    groups: dict[int, list[int]] = {}
    for i in range(len(paths)):
        groups.setdefault(find(i), []).append(i)
    ranked = sorted(groups.values(), key=lambda g: hashlib.sha256(f"{seed}:{min(key(paths[i]) for i in g)}".encode()).hexdigest())
    target = round(0.2 * len(paths))
    evaluation: list[int] = []
    for group in ranked:
        if len(evaluation) < target and len(evaluation) + len(group) <= target + 2:
            evaluation.extend(group)
    if not evaluation:
        evaluation = ranked[0]
    eval_set = set(evaluation)
    return {
        "seed": seed,
        "criterion": "64-bit DCT pHash Hamming <= 6 OR >=20 ORB+RANSAC inliers; transitive groups",
        "group_count": len(groups),
        "dev": [str(paths[i].relative_to(ROOT)) for i in range(len(paths)) if i not in eval_set],
        "eval": [str(paths[i].relative_to(ROOT)) for i in range(len(paths)) if i in eval_set],
        "groups": [[str(paths[i].relative_to(ROOT)) for i in group] for group in ranked],
    }


def components(mask: np.ndarray, min_area: int = 300) -> list[np.ndarray]:
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    return [(labels == i).astype(np.uint8) for i in range(1, n) if stats[i, cv2.CC_STAT_AREA] >= min_area]


def shape_features(mask: np.ndarray) -> dict[str, float]:
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return {"area": 0.0, "circularity": 0.0, "solidity": 0.0, "aspect": 0.0, "border": 1.0}
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    perimeter = cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    x, y, w, h = cv2.boundingRect(c)
    ih, iw = mask.shape
    return {
        "area": area,
        "circularity": float(4 * np.pi * area / perimeter**2) if perimeter else 0.0,
        "solidity": float(area / cv2.contourArea(hull)) if cv2.contourArea(hull) else 0.0,
        "aspect": float(w / h) if h else 0.0,
        "border": float(x <= 2 or y <= 2 or x + w >= iw - 2 or y + h >= ih - 2),
    }


def clean(mask: np.ndarray, k: int = 5) -> np.ndarray:
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)


def classical(image: np.ndarray, method: str, light: int = 160, sat: int = 90, min_area: float = 0.003) -> np.ndarray:
    h, w = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if method == "global":
        binary = (gray >= light).astype(np.uint8)
    elif method == "otsu":
        _, binary = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif method in ("refined", "grabcut"):
        local = cv2.GaussianBlur(gray, (0, 0), 23)
        binary = ((gray >= light) & (hsv[:, :, 1] <= sat) & (gray >= local - 8)).astype(np.uint8)
    else:
        raise ValueError(method)
    binary = clean(binary, 7 if method in ("refined", "grabcut") else 5)
    if method == "grabcut":
        result = np.zeros((h,w),np.uint8)
        seeds = sorted(components(binary, max(300, int(min_area*h*w))), key=lambda a: int(a.sum()), reverse=True)[:8]
        for seed in seeds:
            f = shape_features(seed)
            if f["area"] > .20*h*w or f["solidity"] < .65 or f["circularity"] < .25:
                continue
            contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            x,y,bw,bh = cv2.boundingRect(max(contours,key=cv2.contourArea))
            pad = int(.28*max(bw,bh))+12
            x0,y0,x1,y1=max(x-pad,0),max(y-pad,0),min(x+bw+pad,w),min(y+bh+pad,h)
            region = image[y0:y1,x0:x1]
            core = seed[y0:y1,x0:x1]
            if min(region.shape[:2]) < 10:
                continue
            labels = np.full(core.shape, cv2.GC_BGD, np.uint8)
            labels[5:-5,5:-5] = cv2.GC_PR_BGD
            dilation = cv2.dilate(core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(41,41)))
            labels[dilation>0] = cv2.GC_PR_FGD
            eroded = cv2.erode(core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))
            labels[eroded>0] = cv2.GC_FGD
            if not np.any(labels==cv2.GC_FGD) or not np.any(labels==cv2.GC_BGD):
                continue
            try:
                cv2.grabCut(region,labels,None,np.zeros((1,65),np.float64),np.zeros((1,65),np.float64),2,cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                continue
            proposal = np.isin(labels,[cv2.GC_FGD,cv2.GC_PR_FGD]).astype(np.uint8)
            proposal = clean(proposal,5)
            if shape_features(proposal)["area"] < .45*h*w:
                result[y0:y1,x0:x1] |= proposal
        return result
    selected = np.zeros((h, w), np.uint8)
    for part in components(binary, max(100, int(min_area * h * w))):
        f = shape_features(part)
        if method == "refined" and (f["solidity"] < 0.78 or f["circularity"] < 0.35 or not 0.35 < f["aspect"] < 2.8):
            continue
        if f["area"] > 0.45 * h * w:
            continue
        selected |= part
    return selected


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    t, p = truth.astype(bool), pred.astype(bool)
    tp = np.count_nonzero(t & p)
    fp = np.count_nonzero(~t & p)
    fn = np.count_nonzero(t & ~p)
    return {
        "iou": float(tp / (tp + fp + fn)) if tp + fp + fn else 1.0,
        "dice": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else 1.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
    }


def overlay(image: np.ndarray, truth: np.ndarray, pred: np.ndarray) -> np.ndarray:
    result = image.copy()
    result[truth.astype(bool) & ~pred.astype(bool)] = (0, 0, 255)
    result[pred.astype(bool) & ~truth.astype(bool)] = (255, 0, 0)
    result[truth.astype(bool) & pred.astype(bool)] = (0, 190, 0)
    return cv2.addWeighted(image, 0.45, result, 0.55, 0)
