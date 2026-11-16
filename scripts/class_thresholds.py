"""Per-class confidence thresholds for the app, chosen on the validation set.

The detector is less confident about potholes than cracks, so one global threshold (0.40/0.50)
hides many correct pothole boxes. This script picks, for each preset and each class, the score
threshold with the best validation F1 for that class, then reports test precision / recall / F1
per class with the global threshold vs the per-class thresholds (no tuning on test).

    python scripts/class_thresholds.py      # writes results/class_thresholds.json

Predictions are cached in data/stability/preds_<split>_<preset>.pkl (gitignored).
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detect import (DetDataset, box_iou, load_detector, predict_dataset, select_records)  # noqa: E402
from src.utils import get_device, get_paths, load_config  # noqa: E402

PRESETS = {"quick": (False, None), "balanced": (True, None), "best": (True, (576, 704))}
GLOBAL = {"quick": 0.40, "balanced": 0.40, "best": 0.50}  # the app's previous single thresholds
CACHE = ROOT / "data" / "stability"


def preds(split: str, preset: str):
    f = CACHE / f"preds_{split}_{preset}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    cfg = load_config()
    paths = get_paths(cfg, False)
    dev = get_device(cfg["device"])
    model, ck = load_detector(paths.checkpoints / "detector.pt", dev, score_thresh=0.05)
    recs = [r for r in select_records(cfg, paths)[split] if (paths.raw / r["file"]).exists()]
    tta, scales = PRESETS[preset]
    out = predict_dataset(model, DetDataset(recs, paths.raw, ck["max_size"], train=False), dev, tta=tta,
                          scales=scales)
    CACHE.mkdir(parents=True, exist_ok=True)
    pickle.dump(out, open(f, "wb"))
    return out


def per_class_prf(gts, ps, thr: np.ndarray, n_cls: int = 4, iou_thr: float = 0.5) -> list[dict]:
    """Precision / recall / F1 per class; detections of class c kept if score >= thr[c]."""
    tp, fp, ng = np.zeros(n_cls), np.zeros(n_cls), np.zeros(n_cls)
    for g, p in zip(gts, ps):
        for c in range(n_cls):
            gb = g["boxes"][g["labels"] == c + 1]
            k = (p["labels"] == c + 1) & (p["scores"] >= thr[c])
            b = p["boxes"][k][np.argsort(-p["scores"][k])]
            ng[c] += len(gb)
            used = np.zeros(len(gb), bool)
            iou = box_iou(b, gb)
            for i in range(len(b)):
                cand = np.flatnonzero(~used)
                j = cand[iou[i, cand].argmax()] if len(cand) else -1
                if j >= 0 and iou[i, j] >= iou_thr:
                    used[j] = True
                    tp[c] += 1
                else:
                    fp[c] += 1
    out = []
    for c in range(n_cls):
        pr, rc = tp[c] / max(1, tp[c] + fp[c]), tp[c] / max(1, ng[c])
        out.append({"precision": round(pr, 3), "recall": round(rc, 3),
                    "f1": round(2 * pr * rc / max(1e-9, pr + rc), 3)})
    return out


def main():
    classes = ["D00", "D10", "D20", "D40"]
    grid = np.round(np.arange(0.10, 0.91, 0.05), 2)
    res = {}
    for preset in PRESETS:
        gv, pv = preds("val", preset)
        best = []
        for c in range(4):
            scores = [per_class_prf(gv, pv, np.full(4, t))[c]["f1"] for t in grid]
            best.append(float(grid[int(np.argmax(scores))]))
        gt_, pt_ = preds("test", preset)
        g = GLOBAL[preset]
        before = per_class_prf(gt_, pt_, np.full(4, g))
        after = per_class_prf(gt_, pt_, np.array(best))
        res[preset] = {"global_threshold": g, "per_class_threshold": dict(zip(classes, best)),
                       "test_global": dict(zip(classes, before)), "test_per_class": dict(zip(classes, after))}
        print(preset, "thresholds", dict(zip(classes, best)))
        for c in range(4):
            print(f"  {classes[c]} test  global {before[c]}  ->  per-class {after[c]}")
    json.dump(res, open(ROOT / "results" / "class_thresholds.json", "w"), indent=1)


if __name__ == "__main__":
    main()
