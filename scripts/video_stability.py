"""Measure how steady the video boxes are: raw per-run detections vs the stabilised tracker.

RDD2022 has photos, not videos, so this builds simulated drives with exact ground truth: each
clip slowly zooms into one held-out photo towards the horizon (the road "approaches", damage
grows), with camera shake and road vibration. The ground-truth boxes are moved with the same
transform, so every frame has labels. Clips from validation images form the *dev* video (used to
pick settings); clips from test images form the *test* video (reported).

    python scripts/video_stability.py            # build videos, detect once (cached), compare
    python scripts/video_stability.py --tune     # also sweep tracker settings on the dev video

Writes data/stability/ (videos, cached detections; gitignored) and results/video_stability.json.

Metrics (per minute of video unless noted; lower is better except F1):
- blinks: a damage's box disappears for >= 1 frame and comes back (flicker)
- label switches: the class shown on the same damage changes between consecutive frames
- ID switches: the track number shown on the same damage changes
- jitter: mean frame-to-frame change of the drawn box relative to the true box, in % of box size
- false boxes / frame: drawn boxes that overlap no true damage (IoU < 0.3)
- F1: per-frame precision/recall at IoU >= 0.5 with the right class (the RDD challenge rule)
"""
from __future__ import annotations

import argparse
import json
import math
import pickle
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.detect import box_iou  # noqa: E402
from src.utils import get_paths, load_config  # noqa: E402
from src.video import track_video  # noqa: E402

OUT = ROOT / "data" / "stability"
COUNTRIES = ("India", "Japan", "Czech", "United_States", "China_MotorBike")  # forward-facing cameras
S, FPS, SECS, CLIPS = 720, 25, 3.6, 16


# ------------------------------------------------------------------ simulated drives
def build_video(split: str, seed: int) -> Path:
    import av
    from PIL import Image

    from src.detect import select_records
    vid, gt_file = OUT / f"{split}_drive.mp4", OUT / f"{split}_drive_gt.json"
    if vid.exists() and gt_file.exists():
        return vid
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_config()
    paths = get_paths(cfg, False)
    recs = [r for r in select_records(cfg, paths)[split] if r["country"] in COUNTRIES
            and (paths.raw / r["file"]).exists()
            and any(min(b["box"][2] - b["box"][0], b["box"][3] - b["box"][1]) >= 30 for b in r["boxes"])]
    rng = random.Random(seed)
    pot = [r for r in recs if any(b["cls"] == "D40" for b in r["boxes"])]
    other = [r for r in recs if r not in pot]
    pick = rng.sample(pot, CLIPS // 2) + rng.sample(other, CLIPS - CLIPS // 2)
    rng.shuffle(pick)
    n = int(FPS * SECS)
    gt_frames, obj = [], 0
    with av.open(str(vid), "w") as out:
        st = out.add_stream("libx264", rate=FPS)
        st.width = st.height = S
        st.pix_fmt = "yuv420p"
        st.options = {"crf": "23"}
        for ci, r in enumerate(pick):
            img = Image.open(paths.raw / r["file"]).convert("RGB")
            W, H = img.size
            ax, ay = W * 0.5, H * 0.42                         # roughly the horizon / vanishing point
            ph = [rng.uniform(0, 2 * math.pi) for _ in range(4)]
            gts = [(obj + j, b["label"], np.array(b["box"], float)) for j, b in enumerate(r["boxes"])]
            obj += len(r["boxes"])
            for k in range(n):
                t = k / FPS
                z = 1 + 1.0 * (k / (n - 1)) ** 1.3             # driving forward: zoom 1x -> 2x
                w, h = W / z, H / z
                shake = 0.012 * W                              # slow sway + faster road vibration
                sx = shake * (0.6 * math.sin(2 * math.pi * 1.3 * t + ph[0]) + 0.4 * math.sin(2 * math.pi * 3.7 * t + ph[1]))
                sy = shake * (0.6 * math.sin(2 * math.pi * 1.1 * t + ph[2]) + 0.4 * math.sin(2 * math.pi * 4.3 * t + ph[3]))
                x0 = min(max(0.0, ax * (1 - 1 / z) + sx / z), W - w)
                y0 = min(max(0.0, ay * (1 - 1 / z) + sy / z), H - h)
                fr = img.transform((S, S), Image.EXTENT, (x0, y0, x0 + w, y0 + h), Image.BILINEAR)
                for p in st.encode(av.VideoFrame.from_image(fr)):
                    out.mux(p)
                g = []
                for oid, lab, b in gts:
                    tb = (b - [x0, y0, x0, y0]) * [S / w, S / h, S / w, S / h]
                    cb = np.clip(tb, 0, S)
                    area = (tb[2] - tb[0]) * (tb[3] - tb[1])
                    vis = (cb[2] - cb[0]) * (cb[3] - cb[1]) / max(area, 1e-9)
                    if vis >= 0.6 and min(cb[2] - cb[0], cb[3] - cb[1]) >= 10:
                        g.append({"obj": oid, "label": lab, "box": cb.round(2).tolist()})
                gt_frames.append(g)
        for p in st.encode():
            out.mux(p)
    json.dump({"clips": [r["id"] for r in pick], "fps": FPS, "frames": gt_frames}, open(gt_file, "w"))
    return vid


# ------------------------------------------------------------------ cached detector
def cached_detections(vid: Path, every: int) -> list:
    """Run the app's detector ("Best" preset: flip + 576/704 px, scores >= 0.05) once per video and cache the output,
    so tracker settings can be compared without re-running the network."""
    f = OUT / f"{vid.stem}_dets_best_e{every}.pkl"
    if f.exists():
        return pickle.load(open(f, "rb"))
    import torch

    from src.detect import detect, load_detector
    dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    model, _ = load_detector(ROOT / "checkpoints" / "detector.pt", dev, score_thresh=0.05)
    res = []

    @torch.no_grad()
    def fn(imgs):
        xs = [torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float().div(255).to(dev)
              for im in imgs]
        out = []
        for o in detect(model, xs, tta=True, scales=(576, 704)):
            out.append((o["boxes"].cpu().numpy(), o["labels"].cpu().numpy() - 1, o["scores"].cpu().numpy()))
        res.extend(out)
        return out

    track_video(vid.read_bytes(), fn, ["D00", "D10", "D20", "D40"], thresh=0.4, detect_every=every,
                stabilize=False)
    pickle.dump(res, open(f, "wb"))
    return res


def replay(dets: list):
    it = iter(dets)
    return lambda imgs: [next(it) for _ in imgs]


# ------------------------------------------------------------------ metrics
def greedy_match(D: np.ndarray, G: np.ndarray, thr: float, ok=None) -> list[tuple[int, int]]:
    iou = box_iou(D, G)
    if ok is not None:
        iou = np.where(ok, iou, 0)
    pairs, ud, ug = [], set(), set()
    for flat in np.argsort(-iou, axis=None, kind="stable"):
        i, j = np.unravel_index(flat, iou.shape)
        if iou[i, j] < thr:
            break
        if i not in ud and j not in ug:
            pairs.append((int(i), int(j)))
            ud.add(i)
            ug.add(j)
    return pairs


def stability_metrics(per_frame: list[list[dict]], gt_frames: list[list[dict]], fps: float) -> dict:
    minutes = len(per_frame) / fps / 60
    hist: dict[int, list] = {}           # gt object -> [(frame, det id, label, residual) or (frame, None)]
    tp5 = fp5 = n_gt = false_boxes = n_boxes = 0
    for n, (dets, gts) in enumerate(zip(per_frame, gt_frames)):
        D = np.array([d["box"] for d in dets], float).reshape(-1, 4)
        G = np.array([g["box"] for g in gts], float).reshape(-1, 4)
        n_gt, n_boxes = n_gt + len(G), n_boxes + len(D)
        m = greedy_match(D, G, 0.3)
        false_boxes += len(D) - len(m)
        matched = {j: i for i, j in m}
        for j, g in enumerate(gts):
            if j in matched:
                d = dets[matched[j]]
                gb = G[j]
                gw, gh = max(gb[2] - gb[0], 1), max(gb[3] - gb[1], 1)
                res = (np.array(d["box"]) - gb) / [gw, gh, gw, gh]
                hist.setdefault(g["obj"], []).append((n, d.get("id"), d["label"], res))
            else:
                hist.setdefault(g["obj"], []).append((n, None, None, None))
        ok = np.array([[d["label"] == g["label"] for g in gts] for d in dets], bool).reshape(len(D), len(G))
        tp = len(greedy_match(D, G, 0.5, ok))
        tp5, fp5 = tp5 + tp, fp5 + len(D) - tp
    blinks = label_sw = id_sw = 0
    jit = []
    for seq in hist.values():
        seen = [k for k, s in enumerate(seq) if s[3] is not None]
        if not seen:
            continue
        for a, b in zip(seen, seen[1:]):
            if b > a + 1:
                blinks += 1
            pa, pb = seq[a], seq[b]
            label_sw += pa[2] != pb[2]
            id_sw += pa[1] != pb[1]
            if b == a + 1 and pa[1] == pb[1] and pb[0] == pa[0] + 1:
                jit.append(np.abs(pb[3] - pa[3]).mean())
    prec, rec = tp5 / max(1, tp5 + fp5), tp5 / max(1, n_gt)
    return {"blinks_per_min": round(blinks / minutes, 1), "label_switches_per_min": round(label_sw / minutes, 1),
            "id_switches_per_min": round(id_sw / minutes, 1),
            "jitter_pct": round(100 * float(np.mean(jit)) if jit else float("nan"), 2),
            "false_boxes_per_frame": round(false_boxes / len(per_frame), 2),
            "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4),
            "boxes_per_frame": round(n_boxes / len(per_frame), 2)}


def run(split: str, every: int, stabilize: bool, thresh: float = 0.5, **kw) -> tuple[dict, dict]:
    vid = OUT / f"{split}_drive.mp4"
    gt = json.load(open(OUT / f"{split}_drive_gt.json"))
    r = track_video(vid.read_bytes(), replay(cached_detections(vid, every)), ["D00", "D10", "D20", "D40"],
                    thresh=thresh, detect_every=every, stabilize=stabilize, **kw)
    m = stability_metrics(r["per_frame"], gt["frames"], r["fps"])
    m["damages_counted"] = len(r["damages"])
    m["true_damages"] = len({g["obj"] for f in gt["frames"] for g in f})
    return m, r


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tune", action="store_true", help="sweep tracker settings on the dev video")
    ap.add_argument("--every", type=int, nargs="+", default=[2])
    args = ap.parse_args()
    build_video("val", seed=1)
    build_video("test", seed=2)
    out = {"setup": f"{CLIPS} clips x {SECS}s at {FPS} fps, {S}x{S}; zoom 1x->2x towards the horizon with "
                    f"camera shake; detector 'Best' preset (flip + 576/704 px); confidence threshold 0.50", "runs": []}
    if args.tune:
        for mh in (2, 3, 4):
            for mi in (0.5, 0.6, 0.7):
                for bw in (0.10, 0.15, 0.25):
                    for gap in (0.4, 0.6, 1.0):
                        m, _ = run("val", 2, True, min_hits=mh, merge_iou=mi, bandwidth=bw, max_gap=gap)
                        print(f"dev min_hits={mh} merge={mi} bw={bw} gap={gap}: {m}", flush=True)
                        out["runs"].append({"video": "dev", "min_hits": mh, "merge_iou": mi, "bandwidth": bw,
                                            "max_gap": gap, **m})
    for split, name in (("val", "dev"), ("test", "test")):
        for every in args.every:
            for stab in (False, True):
                m, _ = run(split, every, stab)
                tag = "stabilised" if stab else "raw (before)"
                print(f"{name:4s} every={every} {tag:13s} {m}", flush=True)
                out["runs"].append({"video": name, "detect_every": every, "mode": tag, **m})
    if not args.tune:
        (ROOT / "results").mkdir(exist_ok=True)
        json.dump(out, open(ROOT / "results" / "video_stability.json", "w"), indent=1)


if __name__ == "__main__":
    main()
