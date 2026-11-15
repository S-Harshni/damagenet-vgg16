"""Road-damage *detector*: finds and boxes damage by itself (used by the app's video mode).

The VGG16 model in this project is a patch classifier: it needs a box and has no "no damage"
class. To find damage automatically we fine-tune an object detector, Faster R-CNN with a
MobileNetV3-Large FPN backbone (torchvision, COCO-pretrained), on the full RDD2022 boxes of
the same seeded image-level split (data/splits.json), so test images stay unseen.

    python -m src.detect                  # download extra images, train, evaluate
    python -m src.detect --only download|train|eval

Outputs: checkpoints/detector.pt, results/detector.json (VOC AP@0.5 per class on test images).
Augmentation is horizontal flip only, for the same reason as the classifier (a vertical flip or
rotation would turn a longitudinal crack into a transverse one).
"""
from __future__ import annotations

import argparse
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .utils import (Paths, get_device, get_logger, get_paths, load_config, load_json, save_json, set_seed,
                    setup_logging, use_project_torch_hub)


# ------------------------------------------------------------------------- model
def build_detector(num_classes: int, pretrained: bool = True, min_size: int = 640, max_size: int = 1066,
                   **kw):
    """Faster R-CNN MobileNetV3-Large FPN with a new box predictor (num_classes + background)."""
    from torchvision.models.detection import (FasterRCNN_MobileNet_V3_Large_FPN_Weights,
                                              fasterrcnn_mobilenet_v3_large_fpn)
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    use_project_torch_hub()  # COCO weights go to data/torch_hub, not ~/.cache
    weights = FasterRCNN_MobileNet_V3_Large_FPN_Weights.COCO_V1 if pretrained else None
    # min_size may be a list: training picks one at random per batch (multi-scale), eval uses the last
    min_size = tuple(min_size) if isinstance(min_size, (list, tuple)) else min_size
    m = fasterrcnn_mobilenet_v3_large_fpn(weights=weights, weights_backbone=None, min_size=min_size,
                                          max_size=max_size, **kw)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features,
                                                  num_classes + 1)
    return m


def load_detector(path: Path, device="cpu", score_thresh: float = 0.05):
    ck = torch.load(path, map_location="cpu")
    m = build_detector(len(ck["classes"]), pretrained=False, min_size=ck["min_size"],
                       max_size=ck["max_size"], box_score_thresh=score_thresh)
    m.load_state_dict(ck["state_dict"])
    return m.eval().to(device), ck


# -------------------------------------------------------------------------- data
def load_image(path: Path, max_side: int) -> tuple[Image.Image, float]:
    """Open an image, downscaled so its longer side is <= max_side. Returns (image, scale)."""
    img = Image.open(path)
    W, H = img.size
    s = min(1.0, max_side / max(W, H))
    if s < 1:
        img.draft("RGB", (int(W * s), int(H * s)))  # fast JPEG decode at reduced size
        img = img.convert("RGB").resize((max(1, round(W * s)), max(1, round(H * s))), Image.BILINEAR)
    else:
        img = img.convert("RGB")
    return img, s


class DetDataset(torch.utils.data.Dataset):
    def __init__(self, records: list[dict], raw: Path, max_side: int, train: bool, min_side: float = 6):
        self.r, self.raw, self.max_side, self.train, self.min_side = records, raw, max_side, train, min_side

    def __len__(self):
        return len(self.r)

    def __getitem__(self, i):
        rec = self.r[i]
        img, s = load_image(self.raw / rec["file"], self.max_side)
        boxes = [[v * s for v in b["box"]] for b in rec["boxes"]]
        labels = [b["label"] + 1 for b in rec["boxes"]]  # 0 = background
        keep = [k for k, b in enumerate(boxes) if min(b[2] - b[0], b[3] - b[1]) >= self.min_side]
        boxes = torch.tensor([boxes[k] for k in keep], dtype=torch.float32).reshape(-1, 4)
        labels = torch.tensor([labels[k] for k in keep], dtype=torch.int64)
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)
        if self.train and random.random() < 0.5:  # horizontal flip only
            x = x.flip(-1)
            W = x.shape[-1]
            boxes = boxes[:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1]) + torch.tensor([W, 0, W, 0])
        return x, {"boxes": boxes, "labels": labels}


def collate(batch):
    return tuple(zip(*batch))


def select_records(cfg: dict, paths: Paths, debug: bool = False) -> dict[str, list[dict]]:
    """Detector images per split (deterministic, independent of what is on disk):
    - every image used by the patch dataset (data/patches/meta.json),
    - every image with a pothole (D40), so the rarest class gets as many examples as possible,
    - train only: a seeded random sample of `extra_train_images` further training images.
    Val and test therefore stay identical when `extra_train_images` changes."""
    dc = cfg["detector"]
    records = load_json(paths.data / "ann.json")
    split_ids = load_json(paths.data / "splits.json")
    patch_ids = {m["id"] for v in load_json(paths.patches / "meta.json").values() for m in v}
    by_id = {r["id"]: r for r in records}
    out = {}
    for s, ids in split_ids.items():
        recs = [by_id[i] for i in ids]
        pick = [r for r in recs if r["id"] in patch_ids
                or (dc["all_pothole_images"] and any(b["cls"] == "D40" for b in r["boxes"]))]
        if s == "train" and dc.get("extra_train_images"):
            chosen = {r["id"] for r in pick}
            rest = [r for r in recs if r["id"] not in chosen]
            pick += random.Random(cfg["seed"]).sample(rest, min(len(rest), dc["extra_train_images"]))
        if debug:  # a few already-downloaded images, spread over countries
            pick = [r for r in pick if (paths.raw / r["file"]).exists()]
            pick = random.Random(cfg["seed"]).sample(pick, min(len(pick), cfg["debug"]["det_images"]))
        out[s] = sorted(pick, key=lambda r: r["id"])
    return out


def download_images(cfg: dict, paths: Paths, recs: dict[str, list[dict]]) -> None:
    from .data import download_files
    files = sorted({r["file"] for v in recs.values() for r in v})
    failed = download_files(cfg["dataset"]["repo_id"], files, paths.raw,
                            workers=cfg["dataset"]["download_workers"], desc="detector images")
    if failed:
        get_logger().warning(f"{len(failed)} detector images failed to download; they are skipped")


# ---------------------------------------------------------------------- evaluate
def voc_ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """Area under the precision-recall curve (all-point interpolation, VOC 2010+)."""
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[0.0], prec, [0.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    idx = np.flatnonzero(mrec[1:] != mrec[:-1])
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x0 = np.maximum(a[:, None, 0], b[None, :, 0]); y0 = np.maximum(a[:, None, 1], b[None, :, 1])
    x1 = np.minimum(a[:, None, 2], b[None, :, 2]); y1 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])  # noqa: E731
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def average_precision(gts: list[dict], preds: list[dict], num_classes: int, iou_thr: float = 0.5) -> list[float]:
    """Per-class VOC AP. gts/preds: one dict per image with numpy boxes/labels(/scores)."""
    aps = []
    for c in range(1, num_classes + 1):
        n_gt = sum(int((g["labels"] == c).sum()) for g in gts)
        dets = [(p["scores"][k], i, p["boxes"][k]) for i, p in enumerate(preds)
                for k in np.flatnonzero(p["labels"] == c)]
        dets.sort(key=lambda t: -t[0])
        used = {i: np.zeros(int((g["labels"] == c).sum()), bool) for i, g in enumerate(gts)}
        tp = np.zeros(len(dets)); fp = np.zeros(len(dets))
        for k, (_, i, b) in enumerate(dets):
            g = gts[i]["boxes"][gts[i]["labels"] == c]
            iou = box_iou(b[None], g)[0] if len(g) else np.array([])
            j = int(iou.argmax()) if len(iou) else -1
            if j >= 0 and iou[j] >= iou_thr and not used[i][j]:
                tp[k], used[i][j] = 1, True
            else:
                fp[k] = 1
        if n_gt == 0:
            aps.append(float("nan"))
            continue
        tp, fp = np.cumsum(tp), np.cumsum(fp)
        aps.append(voc_ap(tp / n_gt, tp / np.maximum(tp + fp, 1e-9)))
    return aps


@torch.no_grad()
def detect(model, xs: list[torch.Tensor], tta: bool = False, iou: float = 0.5, max_det: int = 100) -> list[dict]:
    """Run the detector on float images in [0, 1]. With `tta`, also run on the horizontally
    flipped images, flip those boxes back and merge both sets with per-class NMS."""
    from torchvision.ops import batched_nms
    out = model(xs)
    if not tta:
        return out
    out_f = model([x.flip(-1) for x in xs])
    merged = []
    for x, a, b in zip(xs, out, out_f):
        W = x.shape[-1]
        bf = b["boxes"][:, [2, 1, 0, 3]] * torch.tensor([-1, 1, -1, 1], device=x.device) + \
            torch.tensor([W, 0, W, 0], device=x.device)
        boxes, scores = torch.cat([a["boxes"], bf]), torch.cat([a["scores"], b["scores"]])
        labels = torch.cat([a["labels"], b["labels"]])
        keep = batched_nms(boxes, scores, labels, iou)[:max_det]
        merged.append({"boxes": boxes[keep], "scores": scores[keep], "labels": labels[keep]})
    return merged


@torch.no_grad()
def predict_dataset(model, ds: DetDataset, device, batch_size: int = 8, tta: bool = False) -> tuple[list[dict], list[dict]]:
    model.eval()
    dl = torch.utils.data.DataLoader(ds, batch_size=batch_size, collate_fn=collate, num_workers=4,
                                     persistent_workers=False)
    gts, preds = [], []
    for xs, ts in dl:
        out = detect(model, [x.to(device).float().div(255) for x in xs], tta)
        for t, o in zip(ts, out):
            gts.append({k: v.numpy() for k, v in t.items()})
            preds.append({k: v.cpu().numpy() for k, v in o.items()})
    return gts, preds


def f1_at(gts: list[dict], preds: list[dict], num_classes: int, score: float, iou_thr: float = 0.5) -> dict:
    """Precision / recall / F1 over all classes for detections with score >= `score`
    (a detection is correct if it matches an unused ground-truth box of its class at IoU >= 0.5)."""
    tp = fp = n_gt = 0
    for g, p in zip(gts, preds):
        n_gt += len(g["labels"])
        keep = p["scores"] >= score
        order = np.argsort(-p["scores"][keep])
        boxes, labels = p["boxes"][keep][order], p["labels"][keep][order]
        used = np.zeros(len(g["labels"]), bool)
        iou = box_iou(boxes, g["boxes"])
        for k in range(len(boxes)):
            cand = np.flatnonzero((g["labels"] == labels[k]) & ~used)
            j = cand[iou[k, cand].argmax()] if len(cand) else -1
            if j >= 0 and iou[k, j] >= iou_thr:
                used[j], tp = True, tp + 1
            else:
                fp += 1
    prec, rec = tp / max(1, tp + fp), tp / max(1, n_gt)
    return {"score": float(score), "precision": round(prec, 4), "recall": round(rec, 4),
            "f1": round(2 * prec * rec / max(1e-9, prec + rec), 4)}


def evaluate(model, ds, device, classes, with_f1: bool = False, tta: bool = False) -> dict:
    gts, preds = predict_dataset(model, ds, device, tta=tta)
    aps = average_precision(gts, preds, len(classes))
    extra = {}
    if with_f1:
        f1s = [f1_at(gts, preds, len(classes), t) for t in np.round(np.arange(0.2, 0.91, 0.05), 2)]
        extra = {"f1_at_0.5": next(f for f in f1s if f["score"] == 0.5), "best_f1": max(f1s, key=lambda f: f["f1"])}
    return {**extra, "mAP50": round(float(np.nanmean(aps)), 4),
            "AP50": {c: round(a, 4) for c, a in zip(classes, aps)},
            "n_images": len(gts), "n_boxes": {c: int(sum((g["labels"] == k + 1).sum() for g in gts))
                                              for k, c in enumerate(classes)}}


# ------------------------------------------------------------------------- train
def train(cfg: dict, paths: Paths, recs: dict[str, list[dict]], debug: bool = False,
          name: str = "detector") -> dict:
    log = get_logger()
    dc = cfg["detector"]
    classes = cfg["dataset"]["classes"]
    set_seed(cfg["seed"])
    device = get_device(cfg["device"])
    have = {s: [r for r in v if (paths.raw / r["file"]).exists()] for s, v in recs.items()}
    log.info("detector images: " + ", ".join(f"{s}={len(v)}" for s, v in have.items()))
    tr = DetDataset(have["train"], paths.raw, dc["max_size"], train=True)
    va = DetDataset(have["val"], paths.raw, dc["max_size"], train=False)
    dl = torch.utils.data.DataLoader(tr, batch_size=dc["batch_size"], shuffle=True, collate_fn=collate,
                                     num_workers=dc["workers"], persistent_workers=dc["workers"] > 0,
                                     generator=torch.Generator().manual_seed(cfg["seed"]))
    model = build_detector(len(classes), True, dc["min_size"], dc["max_size"],
                           trainable_backbone_layers=dc.get("trainable_backbone_layers")).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=dc["lr"], momentum=0.9, weight_decay=dc["weight_decay"])
    epochs = cfg["debug"]["epochs"] if debug else dc["epochs"]
    total, warm = epochs * len(dl), min(500, len(dl))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda it: min(1.0, (it + 1) / warm) * 0.5 * (1 + math.cos(math.pi * it / total)))
    hist, best = [], -1.0
    for ep in range(1, epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for k, (xs, ts) in enumerate(dl, 1):
            xs = [x.to(device).float().div(255) for x in xs]
            ts = [{kk: v.to(device) for kk, v in t.items()} for t in ts]
            loss = sum(model(xs, ts).values())
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 10.0)
            opt.step()
            sched.step()
            run += loss.item()
            if k % 100 == 0:
                log.info(f"  epoch {ep} it {k}/{len(dl)} loss {run / k:.3f} ({time.time() - t0:.0f}s)")
        ev = evaluate(model, va, device, classes)
        hist.append({"epoch": ep, "train_loss": round(run / len(dl), 4), "val_mAP50": ev["mAP50"],
                     "val_AP50": ev["AP50"], "seconds": round(time.time() - t0, 1)})
        log.info(f"[detector] epoch {ep}/{epochs} loss {run / len(dl):.3f} val mAP50 {ev['mAP50']:.4f} "
                 f"{ev['AP50']} ({time.time() - t0:.0f}s)")
        if ev["mAP50"] > best:  # save right away so a crash never loses the best epoch
            best = ev["mAP50"]
            ck = {"state_dict": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                  "classes": classes, "class_names": cfg["dataset"]["class_names"],
                  "min_size": dc["min_size"], "max_size": dc["max_size"],
                  "arch": "fasterrcnn_mobilenet_v3_large_fpn", "best_epoch": ep}
            paths.checkpoints.mkdir(parents=True, exist_ok=True)
            torch.save(ck, paths.checkpoints / f"{name}.pt")
            log.info(f"saved {paths.checkpoints / f'{name}.pt'} (epoch {ep}, val mAP50 {best:.4f})")
    best_ep = max(hist, key=lambda h: h["val_mAP50"])["epoch"]
    info = {"history": hist, "best_epoch": best_ep, "train_config": {k: dc[k] for k in sorted(dc)},
            "n_train_images": len(have["train"]), "n_val_images": len(have["val"])}
    save_json(info, paths.results / f"{name}_train.json")  # kept for a later --only eval
    return info


def run_eval(cfg: dict, paths: Paths, recs: dict[str, list[dict]], train_info: dict | None = None,
             name: str = "detector", split: str = "test", tta: bool = False) -> dict:
    """Evaluate checkpoints/<name>.pt. The test result goes to results/<name>.json; other
    splits / TTA go to results/<name>_<split>[_tta].json (used to pick settings on val)."""
    device = get_device(cfg["device"])
    model, ck = load_detector(paths.checkpoints / f"{name}.pt", device, score_thresh=0.01)
    have = [r for r in recs[split] if (paths.raw / r["file"]).exists()]
    res = evaluate(model, DetDataset(have, paths.raw, ck["max_size"], train=False), device, ck["classes"],
                   with_f1=True, tta=tta)
    res = {"name": name, "arch": ck["arch"], "split": f"{split} (image-level, seed 42)", "tta": tta, **res,
           "min_size": ck["min_size"], "best_epoch": ck.get("best_epoch"),
           **(train_info or {})}
    suffix = "" if split == "test" and not tta else f"_{split}" + ("_tta" if tta else "")
    old = paths.results / f"{name}{suffix}.json"
    tr_json = paths.results / f"{name}_train.json"
    if train_info is None and not suffix and tr_json.exists():  # eval-only: attach the saved training history
        res = {**res, **{k: v for k, v in load_json(tr_json).items() if k != "best_epoch"}}
    save_json(res, old)
    get_logger().info(f"[{name}{suffix}] {split.upper()} mAP50 {res['mAP50']:.4f} AP50 {res['AP50']} "
                      f"F1@0.5 {res['f1_at_0.5']['f1']:.4f} best F1 {res['best_f1']}")
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--only", choices=["download", "train", "eval"])
    ap.add_argument("--name", default="detector", help="checkpoint/results name (checkpoints/<name>.pt)")
    ap.add_argument("--split", default="test", choices=["val", "test"], help="split to evaluate (eval step)")
    ap.add_argument("--tta", action="store_true", help="evaluate with horizontal-flip test-time augmentation")
    args = ap.parse_args(argv)
    cfg = load_config()
    paths = get_paths(cfg, args.debug).make()
    setup_logging(paths.logs, "detect")
    if args.debug:  # debug run reads the full run's split + annotations
        full = get_paths(cfg, False)
        paths.data = full.data
    recs = select_records(cfg, paths, args.debug)
    if args.only in (None, "download"):
        download_images(cfg, paths, recs)
    info = None
    if args.only in (None, "train"):
        info = train(cfg, paths, recs, args.debug, args.name)
    if args.only in (None, "eval"):
        run_eval(cfg, paths, recs, info, args.name, args.split, args.tta)


if __name__ == "__main__":
    main()
