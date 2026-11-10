"""Build the class-balanced 128x128 patch dataset from the parsed annotations.

Steps
1. Seeded 70/15/15 random split at the IMAGE level (an image never appears in two splits).
2. For every eligible box (shorter side >= 16 px and area >= 900 px^2) we can cut a square
   patch centred on the box with side = 1.15 * max(w, h).
3. Class-balanced sampling: 900/200/300 patches per class for train/val/test, at most
   2 patches of the same class from one image (avoids one image dominating a class).
4. Only the images that are actually needed are downloaded.
5. Patches are saved as compressed .npz per split + meta.json.
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from tqdm import tqdm

from .data import download_files
from .utils import Paths, get_logger, get_paths, load_config, load_json, save_json, setup_logging

SPLITS = ("train", "val", "test")


# --------------------------------------------------------------------------- split
def split_images(image_ids: list[str], fractions: dict, seed: int) -> dict[str, list[str]]:
    """Random image-level split. Sorting first makes the result independent of input order."""
    ids = sorted(set(image_ids))
    random.Random(seed).shuffle(ids)
    n = len(ids)
    n_train = int(round(fractions["train"] * n))
    n_val = int(round(fractions["val"] * n))
    return {"train": sorted(ids[:n_train]),
            "val": sorted(ids[n_train:n_train + n_val]),
            "test": sorted(ids[n_train + n_val:])}


# ---------------------------------------------------------------------------- crop
def box_ok(box, min_side: float, min_area: float) -> bool:
    x0, y0, x1, y1 = box
    w, h = x1 - x0, y1 - y0
    return min(w, h) >= min_side and w * h >= min_area


def square_crop_box(box, img_w: int, img_h: int, context: float = 1.15) -> tuple[int, int, int, int]:
    """Square window centred on ``box`` with side = context * max(w, h), kept inside the image.

    If the window would cross an edge it is shifted back inside (so it stays square);
    if it is larger than the image it is capped at min(img_w, img_h).
    Returns integer (left, top, right, bottom) with right-left == bottom-top.
    """
    x0, y0, x1, y1 = box
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    side = context * max(x1 - x0, y1 - y0)
    side = int(round(min(side, img_w, img_h)))
    side = max(side, 1)
    left = int(round(cx - side / 2.0))
    top = int(round(cy - side / 2.0))
    left = min(max(left, 0), img_w - side)
    top = min(max(top, 0), img_h - side)
    return left, top, left + side, top + side


def cut_patch(img: Image.Image, box, size: int = 128, context: float = 1.15) -> np.ndarray:
    """Cut the square context patch for one box and resize it (bilinear) to size x size."""
    crop = square_crop_box(box, img.width, img.height, context)
    patch = img.crop(crop).resize((size, size), Image.BILINEAR)
    return np.asarray(patch.convert("RGB"), dtype=np.uint8)


# -------------------------------------------------------------------------- sample
def sample_boxes(records: list[dict], split_ids: dict[str, list[str]], cfg: dict,
                 per_class: dict[str, int]) -> dict[str, list[dict]]:
    """Choose which boxes become patches. Uses only annotation info (no images needed)."""
    log = get_logger()
    pc = cfg["patches"]
    n_cls = len(cfg["dataset"]["classes"])
    by_id = {r["id"]: r for r in records}
    rng = random.Random(cfg["seed"])
    chosen: dict[str, list[dict]] = {}
    for split in SPLITS:
        cands = defaultdict(list)  # label -> list of (img_id, box index)
        for img_id in split_ids[split]:
            r = by_id[img_id]
            for k, b in enumerate(r["boxes"]):
                if box_ok(b["box"], pc["min_side"], pc["min_area"]):
                    cands[b["label"]].append((img_id, k))
        picks = []
        for label in range(n_cls):
            pool = sorted(cands[label])
            rng.shuffle(pool)
            used = Counter()
            n_target = per_class[split]
            got = 0
            for img_id, k in pool:
                if got >= n_target:
                    break
                if used[img_id] >= pc["max_per_class_per_image"]:
                    continue
                used[img_id] += 1
                got += 1
                r = by_id[img_id]
                picks.append({"id": img_id, "file": r["file"], "country": r["country"],
                              "label": label, "cls": r["boxes"][k]["cls"],
                              "box": r["boxes"][k]["box"]})
            if got < n_target:
                log.warning(f"{split}: only {got}/{n_target} patches available for class {label}")
        chosen[split] = picks
        log.info(f"{split}: {len(picks)} patches from {len({p['id'] for p in picks})} images "
                 f"({Counter(p['label'] for p in picks)})")
    return chosen


# ------------------------------------------------------------------------- build
def build_patches(cfg: dict, paths: Paths, debug: bool = False) -> None:
    log = get_logger()
    pc = cfg["patches"]
    records = load_json(paths.data / "ann.json")
    split_ids = split_images([r["id"] for r in records], cfg["split"], cfg["seed"])
    save_json(split_ids, paths.data / "splits.json")
    log.info("image split: " + ", ".join(f"{k}={len(v)}" for k, v in split_ids.items()))

    per_class = ({s: cfg["debug"]["per_class"] for s in SPLITS} if debug else pc["per_class"])
    chosen = sample_boxes(records, split_ids, cfg, per_class)

    needed = sorted({p["file"] for ps in chosen.values() for p in ps})
    failed = download_files(cfg["dataset"]["repo_id"], needed, paths.raw,
                            cfg["dataset"]["download_workers"], desc="images")
    failed = set(failed)

    meta = {}
    for split in SPLITS:
        xs, ys, rows = [], [], []
        # group by image so each JPEG is decoded once
        by_file = defaultdict(list)
        for p in chosen[split]:
            by_file[p["file"]].append(p)
        for f in tqdm(sorted(by_file), desc=f"patches {split}", mininterval=5):
            if f in failed:
                continue
            with Image.open(paths.raw / f) as im:
                im = im.convert("RGB")
                for p in by_file[f]:
                    xs.append(cut_patch(im, p["box"], pc["size"], pc["context"]))
                    ys.append(p["label"])
                    x0, y0, x1, y1 = p["box"]
                    rows.append({"file": f, "id": p["id"], "country": p["country"],
                                 "label": p["label"], "cls": p["cls"], "box": p["box"],
                                 "box_w": round(x1 - x0, 2), "box_h": round(y1 - y0, 2),
                                 "img_w": im.width, "img_h": im.height})
        x = np.stack(xs).astype(np.uint8)
        y = np.asarray(ys, dtype=np.int64)
        np.savez_compressed(paths.patches / f"{split}.npz", x=x, y=y)
        meta[split] = rows
        log.info(f"saved {split}.npz: x={x.shape} counts={np.bincount(y, minlength=4).tolist()}")
    save_json(meta, paths.patches / "meta.json")


def load_patches(paths: Paths, split: str) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    d = np.load(paths.patches / f"{split}.npz")
    meta = load_json(paths.patches / "meta.json")[split]
    return d["x"], d["y"], meta


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    paths = get_paths(cfg, args.debug).make()
    setup_logging(paths.logs, "patches")
    build_patches(cfg, paths, args.debug)


if __name__ == "__main__":
    main()
