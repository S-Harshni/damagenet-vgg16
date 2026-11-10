"""Label audit (optional): why we do NOT use the dronefreak/RDD2022 mirror.

The mirror is in YOLO format and maps the original codes D43 (crosswalk blur),
D44 (white-line blur) and D50 (manhole cover) into its class 3 "pothole". This script
samples random class-3 boxes from the mirror and shows them next to clean D40 patches
from our own dataset, so the difference is visible.

    python scripts/label_audit.py [--n 12]

Output: figures/label_audit.png. Needs the patch dataset (python -m src.run_all) for the
right-hand panel, and network access to Hugging Face.
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.figures import CHARCOAL, DPI, plt  # noqa: E402
from src.patches import cut_patch  # noqa: E402

MIRROR = "dronefreak/RDD2022"


def mirror_pothole_patches(n: int, seed: int, cache: Path) -> list[np.ndarray]:
    from huggingface_hub import HfApi, hf_hub_download
    api = HfApi()
    files = [f for f in api.list_repo_files(MIRROR, repo_type="dataset")
             if f.startswith("data/labels/") and f.endswith(".txt")]
    rng = random.Random(seed)
    rng.shuffle(files)
    out = []
    for lf in files:
        if len(out) >= n:
            break
        txt = Path(hf_hub_download(MIRROR, lf, repo_type="dataset", local_dir=str(cache))).read_text()
        rows = [r.split() for r in txt.splitlines() if r.strip()]
        rows = [r for r in rows if r[0] == "3"]
        if not rows:
            continue
        stem = Path(lf).stem
        img_rel = None
        for ext in (".jpg", ".png", ".jpeg"):
            cand = lf.replace("data/labels/", "data/images/").rsplit(".", 1)[0] + ext
            try:
                img_rel = hf_hub_download(MIRROR, cand, repo_type="dataset", local_dir=str(cache))
                break
            except Exception:
                continue
        if img_rel is None:
            print(f"no image for {stem}")
            continue
        img = Image.open(img_rel).convert("RGB")
        _, cx, cy, w, h = map(float, rows[0][:5])  # YOLO: normalised centre + size
        W, H = img.size
        box = [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]
        out.append(cut_patch(img, box, 128, 1.15))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    cache = ROOT / "data" / "raw_mirror"
    left = mirror_pothole_patches(args.n, args.seed, cache)

    right = []
    npz = ROOT / "data" / "patches" / "train.npz"
    if npz.exists():
        d = np.load(npz)
        idx = np.flatnonzero(d["y"] == 3)
        right = list(d["x"][np.random.default_rng(args.seed).choice(idx, args.n, replace=False)])

    cols = 4
    rows = int(np.ceil(args.n / cols))
    fig, axes = plt.subplots(rows, 2 * cols + 1, figsize=(2 * cols * 1.5 + 0.4, rows * 1.6),
                             gridspec_kw={"width_ratios": [1] * cols + [0.15] + [1] * cols})
    for ax in axes.ravel():
        ax.axis("off")
    for k, p in enumerate(left):
        axes[k // cols, k % cols].imshow(p)
    for k, p in enumerate(right):
        axes[k // cols, cols + 1 + k % cols].imshow(p)
    axes[0, 0].set_title("dronefreak mirror — class 3 \"pothole\"", loc="left", fontsize=10,
                         fontweight="bold", color=CHARCOAL)
    axes[0, cols + 1].set_title("Original RDD2022 — D40 pothole (ours)", loc="left", fontsize=10,
                                fontweight="bold", color=CHARCOAL)
    out = ROOT / "figures" / "label_audit.png"
    out.parent.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
