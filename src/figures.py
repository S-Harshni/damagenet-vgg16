"""All report figures (matplotlib, 200 dpi) saved to figures/.

Palette: charcoal #2B2D2F, orange #FF6B35, yellow #FFC845 and greys. Because yellow and
grey are light on a white background, every bar carries a direct value label and a thin
charcoal edge so no value relies on colour alone.
"""
from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

from .patches import load_patches  # noqa: E402
from .utils import Paths, get_logger, get_paths, load_config, load_json, setup_logging  # noqa: E402

CHARCOAL, ORANGE, YELLOW = "#2B2D2F", "#FF6B35", "#FFC845"
GREY, LIGHT_GREY, GRID = "#8A8D91", "#C9CBCE", "#E6E7E8"
CLASS_COLORS = [CHARCOAL, ORANGE, YELLOW, GREY]  # D00, D10, D20, D40 (fixed order)
DPI = 200

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": CHARCOAL, "text.color": CHARCOAL,
    "xtick.color": CHARCOAL, "ytick.color": CHARCOAL, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "axes.titleweight": "bold", "figure.facecolor": "white",
    "legend.frameon": False,
})


def _save(fig, paths: Paths, name: str):
    fig.tight_layout()
    fig.savefig(paths.figures / name, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    get_logger().info(f"figure saved: figures/{name}")


def _bar_labels(ax, bars, fmt="{:.3f}", fontsize=8):
    for b in bars:
        ax.annotate(fmt.format(b.get_height()), (b.get_x() + b.get_width() / 2, b.get_height()),
                    xytext=(0, 2), textcoords="offset points", ha="center", va="bottom",
                    fontsize=fontsize, color=CHARCOAL)


def label_distribution(cfg, paths):
    stats = load_json(paths.data / "stats.json")
    classes, names = cfg["dataset"]["classes"], cfg["dataset"]["class_names"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"width_ratios": [1, 1.6]})
    vals = [stats["overall"][c] for c in classes]
    bars = a1.bar([f"{c}\n" + n.replace(" ", "\n") for c, n in zip(classes, names)], vals, color=CLASS_COLORS,
                  edgecolor=CHARCOAL, linewidth=0.6, width=0.65)
    _bar_labels(a1, bars, "{:,.0f}")
    a1.set_title(f"Boxes per class ({stats['n_images']:,} images)")
    a1.set_ylabel("Number of boxes")
    a1.margins(y=0.1)
    a1.grid(axis="x", visible=False)

    countries = list(stats["per_country"])
    bottom = np.zeros(len(countries))
    for c, col in zip(classes, CLASS_COLORS):
        v = np.array([stats["per_country"][k][c] for k in countries])
        a2.barh([k.replace("_", " ") for k in countries], v, left=bottom, color=col,
                edgecolor="white", linewidth=1.0, label=c)
        bottom += v
    for i, t in enumerate(bottom):
        a2.annotate(f"{int(t):,}", (t, i), xytext=(3, 0), textcoords="offset points", va="center", fontsize=8)
    a2.set_title("Boxes per country and class")
    a2.set_xlabel("Number of boxes")
    a2.legend(ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.32))
    a2.grid(axis="y", visible=False)
    a2.invert_yaxis()
    _save(fig, paths, "label_distribution.png")


def sample_patches(cfg, paths, n_per_class=8):
    x, y, _ = load_patches(paths, "train")
    rng = np.random.default_rng(cfg["seed"])
    classes, names = cfg["dataset"]["classes"], cfg["dataset"]["class_names"]
    fig, axes = plt.subplots(len(classes), n_per_class, figsize=(n_per_class * 1.3, len(classes) * 1.45))
    for r, c in enumerate(classes):
        idx = np.flatnonzero(y == r)
        pick = rng.choice(idx, size=min(n_per_class, len(idx)), replace=False)
        for k in range(n_per_class):
            ax = axes[r, k]
            ax.axis("off")
            if k < len(pick):
                ax.imshow(x[pick[k]])
        axes[r, 0].set_title(f"{c} {names[r]}", loc="left", fontsize=9, fontweight="bold")
    fig.suptitle("Sample 128×128 training patches per class", fontweight="bold")
    _save(fig, paths, "sample_patches.png")


def learning_curves(cfg, paths):
    runs = {"A0 (no regularisation)": (load_json(paths.results / "vgg16_A0.json"), CHARCOAL),
            "A4 (dropout + L2 + flip + early stop)": (load_json(paths.results / "vgg16_A4.json"), ORANGE)}
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, metric, title in [(axes[0], "acc", "Accuracy"), (axes[1], "loss", "Cross-entropy loss")]:
        for label, (r, col) in runs.items():
            h = r["history"]
            ep = np.arange(1, len(h[f"train_{metric}"]) + 1)
            ax.plot(ep, h[f"train_{metric}"], color=col, lw=2, label=f"{label} — train")
            ax.plot(ep, h[f"val_{metric}"], color=col, lw=2, ls="--", label=f"{label} — val")
            if r.get("ablation", {}).get("early_stopping"):
                ax.axvline(r["best_epoch"], color=col, lw=1, ls=":", alpha=0.8)
        ax.set_title(f"{title}: A0 vs A4 (VGG16 stage 1)")
        ax.set_xlabel("Epoch")
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.08))
    _save(fig, paths, "learning_curves.png")


def model_comparison(cfg, paths):
    from .evaluate import MODEL_LABELS
    labels = ["Baseline CNN\n(scratch)", "VGG16 frozen\n(A4)", "VGG16 fine-tuned\n(block 5)"]
    R = [load_json(paths.results / f"{k}.json") for k in MODEL_LABELS]
    xs, w = np.arange(len(R)), 0.36
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(xs - w / 2, [r["test_accuracy"] for r in R], w, color=CHARCOAL, label="Test accuracy")
    b2 = ax.bar(xs + w / 2, [r["macro_f1"] for r in R], w, color=ORANGE, label="Macro F1")
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    ax.set_xticks(xs, labels)
    ax.set_ylim(0, 1)
    ax.set_title("Model comparison on the balanced test set")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    _save(fig, paths, "model_comparison.png")


def ablation(cfg, paths):
    names = ["A0", "A1", "A2", "A3", "A4"]
    added = ["none", "+dropout", "+L2", "+flip", "+early stop"]
    R = [load_json(paths.results / f"vgg16_{n}.json") for n in names]
    xs, w = np.arange(len(R)), 0.36
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    b1 = ax.bar(xs - w / 2, [r["test_accuracy"] for r in R], w, color=CHARCOAL, label="Test accuracy")
    b2 = ax.bar(xs + w / 2, [r["macro_f1"] for r in R], w, color=ORANGE, label="Macro F1")
    _bar_labels(ax, b1)
    _bar_labels(ax, b2)
    gap = [r["history"]["train_acc"][-1] - r["history"]["val_acc"][-1] for r in R]
    ax.set_xticks(xs, [f"{n}\n{a}\n(train–val gap {g:+.2f})" for n, a, g in zip(names, added, gap)], fontsize=8.5)
    lo = min(min(r["test_accuracy"], r["macro_f1"]) for r in R)
    ax.set_ylim(max(0, lo - 0.1), min(1, max(r["test_accuracy"] for r in R) + 0.06))
    ax.set_title("Regularisation ablation (cumulative, VGG16 stage 1 head)")
    ax.legend(loc="upper left", ncol=2)
    ax.grid(axis="x", visible=False)
    _save(fig, paths, "ablation.png")


def confusion(cfg, paths, run="vgg16_ft"):
    r = load_json(paths.results / f"{run}.json")
    cm = np.array(r["confusion_matrix"])
    classes = cfg["dataset"]["classes"]
    pct = cm / cm.sum(1, keepdims=True).clip(min=1)
    cmap = LinearSegmentedColormap.from_list("orange_seq", ["#FFFFFF", "#FFD9C9", ORANGE, "#B5401A"])
    fig, ax = plt.subplots(figsize=(5.6, 4.8))
    im = ax.imshow(pct, cmap=cmap, vmin=0, vmax=1)
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, f"{cm[i, j]}\n{pct[i, j]:.0%}", ha="center", va="center", fontsize=9,
                    color="white" if pct[i, j] > 0.6 else CHARCOAL)
    names = [f"{c}\n{n}" for c, n in zip(classes, cfg["dataset"]["class_names"])]
    ax.set_xticks(range(len(classes)), names, fontsize=8)
    ax.set_yticks(range(len(classes)), names, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.grid(False)
    ax.set_title(f"Confusion matrix — VGG16 fine-tuned (acc {r['test_accuracy']:.3f})")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Row share")
    _save(fig, paths, "confusion_matrix.png")


def misclassified(cfg, paths, run="vgg16_ft", n=24):
    r = load_json(paths.results / f"{run}.json")
    x, _, meta = load_patches(paths, "test")
    yt, yp = np.array(r["predictions"]["y_true"]), np.array(r["predictions"]["y_pred"])
    probs = np.array(r["predictions"]["probs"])
    wrong = np.flatnonzero(yt != yp)
    rng = np.random.default_rng(cfg["seed"])
    pick = rng.choice(wrong, size=min(n, len(wrong)), replace=False) if len(wrong) else []
    classes = cfg["dataset"]["classes"]
    cols = 6
    rows = max(1, int(np.ceil(len(pick) / cols)))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.75, rows * 2.0), squeeze=False)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, i in zip(axes.ravel(), pick):
        ax.imshow(x[i])
        ax.set_title(f"{classes[yt[i]]} → {classes[yp[i]]}  ({probs[i, yp[i]]:.2f})\n"
                     f"{meta[i]['country'].replace('_', ' ')}", fontsize=7.5)
    fig.suptitle("Misclassified test patches (true → pred, confidence) — VGG16 fine-tuned", fontweight="bold")
    _save(fig, paths, "misclassified.png")


def make_all(cfg, paths):
    for fn in (label_distribution, sample_patches, learning_curves, model_comparison, ablation,
               confusion, misclassified):
        fn(cfg, paths)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    paths = get_paths(cfg, args.debug).make()
    setup_logging(paths.logs, "figures")
    make_all(cfg, paths)


if __name__ == "__main__":
    main()
