"""Evaluation, error analysis and the results/summary.md report."""
from __future__ import annotations

import argparse
from collections import Counter

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             precision_recall_fscore_support)

from .utils import Paths, get_logger, get_paths, load_config, load_json, save_json, setup_logging

CRACKS = {0, 1, 2}  # D00, D10, D20 are crack types; D40 is pothole

MODEL_LABELS = {"baseline_cnn": "Baseline CNN (scratch, 64px)",
                "vgg16_A4": "VGG16 frozen (feature extraction, A4)",
                "vgg16_ft": "VGG16 fine-tuned (block 5)"}


@torch.no_grad()
def predict(model, X: torch.Tensor, device, batch_size=128) -> np.ndarray:
    """Softmax probabilities for every row of X."""
    model.eval().to(device)
    out = []
    for i in range(0, len(X), batch_size):
        out.append(torch.softmax(model(X[i:i + batch_size].to(device).float()), 1).cpu())
    return torch.cat(out).numpy()


def metrics(y_true: np.ndarray, probs: np.ndarray, classes: list[str]) -> dict:
    y_pred = probs.argmax(1)
    labels = list(range(len(classes)))
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, labels=labels, average="macro",
                                                 zero_division=0)
    return {
        "test_accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "macro_precision": round(float(p), 4),
        "macro_recall": round(float(r), 4),
        "macro_f1": round(float(f), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "classification_report": classification_report(y_true, y_pred, labels=labels,
                                                       target_names=classes, output_dict=True,
                                                       zero_division=0),
    }


def evaluate_run(name, history, best_epoch, y_true, probs, cfg, paths: Paths, extra=None) -> dict:
    """Compute metrics for one run and save everything to results/<name>.json."""
    classes = cfg["dataset"]["classes"]
    res = {"name": name, "history": history, "epochs_run": len(history["val_loss"]),
           "best_epoch": best_epoch, **metrics(np.asarray(y_true), probs, classes),
           "predictions": {"y_true": np.asarray(y_true).tolist(), "y_pred": probs.argmax(1).tolist(),
                           "probs": np.round(probs, 4).tolist()}}
    if extra:
        res.update(extra)
    save_json(res, paths.results / f"{name}.json")
    get_logger().info(f"[{name}] TEST acc {res['test_accuracy']:.4f} | macro P {res['macro_precision']:.4f} "
                      f"R {res['macro_recall']:.4f} F1 {res['macro_f1']:.4f}")
    return res


# -------------------------------------------------------------------- analysis
def analyse(cfg: dict, paths: Paths, run: str = "vgg16_ft") -> dict:
    """Error analysis of one run: by country, by box-size thirds, top confusions,
    and the share of errors that are crack-type confusions."""
    classes = cfg["dataset"]["classes"]
    res = load_json(paths.results / f"{run}.json")
    meta = load_json(paths.patches / "meta.json")["test"]
    yt = np.array(res["predictions"]["y_true"])
    yp = np.array(res["predictions"]["y_pred"])
    assert len(meta) == len(yt), "meta.json and predictions are out of sync"
    correct = yt == yp

    by_country = {}
    countries = np.array([m["country"] for m in meta])
    for c in sorted(set(countries)):
        m = countries == c
        by_country[c] = {"n": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4)}

    size = np.array([max(m["box_w"], m["box_h"]) for m in meta])
    q1, q2 = np.quantile(size, [1 / 3, 2 / 3])
    bins = {"small": size <= q1, "medium": (size > q1) & (size <= q2), "large": size > q2}
    by_size = {k: {"n": int(m.sum()), "accuracy": round(float(correct[m].mean()), 4) if m.any() else None,
                   "max_side_range_px": [round(float(size[m].min()), 1), round(float(size[m].max()), 1)]
                   if m.any() else None}
               for k, m in bins.items()}

    errs = [(int(a), int(b)) for a, b in zip(yt[~correct], yp[~correct])]
    conf = Counter(errs)
    top = [{"true": classes[a], "pred": classes[b], "count": n} for (a, b), n in conf.most_common(6)]
    crack_errs = sum(n for (a, b), n in conf.items() if a in CRACKS and b in CRACKS)
    out = {"run": run, "n_test": int(len(yt)), "n_errors": int((~correct).sum()),
           "accuracy_by_country": by_country, "accuracy_by_box_size": by_size,
           "box_size_thirds_px": [round(float(q1), 1), round(float(q2), 1)],
           "top_confusions": top,
           "crack_confusion_share": round(crack_errs / max(1, len(errs)), 4)}
    save_json(out, paths.results / "analysis.json")
    get_logger().info(f"analysis: {out['n_errors']} errors, crack-type confusions "
                      f"{out['crack_confusion_share']:.1%}")
    return out


# -------------------------------------------------------------------- summary
def write_summary(cfg: dict, paths: Paths, timings: dict | None = None) -> str:
    classes = cfg["dataset"]["classes"]
    names = cfg["dataset"]["class_names"]
    R = {n: load_json(paths.results / f"{n}.json") for n in
         ["baseline_cnn", "vgg16_A0", "vgg16_A1", "vgg16_A2", "vgg16_A3", "vgg16_A4", "vgg16_ft"]}
    an = load_json(paths.results / "analysis.json")
    stats = load_json(paths.data / "stats.json")
    L = ["# DamageNet-VGG16 — results summary", "",
         "Road-damage patch classification on RDD2022 (4 classes, class-balanced test set of "
         f"{len(R['vgg16_ft']['predictions']['y_true'])} patches).", "",
         "## Main results", "",
         "| Model | Test accuracy | Macro precision | Macro recall | Macro F1 | Epochs (best) |",
         "|---|---|---|---|---|---|"]
    for k, label in MODEL_LABELS.items():
        r = R[k]
        L.append(f"| {label} | {r['test_accuracy']:.4f} | {r['macro_precision']:.4f} | "
                 f"{r['macro_recall']:.4f} | {r['macro_f1']:.4f} | {r['epochs_run']} ({r['best_epoch']}) |")
    L += ["", "## Regularisation ablation (stage 1 head, cumulative)", "",
          "| Run | Added | Test accuracy | Macro F1 | Final train acc | Final val acc |", "|---|---|---|---|---|---|"]
    added = {"A0": "none", "A1": "+ dropout 0.5", "A2": "+ L2 1e-3", "A3": "+ horizontal-flip copies",
             "A4": "+ early stopping"}
    for a, txt in added.items():
        r = R[f"vgg16_{a}"]
        L.append(f"| {a} | {txt} | {r['test_accuracy']:.4f} | {r['macro_f1']:.4f} | "
                 f"{r['history']['train_acc'][-1]:.3f} | {r['history']['val_acc'][-1]:.3f} |")
    L += ["", "## Per-class F1", "", "| Class | " + " | ".join(MODEL_LABELS.values()) + " |",
          "|---|" + "---|" * len(MODEL_LABELS)]
    for c, nm in zip(classes, names):
        L.append(f"| {c} {nm} | " + " | ".join(f"{R[k]['classification_report'][c]['f1-score']:.3f}"
                                                for k in MODEL_LABELS) + " |")
    L += ["", "## Error analysis (VGG16 fine-tuned)", "",
          f"- Errors: {an['n_errors']} / {an['n_test']}; "
          f"**{an['crack_confusion_share']:.1%} of errors are confusions between crack types** (D00/D10/D20).",
          "- Top confusions: " + ", ".join(f"{t['true']} → {t['pred']} ({t['count']})" for t in an["top_confusions"]),
          "- Accuracy by box size (thirds of max side, px): " + ", ".join(
              f"{k} {v['max_side_range_px'][0]:.0f}–{v['max_side_range_px'][1]:.0f}: {v['accuracy']:.3f}"
              for k, v in an["accuracy_by_box_size"].items()),
          "- Accuracy by country: " + ", ".join(f"{k} {v['accuracy']:.3f} (n={v['n']})"
                                                 for k, v in an["accuracy_by_country"].items()),
          "", "## Data", "",
          f"- Annotated images parsed: {stats['n_images']}; boxes kept: " +
          ", ".join(f"{k} {v}" for k, v in stats["overall"].items()),
          "- Patches per class: train/val/test = " + "/".join(
              str(len(v) // len(classes)) for v in load_json(paths.patches / "meta.json").values()),
          ""]
    if timings:
        L += ["## Runtime", ""] + [f"- {k}: {v / 60:.1f} min" for k, v in timings.items()] + [""]
    text = "\n".join(L)
    (paths.results / "summary.md").write_text(text, encoding="utf-8")
    return text


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config()
    paths = get_paths(cfg, args.debug).make()
    setup_logging(paths.logs, "evaluate")
    analyse(cfg, paths)
    print(write_summary(cfg, paths))


if __name__ == "__main__":
    main()
