"""Training: scratch baseline CNN, VGG16 stage 1 (feature extraction + regularisation
ablation A0-A4) and VGG16 stage 2 (fine-tuning block 5).

Augmentation: only random HORIZONTAL flips. Never vertical flips or 90-degree rotations:
they would swap longitudinal (D00) and transverse (D10) cracks and corrupt the labels.
"""
from __future__ import annotations

import argparse
import copy
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .evaluate import evaluate_run, predict
from .features import build_caches, load_cache, to_tensor
from .models import BaselineCNN, Block5Head, VGG16Classifier, VGGHead, load_vgg16_features
from .patches import load_patches
from .utils import Paths, get_device, get_logger, get_paths, load_config, set_seed, setup_logging


# ------------------------------------------------------------------ generic loop
def run_epoch(model, X, y, device, batch_size, optimizer=None, augment=None, l2_params=None,
              l2=0.0, generator=None):
    """One pass over (X, y). Trains if an optimizer is given, otherwise evaluates.
    Returns (mean cross-entropy loss, accuracy). The L2 penalty is not included in the
    reported loss so train/val losses stay comparable."""
    train = optimizer is not None
    model.train(train)
    n = len(X)
    order = torch.randperm(n, generator=generator) if train else torch.arange(n)
    tot_loss, correct = 0.0, 0
    with torch.set_grad_enabled(train):
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = X[idx].to(device).float()
            yb = y[idx].to(device)
            if train and augment is not None:
                xb = augment(xb, generator)
            out = model(xb)
            loss = F.cross_entropy(out, yb)
            if train:
                total = loss
                if l2_params and l2 > 0:
                    # Keras-style L2 kernel regulariser: l2 * sum(w^2)
                    total = total + l2 * sum((w ** 2).sum() for w in l2_params)
                optimizer.zero_grad(set_to_none=True)
                total.backward()
                optimizer.step()
            tot_loss += loss.item() * len(idx)
            correct += (out.argmax(1) == yb).sum().item()
    return tot_loss / n, correct / n


def fit(model, train_xy, val_xy, optimizer, epochs, batch_size, device, seed, augment=None,
        l2_params=None, l2=0.0, patience=None, name="model"):
    """Train for up to ``epochs``. With ``patience`` set, early stopping on val loss and
    the best weights are restored. Returns (history, best_epoch [1-based])."""
    log = get_logger()
    g = torch.Generator().manual_seed(seed)
    hist = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_loss, best_state, best_epoch, bad = float("inf"), None, epochs, 0
    for ep in range(1, epochs + 1):
        t0 = time.time()
        tl, ta = run_epoch(model, *train_xy, device, batch_size, optimizer, augment, l2_params, l2, g)
        vl, va = run_epoch(model, *val_xy, device, batch_size)
        for k, v in zip(hist, (tl, ta, vl, va)):
            hist[k].append(round(v, 5))
        log.info(f"[{name}] epoch {ep:02d}/{epochs} train loss {tl:.4f} acc {ta:.3f} | "
                 f"val loss {vl:.4f} acc {va:.3f} ({time.time() - t0:.1f}s)")
        if patience is not None:
            if vl < best_loss - 1e-6:
                best_loss, best_epoch, bad = vl, ep, 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                bad += 1
                if bad >= patience:
                    log.info(f"[{name}] early stopping at epoch {ep}; best epoch {best_epoch}")
                    break
    if patience is not None and best_state is not None:
        model.load_state_dict(best_state)  # restore best weights
    else:
        best_epoch = len(hist["val_loss"])
    return hist, best_epoch


def hflip_batch(xb: torch.Tensor, generator) -> torch.Tensor:
    """Randomly flip half of the batch horizontally (last dim = width). No vertical flips."""
    mask = torch.rand(len(xb), generator=generator) < 0.5
    if mask.any():
        xb = xb.clone()
        xb[mask.to(xb.device)] = xb[mask.to(xb.device)].flip(-1)
    return xb


def epochs_for(cfg, key, debug):
    return cfg["debug"]["epochs"] if debug else key


# ------------------------------------------------------------------ baseline CNN
def train_baseline(cfg: dict, paths: Paths, device, debug=False) -> dict:
    bc = cfg["baseline"]
    set_seed(cfg["seed"])
    data = {}
    for s in ("train", "val", "test"):
        x, y, _ = load_patches(paths, s)
        x = F.interpolate(to_tensor(x), size=(bc["input_size"],) * 2, mode="bilinear",
                          align_corners=False, antialias=True)
        data[s] = (x, torch.from_numpy(y))
    mean = data["train"][0].mean(dim=(0, 2, 3), keepdim=True)
    std = data["train"][0].std(dim=(0, 2, 3), keepdim=True)
    data = {s: ((x - mean) / std, y) for s, (x, y) in data.items()}

    model = BaselineCNN(bc["channels"], bc["fc"], bc["dropout"], len(cfg["dataset"]["classes"]),
                        bc["input_size"]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=bc["lr"], weight_decay=bc["weight_decay"])
    t0 = time.time()
    hist, best = fit(model, data["train"], data["val"], opt, epochs_for(cfg, bc["max_epochs"], debug),
                     bc["batch_size"], device, cfg["seed"], augment=hflip_batch if bc["hflip"] else None,
                     patience=bc["patience"], name="baseline")
    secs = time.time() - t0
    probs = predict(model, data["test"][0], device, bc["batch_size"])
    torch.save({"state_dict": model.state_dict(), "mean": mean, "std": std, "config": bc},
               paths.checkpoints / "baseline_cnn.pt")
    return evaluate_run("baseline_cnn", hist, best, data["test"][1].numpy(), probs, cfg, paths,
                        extra={"train_seconds": round(secs, 1),
                               "n_params": sum(p.numel() for p in model.parameters())})


# ------------------------------------------------------------------ VGG16 stage 1
def stage1_data(paths: Paths, flip: bool):
    xtr, ytr = load_cache(paths, "b5", "train")
    if flip:  # A3+: add the horizontally flipped copies of every training patch
        xf, yf = load_cache(paths, "b5", "train_flip")
        xtr, ytr = np.concatenate([xtr, xf]), np.concatenate([ytr, yf])
    xva, yva = load_cache(paths, "b5", "val")
    xte, yte = load_cache(paths, "b5", "test")
    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    return (t(xtr), t(ytr)), (t(xva), t(yva)), (t(xte), t(yte))


def train_stage1(cfg: dict, paths: Paths, device, debug=False) -> dict:
    """Regularisation ablation on the head over cached block-5 features. Returns all runs."""
    s1 = cfg["stage1"]
    n_cls = len(cfg["dataset"]["classes"])
    results = {}
    for name, flags in s1["ablation"].items():
        set_seed(cfg["seed"])
        tr, va, te = stage1_data(paths, flags["flip"])
        head = VGGHead(tr[0].shape[1], cfg["vgg"]["hidden"],
                       s1["dropout"] if flags["dropout"] else 0.0, n_cls).to(device)
        opt = torch.optim.Adam(head.parameters(), lr=s1["lr"])
        t0 = time.time()
        hist, best = fit(head, tr, va, opt, epochs_for(cfg, s1["epochs"], debug), s1["batch_size"],
                         device, cfg["seed"], l2_params=head.dense_weights() if flags["l2"] else None,
                         l2=s1["l2"], patience=s1["patience"] if flags["early_stopping"] else None,
                         name=f"vgg16_{name}")
        secs = time.time() - t0
        probs = predict(head, te[0], device, 256)
        results[name] = evaluate_run(f"vgg16_{name}", hist, best, te[1].numpy(), probs, cfg, paths,
                                     extra={"ablation": flags, "train_seconds": round(secs, 1)})
        if name == "A4":
            torch.save({"state_dict": head.state_dict()}, paths.checkpoints / "vgg16_head_A4.pt")
    return results


# ------------------------------------------------------------------ VGG16 stage 2
def train_stage2(cfg: dict, paths: Paths, device, debug=False) -> dict:
    """Fine-tune block 5 + head (initialised from A4) on cached block-4 maps."""
    s2, vc = cfg["stage2"], cfg["vgg"]
    n_cls = len(cfg["dataset"]["classes"])
    set_seed(cfg["seed"])
    feats = load_vgg16_features(pretrained=True)
    block5 = feats[vc["block5"][0]:vc["block5"][1]]
    head = VGGHead(8192, vc["hidden"], cfg["stage1"]["dropout"], n_cls)
    head.load_state_dict(torch.load(paths.checkpoints / "vgg16_head_A4.pt", map_location="cpu")["state_dict"])
    model = Block5Head(block5, head).to(device)
    for p in model.parameters():
        p.requires_grad_(True)

    t = lambda a: torch.from_numpy(np.ascontiguousarray(a))
    xtr, ytr = load_cache(paths, "b4", "train")
    if s2["flip"]:
        xf, yf = load_cache(paths, "b4", "train_flip")
        xtr, ytr = np.concatenate([xtr, xf]), np.concatenate([ytr, yf])
    xva, yva = load_cache(paths, "b4", "val")
    xte, yte = load_cache(paths, "b4", "test")

    opt = torch.optim.Adam([{"params": model.block5.parameters(), "lr": s2["lr_block5"]},
                            {"params": model.head.parameters(), "lr": s2["lr_head"]}],
                           weight_decay=s2["weight_decay"])
    t0 = time.time()
    hist, best = fit(model, (t(xtr), t(ytr)), (t(xva), t(yva)), opt,
                     epochs_for(cfg, s2["max_epochs"], debug), s2["batch_size"], device, cfg["seed"],
                     patience=s2["patience"], name="vgg16_ft")
    secs = time.time() - t0
    probs = predict(model, t(xte), device, 128)

    # Assemble and save the full end-to-end model (blocks 1-4 pretrained + tuned block 5 + head).
    full = VGG16Classifier(vc["hidden"], cfg["stage1"]["dropout"], n_cls, pretrained=True,
                           input_size=vc["input_size"])
    full.features[vc["block5"][0]:vc["block5"][1]].load_state_dict(model.block5.state_dict())
    full.head.load_state_dict(model.head.state_dict())
    ckpt = {"state_dict": full.cpu().state_dict(), "classes": cfg["dataset"]["classes"],
            "class_names": cfg["dataset"]["class_names"], "input_size": vc["input_size"],
            "hidden": vc["hidden"], "dropout": cfg["stage1"]["dropout"],
            "context": cfg["patches"]["context"], "test_accuracy_cached": None}

    # Sanity check: the assembled model on raw test patches must match the cached-path result.
    x_test, y_test, _ = load_patches(paths, "test")
    probs_full = predict(full.to(device), to_tensor(x_test), device, 64)
    acc_cached = float((probs.argmax(1) == yte).mean())
    acc_full = float((probs_full.argmax(1) == y_test).mean())
    get_logger().info(f"[vgg16_ft] end-to-end check: cached-path acc {acc_cached:.4f}, "
                      f"full-model acc {acc_full:.4f}")
    ckpt["test_accuracy_cached"] = acc_cached
    ckpt["test_accuracy_full"] = acc_full
    torch.save(ckpt, paths.checkpoints / "vgg16_ft.pt")
    return evaluate_run("vgg16_ft", hist, best, yte, probs, cfg, paths,
                        extra={"train_seconds": round(secs, 1), "end_to_end_accuracy": acc_full})


def train_all(cfg: dict, paths: Paths, debug=False) -> None:
    log = get_logger()
    device = get_device(cfg.get("device", "auto"))
    log.info(f"device: {device}")
    train_baseline(cfg, paths, device, debug)
    t0 = time.time()
    build_caches(cfg, paths, device)
    log.info(f"feature caching took {time.time() - t0:.1f}s")
    train_stage1(cfg, paths, device, debug)
    train_stage2(cfg, paths, device, debug)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--debug", action="store_true")
    ap.add_argument("--only", choices=["baseline", "cache", "stage1", "stage2"], default=None)
    args = ap.parse_args(argv)
    cfg = load_config()
    paths = get_paths(cfg, args.debug).make()
    setup_logging(paths.logs, "train")
    device = get_device(cfg.get("device", "auto"))
    if args.only is None:
        train_all(cfg, paths, args.debug)
    elif args.only == "baseline":
        train_baseline(cfg, paths, device, args.debug)
    elif args.only == "cache":
        build_caches(cfg, paths, device)
    elif args.only == "stage1":
        train_stage1(cfg, paths, device, args.debug)
    else:
        train_stage2(cfg, paths, device, args.debug)


if __name__ == "__main__":
    main()
