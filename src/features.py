"""Feature caching for VGG16 (the main speed trick).

Blocks 1-4 (features[:24]) are ALWAYS frozen, so their output for a given patch never
changes. We run them once and cache the (512, 8, 8) maps for: train, train flipped
horizontally, val and test. Stage 1 additionally caches the frozen block-5 output
(512, 4, 4 = 8192 values) so the head trains in seconds; stage 2 runs only block 5 + head
on top of the cached block-4 maps.

AUGMENTATION NOTE: only HORIZONTAL flips are used. A vertical flip or a 90-degree rotation
would turn a longitudinal crack (D00, along the driving direction) into a transverse crack
(D10, across the road) and vice versa, so it would silently corrupt the labels.
Horizontal flips keep a crack's orientation relative to the road and are label-safe.
"""
from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from .models import IMAGENET_MEAN, IMAGENET_STD, load_vgg16_features
from .patches import load_patches
from .utils import Paths, get_logger

CACHE_SETS = ("train", "train_flip", "val", "test")


def to_tensor(x_uint8: np.ndarray) -> torch.Tensor:
    """(N, H, W, 3) uint8 -> (N, 3, H, W) float in [0, 1]."""
    return torch.from_numpy(x_uint8).permute(0, 3, 1, 2).float().div_(255.0)


def normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


@torch.no_grad()
def run_layers(layers: torch.nn.Module, x: np.ndarray | torch.Tensor, device, batch_size=64,
               images=True, desc="features") -> np.ndarray:
    """Run frozen layers over x in batches. If ``images`` x is uint8 NHWC patches."""
    layers.eval().to(device)
    outs = []
    for i in tqdm(range(0, len(x), batch_size), desc=desc, mininterval=5, leave=False):
        xb = x[i:i + batch_size]
        xb = normalize(to_tensor(xb)) if images else torch.as_tensor(xb).float()
        outs.append(layers(xb.to(device)).float().cpu().numpy())
    return np.concatenate(outs)


def build_caches(cfg: dict, paths: Paths, device) -> None:
    """Cache block-4 outputs (float16) for train/train_flip/val/test and block-5 outputs."""
    log = get_logger()
    vc = cfg["vgg"]
    feats = load_vgg16_features(pretrained=True)
    blocks14 = feats[:vc["frozen_layers"]]
    block5 = feats[vc["block5"][0]:vc["block5"][1]]
    for p in feats.parameters():
        p.requires_grad_(False)

    for name in CACHE_SETS:
        split = "train" if name == "train_flip" else name
        x, y, _ = load_patches(paths, split)
        if name == "train_flip":
            x = x[:, :, ::-1, :].copy()  # horizontal flip only (see module docstring)
        f4 = run_layers(blocks14, x, device, vc["feature_batch_size"], desc=f"block1-4 {name}")
        f5 = run_layers(block5, f4, device, vc["feature_batch_size"], images=False, desc=f"block5 {name}")
        np.save(paths.cache / f"b4_{name}.npy", f4.astype(np.float16))
        np.save(paths.cache / f"b5_{name}.npy", f5.reshape(len(f5), -1).astype(np.float32))
        np.save(paths.cache / f"y_{name}.npy", y)
        log.info(f"cached {name}: block4 {f4.shape}, block5 {f5.shape}")


def load_cache(paths: Paths, kind: str, name: str) -> tuple[np.ndarray, np.ndarray]:
    """kind is 'b4' or 'b5'; name in CACHE_SETS."""
    return np.load(paths.cache / f"{kind}_{name}.npy"), np.load(paths.cache / f"y_{name}.npy")
