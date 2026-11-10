"""Shared helpers: configuration, output paths, seeding, logging and device selection."""
from __future__ import annotations

import json
import logging
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
LOGGER_NAME = "damagenet"


def load_config(path: str | Path = ROOT / "configs" / "default.yaml") -> dict:
    """Load the YAML config into a plain dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@dataclass
class Paths:
    """All output locations. In --debug mode every derived file goes under debug/
    so a debug run never overwrites the real results. The raw download cache is shared."""

    raw: Path
    data: Path
    results: Path
    figures: Path
    checkpoints: Path
    logs: Path

    @property
    def patches(self) -> Path:
        return self.data / "patches"

    @property
    def cache(self) -> Path:
        return self.data / "cache"

    def make(self) -> "Paths":
        for p in (self.raw, self.data, self.results, self.figures, self.checkpoints,
                  self.logs, self.patches, self.cache):
            p.mkdir(parents=True, exist_ok=True)
        return self


def get_paths(cfg: dict, debug: bool = False) -> Paths:
    base = ROOT / "debug" if debug else ROOT
    p = cfg["paths"]
    results = base / p["results_dir"]
    return Paths(
        raw=ROOT / p["raw_dir"],
        data=base / p["data_dir"],
        results=results,
        figures=base / p["figures_dir"],
        checkpoints=base / p["checkpoints_dir"],
        logs=results / "logs",
    )


def set_seed(seed: int) -> None:
    """Fix every random number generator we use (python, numpy, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(pref: str = "auto") -> torch.device:
    """Pick CUDA, then Apple MPS, then CPU (or honour an explicit choice)."""
    if pref != "auto":
        return torch.device(pref)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def setup_logging(log_dir: Path | None = None, name: str = "run") -> logging.Logger:
    """Log to the console and (optionally) to results/logs/<name>_<timestamp>.log."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%H:%M:%S")
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_dir / f"{name}_{time.strftime('%Y%m%d_%H%M%S')}.log")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    logger.propagate = False
    return logger


def get_logger() -> logging.Logger:
    logger = logging.getLogger(LOGGER_NAME)
    if not logger.handlers:
        setup_logging()
    return logger


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
