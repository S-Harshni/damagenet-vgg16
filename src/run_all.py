"""One-command pipeline: download -> parse -> patches -> train (baseline, VGG16 stage 1
ablation, stage 2 fine-tuning) -> evaluate/analyse -> figures -> results/summary.md.

    python -m src.run_all            # full run (~1 h on a laptop CPU, faster on GPU/MPS)
    python -m src.run_all --debug    # tiny smoke test, outputs go to debug/
    python -m src.run_all --skip-data  # reuse data/ann.json and patches already built
"""
from __future__ import annotations

import argparse
import time

from . import data, evaluate, figures, patches, train
from .utils import get_device, get_paths, load_config, save_json, set_seed, setup_logging


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="path to a YAML config (default configs/default.yaml)")
    ap.add_argument("--debug", action="store_true", help="tiny subset, 2 epochs, outputs under debug/")
    ap.add_argument("--skip-data", action="store_true", help="reuse existing ann.json and patches")
    args = ap.parse_args(argv)

    cfg = load_config(args.config) if args.config else load_config()
    paths = get_paths(cfg, args.debug).make()
    log = setup_logging(paths.logs, "run_all_debug" if args.debug else "run_all")
    set_seed(cfg["seed"])
    log.info(f"DamageNet-VGG16 pipeline | debug={args.debug} | device={get_device(cfg.get('device', 'auto'))}")

    timings = {}

    def step(name, fn):
        t0 = time.time()
        log.info(f"===== {name} =====")
        fn()
        timings[name] = round(time.time() - t0, 1)
        log.info(f"===== {name} done in {timings[name] / 60:.1f} min =====")

    t_start = time.time()
    if not args.skip_data:
        step("1. download + parse annotations", lambda: data.build_annotations(cfg, paths, args.debug))
        step("2. build patch dataset", lambda: patches.build_patches(cfg, paths, args.debug))
    device = get_device(cfg.get("device", "auto"))
    step("3. baseline CNN", lambda: train.train_baseline(cfg, paths, device, args.debug))
    step("4. VGG16 feature caching", lambda: train.build_caches(cfg, paths, device))
    step("5. VGG16 stage 1 + ablation", lambda: train.train_stage1(cfg, paths, device, args.debug))
    step("6. VGG16 stage 2 fine-tuning", lambda: train.train_stage2(cfg, paths, device, args.debug))
    step("7. analysis", lambda: evaluate.analyse(cfg, paths))
    step("8. figures", lambda: figures.make_all(cfg, paths))
    timings["total"] = round(time.time() - t_start, 1)
    save_json({"timings_seconds": timings, "device": str(device), "debug": args.debug},
              paths.results / "run_info.json")
    text = evaluate.write_summary(cfg, paths, timings)
    log.info("\n" + text)
    log.info(f"pipeline finished in {timings['total'] / 60:.1f} min")


if __name__ == "__main__":
    main()
