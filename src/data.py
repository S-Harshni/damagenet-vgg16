"""Download the ORIGINAL RDD2022 Pascal-VOC annotations and parse them.

Source: Hugging Face dataset ``FelipeS09/urmind-rdd2022`` which mirrors the original
RDD2022 folder layout (data/RDD2022/<Country>/train/{images,annotations/xmls}).

We deliberately do NOT use ``dronefreak/RDD2022``: its class 3 "pothole" merges the
D43/D44/D50 codes (crosswalk blur, white-line blur, manhole cover), so most of its
"potholes" are lane markings, crosswalks and manholes (see scripts/label_audit.py).

Outputs: data/ann.json (one record per image) and data/stats.json (label counts).
"""
from __future__ import annotations

import argparse
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import defusedxml.ElementTree as ET  # safe XML parsing (no entity expansion attacks)
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from .utils import Paths, get_logger, get_paths, load_config, save_json, setup_logging

DATA_PREFIX = "data/RDD2022"


# --------------------------------------------------------------------------- lists
def read_split_list(path: Path) -> list[str]:
    """Read one split list (CRLF line endings) and return 'Country/name' ids.

    Lines look like ``datasets/raw/rdd2022/RDD2022/India/train/images/India_000000.jpg\\r``.
    We only take the file name and country from them; their split is NOT used because it
    is split by country (train/val/test would contain different countries).
    """
    ids = []
    with open(path, "r", encoding="utf-8", newline="") as f:
        for line in f:
            line = line.strip().replace("\r", "")
            if not line:
                continue
            parts = line.split("/")
            if "RDD2022" not in parts:
                continue
            i = parts.index("RDD2022")
            country, name = parts[i + 1], Path(parts[-1]).stem
            ids.append(f"{country}/{name}")
    return ids


def image_repo_path(img_id: str) -> str:
    country, name = img_id.split("/")
    return f"{DATA_PREFIX}/{country}/train/images/{name}.jpg"


def xml_repo_path(img_id: str) -> str:
    country, name = img_id.split("/")
    return f"{DATA_PREFIX}/{country}/train/annotations/xmls/{name}.xml"


# ------------------------------------------------------------------------ download
def download_files(repo_id: str, repo_paths: list[str], raw_dir: Path, workers: int = 16,
                   desc: str = "download") -> list[str]:
    """Parallel, resumable download: files already present in raw_dir are skipped.
    Returns the list of repo paths that failed."""
    log = get_logger()
    todo = [p for p in repo_paths if not (raw_dir / p).exists()]
    log.info(f"{desc}: {len(repo_paths) - len(todo)} cached, {len(todo)} to download")
    failed: list[str] = []
    if not todo:
        return failed

    def _get(p: str) -> str:
        for attempt in range(5):
            try:
                hf_hub_download(repo_id, p, repo_type="dataset", local_dir=str(raw_dir))
                return p
            except Exception as e:  # network hiccups / rate limits -> back off and retry
                if attempt == 4:
                    raise e
                import time
                time.sleep(2 ** attempt * 3)
        return p

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_get, p): p for p in todo}
        for fut in tqdm(as_completed(futures), total=len(futures), desc=desc, mininterval=5):
            try:
                fut.result()
            except Exception as e:  # keep going, report at the end
                failed.append(futures[fut])
                log.warning(f"failed {futures[fut]}: {e}")
    return failed


def fetch_xmls_git(repo_id: str, raw_dir: Path) -> int:
    """Fast path for the ~35k small annotation files.

    The XMLs are plain git blobs (only the images are LFS/xet), and fetching 35k files one
    by one through the Hub's resolve endpoint is throttled for anonymous users (~1k files per
    5 min, i.e. hours). A blobless *sparse* git clone that checks out only the xmls/ folders
    gets them all in one pack in ~10 s. Files are moved into the same layout that
    hf_hub_download uses, so both paths share one cache. Returns the number of files added.
    """
    import shutil
    import subprocess

    tmp = raw_dir / "_git_xml"
    if tmp.exists():
        shutil.rmtree(tmp)
    env = {**__import__("os").environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    url = f"https://huggingface.co/datasets/{repo_id}"
    run = lambda *a, cwd=None: subprocess.run(a, cwd=cwd, env=env, check=True, capture_output=True)
    run("git", "clone", "-q", "--filter=blob:none", "--no-checkout", "--sparse", url, str(tmp))
    run("git", "sparse-checkout", "set", "--no-cone", f"/{DATA_PREFIX}/*/train/annotations/xmls/*.xml", cwd=tmp)
    run("git", "checkout", "-q", "HEAD", cwd=tmp)
    added = 0
    for src in (tmp / DATA_PREFIX).glob("*/train/annotations/xmls/*.xml"):
        dst = raw_dir / src.relative_to(tmp)
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), dst)
            added += 1
    shutil.rmtree(tmp, ignore_errors=True)
    return added


def fetch_split_lists(cfg: dict, paths: Paths) -> list[str]:
    """Download the three split lists and return the de-duplicated, sorted image ids."""
    ids: list[str] = []
    for s in cfg["dataset"]["split_lists"]:
        rp = f"splits/rdd2022_subset_{s}.txt"
        local = paths.raw / rp
        if not local.exists():
            local = Path(hf_hub_download(cfg["dataset"]["repo_id"], rp, repo_type="dataset",
                                     local_dir=str(paths.raw)))
        ids += read_split_list(local)
    return sorted(set(ids))


# --------------------------------------------------------------------------- parse
def parse_voc_xml(xml_text: str | bytes, classes: list[str]) -> dict:
    """Parse one Pascal-VOC annotation. Keeps only boxes whose code is in ``classes``
    (D00/D10/D20/D40); D43, D44, D50, Repair, etc. are ignored."""
    root = ET.fromstring(xml_text)
    size = root.find("size")

    def _num(node, tag, default=0.0):
        el = node.find(tag) if node is not None else None
        try:
            return float(el.text) if el is not None and el.text else default
        except ValueError:
            return default

    width, height = int(_num(size, "width")), int(_num(size, "height"))
    boxes = []
    for obj in root.findall("object"):
        name_el = obj.find("name")  # direct child only (attributes also contain <name>)
        bb = obj.find("bndbox")
        if name_el is None or bb is None or name_el.text is None:
            continue
        code = name_el.text.strip()
        if code not in classes:
            continue
        x0, y0, x1, y1 = (_num(bb, t) for t in ("xmin", "ymin", "xmax", "ymax"))
        if x1 <= x0 or y1 <= y0:
            continue
        boxes.append({"cls": code, "label": classes.index(code),
                      "box": [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]})
    return {"width": width, "height": height, "boxes": boxes}


def build_annotations(cfg: dict, paths: Paths, debug: bool = False) -> list[dict]:
    """Download every XML listed in the split files, parse them, write ann.json/stats.json."""
    log = get_logger()
    classes = cfg["dataset"]["classes"]
    ids = fetch_split_lists(cfg, paths)
    log.info(f"{len(ids)} unique image ids in the split lists")
    if debug:
        rng = random.Random(cfg["seed"])
        # Prefer annotations already in the local cache so a debug run needs few downloads.
        cached = [i for i in ids if (paths.raw / xml_repo_path(i)).exists()]
        pool = cached if len(cached) >= cfg["debug"]["max_images"] else ids
        ids = sorted(rng.sample(pool, min(cfg["debug"]["max_images"], len(pool))))
        log.info(f"debug: using a random subset of {len(ids)} images")

    xml_paths = [xml_repo_path(i) for i in ids]
    missing = sum(not (paths.raw / p).exists() for p in xml_paths)
    if missing and cfg["dataset"].get("xml_download", "git") == "git":
        try:
            log.info(f"{missing} annotation files missing -> sparse git fetch of the xmls/ folders")
            log.info(f"git fetch added {fetch_xmls_git(cfg['dataset']['repo_id'], paths.raw)} files")
        except Exception as e:  # no git / network issue -> fall back to hf_hub_download
            log.warning(f"git fetch failed ({e}); falling back to hf_hub_download")
    # hf_hub_download (parallel, resumable) for anything still missing
    failed = download_files(cfg["dataset"]["repo_id"], xml_paths, paths.raw,
                            cfg["dataset"]["download_workers"], desc="xml")
    if failed:
        log.warning(f"{len(failed)} annotation files could not be downloaded")

    records = []
    for img_id in tqdm(ids, desc="parse", mininterval=5):
        xml_path = paths.raw / xml_repo_path(img_id)
        if not xml_path.exists():
            continue
        try:
            ann = parse_voc_xml(xml_path.read_bytes(), classes)
        except Exception as e:
            log.warning(f"could not parse {xml_path}: {e}")
            continue
        records.append({"id": img_id, "country": img_id.split("/")[0],
                        "file": image_repo_path(img_id), **ann})

    stats = compute_stats(records, classes)
    save_json(records, paths.data / "ann.json")
    save_json(stats, paths.data / "stats.json")
    log.info(f"parsed {stats['n_images']} images; box counts: {stats['overall']}")
    return records


def compute_stats(records: list[dict], classes: list[str]) -> dict:
    overall = Counter()
    per_country: dict[str, Counter] = defaultdict(Counter)
    images_per_country = Counter()
    for r in records:
        images_per_country[r["country"]] += 1
        for b in r["boxes"]:
            overall[b["cls"]] += 1
            per_country[r["country"]][b["cls"]] += 1
    return {
        "n_images": len(records),
        "n_images_with_target_boxes": sum(1 for r in records if r["boxes"]),
        "overall": {c: overall[c] for c in classes},
        "per_country": {k: {c: v[c] for c in classes} for k, v in sorted(per_country.items())},
        "images_per_country": dict(sorted(images_per_country.items())),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)
    cfg = load_config(args.config) if args.config else load_config()
    paths = get_paths(cfg, args.debug).make()
    setup_logging(paths.logs, "data")
    build_annotations(cfg, paths, args.debug)


if __name__ == "__main__":
    main()
