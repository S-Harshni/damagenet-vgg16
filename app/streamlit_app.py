"""DamageNet-VGG16 demo: find and classify road damage in videos and photos.

Run from the project root:   streamlit run app/streamlit_app.py

Input modes
1. Road video (automatic): a Faster R-CNN detector (src/detect.py) finds damage in the video,
   draws a box with class + confidence on every frame and tracks each damage so it is counted
   once. The fine-tuned VGG16 classifier re-checks the best view of every tracked damage.
2. Road photo: the same detector boxes damage in a photo; you can also drag your own box and
   classify it with VGG16.
3. Ready-cut patch: classify an already-cropped patch with VGG16.

VGG16 is a patch classifier (it needs a box and has no "no damage" class); the detector is what
finds the damage. For a box, the app cuts the same square context patch as training
(side = 1.15 x max(w, h)) and resizes it to 128x128.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import VGG16Classifier  # noqa: E402
from src.patches import cut_patch  # noqa: E402
from src.video import annotate_video, draw_detections, track_label  # noqa: E402

CKPT = ROOT / "checkpoints" / "vgg16_ft.pt"
DET_CKPT = ROOT / "checkpoints" / "detector.pt"
RELEASE = "https://github.com/S-Harshni/damagenet-vgg16/releases/download"
MIN_HITS = 2  # a tracked damage must be detected on >= 2 detection steps to be counted

st.set_page_config(page_title="DamageNet-VGG16", page_icon="🛣️", layout="wide")


def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@st.cache_resource
def load_model():
    ckpt = torch.load(CKPT, map_location="cpu")
    model = VGG16Classifier(ckpt["hidden"], ckpt["dropout"], len(ckpt["classes"]), pretrained=False,
                            input_size=ckpt["input_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_resource
def load_det():
    from src.detect import load_detector
    model, ck = load_detector(DET_CKPT, device(), score_thresh=0.05)
    return model, ck


def make_detect_fn(det, thresh: float, keep: list[int], tta: bool = True):
    """Wrap the detector: list of PIL images -> list of (boxes, 0-based labels, scores)."""
    from src.detect import detect
    dev = next(det.parameters()).device

    @torch.no_grad()
    def fn(imgs):
        xs = [torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float().div(255).to(dev)
              for im in imgs]
        res = []
        for o in detect(det, xs, tta):
            b, s = o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()
            lab = o["labels"].cpu().numpy() - 1  # detector label 0 is background
            k = (s >= thresh) & np.isin(lab, keep)
            res.append((b[k], lab[k], s[k]))
        return res
    return fn


def classify(model, patch: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(patch).permute(2, 0, 1).float().div(255).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(x), 1)[0].numpy()


@torch.no_grad()
def classify_batch(model, patches: list[np.ndarray], batch_size: int = 32) -> np.ndarray:
    x = torch.from_numpy(np.stack(patches)).permute(0, 3, 1, 2).float().div(255)
    return torch.cat([torch.softmax(model(x[i:i + batch_size]), 1)
                      for i in range(0, len(x), batch_size)]).numpy()


def pick_box(img: Image.Image, key: str) -> list[float] | None:
    """Let the user drag a box on `img`; returns [xmin, ymin, xmax, ymax] in full-res pixels."""
    try:
        from streamlit_cropper import st_cropper
        scale = min(1.0, 900 / img.width)  # interact on a downscaled copy, map back
        shown = img.resize((int(img.width * scale), int(img.height * scale))) if scale < 1 else img
        r = st_cropper(shown, realtime_update=True, box_color="#FF6B35", return_type="box", key=key)
        return [r["left"] / scale, r["top"] / scale, (r["left"] + r["width"]) / scale,
                (r["top"] + r["height"]) / scale]
    except Exception:  # cropper widget unavailable -> numeric box entry
        st.info("Interactive box widget unavailable — enter the box in pixels.")
        st.image(img, use_container_width=True)
        c = st.columns(4)
        x0 = c[0].number_input("xmin", 0, img.width - 2, 0, key=f"{key}_x0")
        y0 = c[1].number_input("ymin", 0, img.height - 2, 0, key=f"{key}_y0")
        x1 = c[2].number_input("xmax", 1, img.width, min(img.width, 200), key=f"{key}_x1")
        y1 = c[3].number_input("ymax", 1, img.height, min(img.height, 200), key=f"{key}_y1")
        return [x0, y0, x1, y1]


def box_ok(box) -> bool:
    return bool(box) and box[2] - box[0] >= 4 and box[3] - box[1] >= 4


def show_result(patch: np.ndarray, probs: np.ndarray, ckpt: dict):
    classes, names = ckpt["classes"], ckpt["class_names"]
    k = int(probs.argmax())
    c1, c2 = st.columns([1, 2])
    with c1:
        st.image(patch, caption="128×128 model input", width=256)
    with c2:
        st.subheader(f"Prediction: {classes[k]} — {names[k]}")
        st.metric("Confidence", f"{probs[k]:.1%}")
        for c, n, p in sorted(zip(classes, names, probs), key=lambda t: -t[2]):
            st.progress(float(p), text=f"{c} {n}: {p:.1%}")


def detector_settings(ckpt: dict, key: str):
    classes, names = ckpt["classes"], ckpt["class_names"]
    c1, c2, c3 = st.columns([2, 3, 1.4])
    thresh = c1.slider("Minimum confidence", 0.10, 0.95, 0.40, 0.05, key=f"{key}_thr",
                       help="Boxes below this detector score are hidden. 0.40 gave the best F1 on the "
                            "validation set.")
    keep = c2.multiselect("Damage types to show", classes, default=classes, key=f"{key}_cls",
                          format_func=lambda c: f"{c} {names[classes.index(c)]}")
    tta = c3.toggle("Flip check", value=True, key=f"{key}_tta",
                    help="Also detect on the mirrored frame and merge the boxes: about 1 mAP point "
                         "more accurate on the validation set, 2x slower.")
    return thresh, [classes.index(c) for c in keep], tta


def missing(path: Path, how: str, tag: str) -> bool:
    if path.exists():
        return False
    st.error(f"Model file `{path.relative_to(ROOT)}` not found. {how}, or download it from the "
             f"GitHub release:\n\n`curl -L -o {path.relative_to(ROOT)} {RELEASE}/{tag}/{path.name}`")
    return True


# ---------------------------------------------------------------------------- page
st.title("🛣️ DamageNet-VGG16 — road damage finder")
st.caption("RDD2022 · 4 classes: D00 longitudinal crack, D10 transverse crack, D20 alligator crack, "
           "D40 pothole · SRM IST 21CSC406J case study")

if missing(CKPT, "Train it with `python -m src.run_all`", "v1.0"):
    st.stop()
model, ckpt = load_model()
classes, names = ckpt["classes"], ckpt["class_names"]
mode = st.radio("Input", ["Road video", "Road photo", "Ready-cut patch"], horizontal=True)

if mode == "Road video":
    f = st.file_uploader("Upload a road video (mp4/mov/avi/mkv/webm, first 2 minutes are used)",
                         type=["mp4", "mov", "m4v", "avi", "mkv", "webm"])
    if f and not missing(DET_CKPT, "Train it with `python -m src.detect`", "v1.1"):
        det, dck = load_det()
        thresh, keep, tta = detector_settings(dck, "vid")
        every = st.select_slider("Run the detector on", options=[1, 2, 3, 5], value=2,
                                 format_func=lambda v: "every frame" if v == 1 else f"every {v}th frame"
                                 if v > 3 else f"every {v}{'nd' if v == 2 else 'rd'} frame",
                                 help="Boxes are drawn on every frame; in-between frames reuse the last "
                                      "detections. Fewer detector runs = faster processing.")
        data = f.getvalue()
        run_key = (hashlib.md5(data).hexdigest(), thresh, tuple(keep), every, tta)
        if st.button("Find damage", type="primary") or st.session_state.get("vid_key") == run_key:
            if st.session_state.get("vid_key") != run_key:
                bar = st.progress(0.0, text="Finding damage…")
                vid, log, tracks = annotate_video(
                    data, make_detect_fn(det, thresh, keep, tta), classes, names, detect_every=every,
                    progress=lambda p, n: bar.progress(p, text=f"Finding damage… frame {n}"))
                bar.empty()
                st.session_state.update(vid_key=run_key, vid=(vid, log, tracks))
            vid, log, tracks = st.session_state["vid"]
            st.video(vid)
            st.download_button("Download annotated video", vid, file_name="damage_detections.mp4",
                               mime="video/mp4")

            # One row per tracked damage (seen on >= MIN_HITS detector runs)
            rows = [(tid, tr) for tid, tr in tracks.items() if tr["hits"] >= MIN_HITS or len(tracks) == 1]
            labels = {tid: track_label(tr) for tid, tr in rows}
            cols = st.columns(len(classes))
            for j, c in enumerate(classes):
                cols[j].metric(f"{c} {names[j]}", sum(labels[t] == j for t, _ in rows))
            st.caption(f"Distinct damages tracked across the video (seen on at least {MIN_HITS} detector "
                       f"runs) · {len(log)} boxes drawn in total.")
            if rows:
                P = classify_batch(model, [tr["patch"] for _, tr in rows])
                table = [{"ID": f"#{tid}", "damage": f"{classes[labels[tid]]} {names[labels[tid]]}",
                          "best score": tr["best"], "first seen (s)": round(tr["first"], 2),
                          "last seen (s)": round(tr["last"], 2), "detections": tr["hits"],
                          "VGG16 check": f"{classes[int(p.argmax())]} {p.max():.0%}"}
                         for (tid, tr), p in zip(rows, P)]
                st.dataframe(table, hide_index=True, use_container_width=True, column_config={
                    "best score": st.column_config.ProgressColumn("best score", min_value=0.0, max_value=1.0,
                                                                  format="%.2f")})
                st.markdown("**Best view of each damage**")
                order = sorted(range(len(rows)), key=lambda i: -rows[i][1]["best"])[:24]
                g = st.columns(6)
                for n, i in enumerate(order):
                    tid, tr = rows[i]
                    g[n % 6].image(tr["patch"], use_container_width=True,
                                   caption=f"#{tid} {classes[labels[tid]]} {tr['best']:.2f}")
            else:
                st.info("No damage found above this confidence. Try lowering the minimum confidence.")

elif mode == "Road photo":
    f = st.file_uploader("Upload a road photo (jpg/png)", type=["jpg", "jpeg", "png"])
    if f:
        img = Image.open(f).convert("RGB")
        if not missing(DET_CKPT, "Train it with `python -m src.detect`", "v1.1"):
            det, dck = load_det()
            thresh, keep, tta = detector_settings(dck, "img")
            b, lab, s = make_detect_fn(det, thresh, keep, tta)([img])[0]
            dets = [{"box": bb.tolist(), "label": int(ll), "score": float(ss), "id": i + 1}
                    for i, (bb, ll, ss) in enumerate(zip(b, lab, s))]
            st.image(draw_detections(img.copy(), dets, classes, names), use_container_width=True,
                     caption=f"{len(dets)} damage box(es) found")
            if dets:
                P = classify_batch(model, [cut_patch(img, d["box"], ckpt["input_size"], ckpt["context"])
                                           for d in dets])
                st.dataframe([{"ID": f"#{d['id']}", "damage": f"{classes[d['label']]} {names[d['label']]}",
                               "score": d["score"], "VGG16 check": f"{classes[int(p.argmax())]} {p.max():.0%}",
                               "box (px)": [round(v) for v in d["box"]]} for d, p in zip(dets, P)],
                             hide_index=True, use_container_width=True, column_config={
                                 "score": st.column_config.ProgressColumn("score", min_value=0.0,
                                                                          max_value=1.0, format="%.2f")})
        with st.expander("Classify your own box with VGG16"):
            box = pick_box(img, key="photo_box")
            if box_ok(box):
                patch = cut_patch(img, box, ckpt["input_size"], ckpt["context"])
                show_result(patch, classify(model, patch), ckpt)
                st.caption(f"Box (px): {[int(v) for v in box]} · square context crop ×{ckpt['context']}")

else:
    f = st.file_uploader("Upload a patch (jpg/png)", type=["jpg", "jpeg", "png"])
    if f:
        img = Image.open(f).convert("RGB").resize((ckpt["input_size"],) * 2, Image.BILINEAR)
        patch = np.asarray(img, dtype=np.uint8)
        show_result(patch, classify(model, patch), ckpt)

with st.expander("About"):
    st.markdown("**Detector:** Faster R-CNN (MobileNetV3-Large FPN, COCO-pretrained) fine-tuned on 14.8k "
                "RDD2022 images; it finds the damage (held-out test: mAP@0.5 0.355, F1 0.40 with the flip check). **Classifier:** VGG16 (ImageNet weights, block 5 + new head "
                "fine-tuned) on 128×128 patches; it re-checks each found damage. Data: RDD2022 (Arya et al., "
                "CC BY-SA 4.0). Team: S Harshni, Khushal Narsaria, Aditya Sharma, Harsh Pandey.")
