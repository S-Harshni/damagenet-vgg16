"""DamageNet-VGG16 demo: classify a road-damage region with the fine-tuned VGG16.

Run from the project root:   streamlit run app/streamlit_app.py

Two input modes
1. Road photo + box: upload a photo and drag a box around the damage. The app cuts the
   same square context patch as training (side = 1.15 x max(w, h)) and resizes to 128x128.
2. Ready-cut patch: upload an already-cropped patch; it is resized to 128x128.
"""
from __future__ import annotations

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

CKPT = ROOT / "checkpoints" / "vgg16_ft.pt"
RELEASE_URL = "https://github.com/S-Harshni/damagenet-vgg16/releases/download/v1.0/vgg16_ft.pt"

st.set_page_config(page_title="DamageNet-VGG16", page_icon="🛣️", layout="wide")


@st.cache_resource
def load_model():
    ckpt = torch.load(CKPT, map_location="cpu")
    model = VGG16Classifier(ckpt["hidden"], ckpt["dropout"], len(ckpt["classes"]), pretrained=False,
                            input_size=ckpt["input_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def classify(model, patch: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(patch).permute(2, 0, 1).float().div(255).unsqueeze(0)
    with torch.no_grad():
        return torch.softmax(model(x), 1)[0].numpy()


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


st.title("🛣️ DamageNet-VGG16 — road damage classifier")
st.caption("Transfer learning with VGG16 on RDD2022 · 4 classes: D00 longitudinal crack, "
           "D10 transverse crack, D20 alligator crack, D40 pothole · SRM IST 21CSC406J case study")

if not CKPT.exists():
    st.error(f"Model checkpoint not found at `{CKPT.relative_to(ROOT)}`. Train it with "
             f"`python -m src.run_all`, or download it from the GitHub release:\n\n"
             f"`curl -L -o checkpoints/vgg16_ft.pt {RELEASE_URL}`")
    st.stop()

model, ckpt = load_model()
mode = st.radio("Input", ["Road photo + box", "Ready-cut patch"], horizontal=True)

if mode == "Ready-cut patch":
    f = st.file_uploader("Upload a patch (jpg/png)", type=["jpg", "jpeg", "png"])
    if f:
        img = Image.open(f).convert("RGB").resize((ckpt["input_size"],) * 2, Image.BILINEAR)
        patch = np.asarray(img, dtype=np.uint8)
        show_result(patch, classify(model, patch), ckpt)
else:
    f = st.file_uploader("Upload a road photo (jpg/png)", type=["jpg", "jpeg", "png"])
    if f:
        img = Image.open(f).convert("RGB")
        st.write("Drag the box around one damaged region.")
        box = None
        try:
            from streamlit_cropper import st_cropper
            # Show a downscaled copy for interaction, map the box back to full resolution.
            scale = min(1.0, 900 / img.width)
            shown = img.resize((int(img.width * scale), int(img.height * scale))) if scale < 1 else img
            r = st_cropper(shown, realtime_update=True, box_color="#FF6B35", return_type="box")
            box = [r["left"] / scale, r["top"] / scale, (r["left"] + r["width"]) / scale,
                   (r["top"] + r["height"]) / scale]
        except Exception:  # cropper widget unavailable -> numeric box entry
            st.info("Interactive box widget unavailable — enter the box in pixels.")
            st.image(img, use_container_width=True)
            c = st.columns(4)
            x0 = c[0].number_input("xmin", 0, img.width - 2, 0)
            y0 = c[1].number_input("ymin", 0, img.height - 2, 0)
            x1 = c[2].number_input("xmax", 1, img.width, min(img.width, 200))
            y1 = c[3].number_input("ymax", 1, img.height, min(img.height, 200))
            box = [x0, y0, x1, y1]
        if box and box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
            patch = cut_patch(img, box, ckpt["input_size"], ckpt["context"])
            show_result(patch, classify(model, patch), ckpt)
            st.caption(f"Box (px): {[int(v) for v in box]} · square context crop ×{ckpt['context']}")

with st.expander("About"):
    st.markdown("Fine-tuned VGG16 (ImageNet weights, blocks 1–4 frozen, block 5 + new head trained) on "
                "class-balanced 128×128 patches from RDD2022 (Arya et al., CC BY-SA 4.0). "
                "Team: S Harshni, Khushal Narsaria, Aditya Sharma, Harsh Pandey.")
