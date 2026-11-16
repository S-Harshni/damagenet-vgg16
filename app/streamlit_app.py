"""DamageNet: find and classify road damage in dashcam videos and photos.

Run from the project root:   streamlit run app/streamlit_app.py
(the theme lives in .streamlit/config.toml, which Streamlit reads from the working directory)

Modes
1. Video: a Faster R-CNN detector (src/detect.py) finds damage, a tracker links it across frames
   and smooths the boxes (src/video.py), and every frame is drawn with class + confidence. Each
   tracked damage is counted once; the fine-tuned VGG16 classifier re-checks its best view.
2. Photo: the same detector boxes damage in a photo; you can also drag your own box for VGG16.
3. Patch: classify an already-cropped patch with VGG16.

VGG16 is a patch classifier (it needs a box and has no "no damage" class); the detector is what
finds the damage. For a box, the app cuts the same square context patch as training
(side = 1.15 x max(w, h)) and resizes it to 128x128.
"""
from __future__ import annotations

import base64
import hashlib
import html
import json
import sys
import time
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.models import VGG16Classifier  # noqa: E402
from src.patches import cut_patch  # noqa: E402
from src.video import CLASS_COLORS, annotate_video, draw_detections, merge_overlaps  # noqa: E402

CKPT = ROOT / "checkpoints" / "vgg16_ft.pt"
DET_CKPT = ROOT / "checkpoints" / "detector.pt"
RELEASE = "https://github.com/S-Harshni/damagenet-vgg16/releases/download"
REPO = "https://github.com/S-Harshni/damagenet-vgg16"
SAMPLES = [ROOT / "data" / "stability" / "test_drive.mp4", ROOT / "data" / "demo_drive.mp4"]
MERGE_IOU = 0.7  # overlapping boxes of different classes -> keep the best (costs < 0.1 F1 point on val)

# Accuracy presets (chosen on the validation set, see README "Automatic damage detection")
PRESETS = {
    "Fast": {"tta": False, "scales": None, "help": "One detector pass per frame."},
    "Balanced": {"tta": True, "scales": None, "help": "Adds a mirrored pass (2 passes per frame)."},
    "Best": {"tta": True, "scales": (576, 704), "help": "Adds a mirrored pass and a larger image size (4 passes per frame)."},
}
DEFAULT_PRESET = "Best"

st.set_page_config(page_title="DamageNet · road damage finder", page_icon="🛣️", layout="wide",
                   initial_sidebar_state="expanded")


def hexc(c: str) -> str:
    return "#%02X%02X%02X" % CLASS_COLORS[c]


# ---------------------------------------------------------------------------- style
st.html("""
<style>
:root { --card:#13161B; --card2:#171A20; --line:#252930; --muted:#9BA1A9; --faint:#6B717A;
        --accent:#FF7A3D; --ink:#ECEAE5; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1320px; }
header[data-testid="stHeader"] { background: transparent; }
h1, h2, h3 { letter-spacing: -0.02em; }

/* hero */
.hero { position:relative; overflow:hidden; border:1px solid var(--line); border-radius:20px;
        padding:28px 32px 26px; margin-bottom:18px;
        background: radial-gradient(1200px 300px at 85% -40%, rgba(255,122,61,.20), transparent 60%),
                    radial-gradient(800px 260px at -10% 120%, rgba(79,195,247,.10), transparent 60%),
                    linear-gradient(180deg, #14171C, #101216); }
.hero::after { content:""; position:absolute; right:-40px; top:0; bottom:0; width:340px; opacity:.07;
        background: repeating-linear-gradient(90deg, transparent 0 22px, #fff 22px 30px);
        transform: skewX(-24deg); pointer-events:none; }
.eyebrow { font: 600 .72rem/1 Inter, sans-serif; letter-spacing:.14em; text-transform:uppercase;
           color: var(--accent); display:flex; gap:8px; align-items:center; }
.eyebrow .dot { width:7px; height:7px; border-radius:50%; background:var(--accent);
                box-shadow:0 0 0 4px rgba(255,122,61,.18); }
.hero h1 { font-family:'Space Grotesk', Inter, sans-serif; font-size:2.35rem; line-height:1.08;
           margin:.55rem 0 .45rem; padding:0; color:var(--ink); }
.hero p { color:var(--muted); font-size:1.0rem; max-width:720px; margin:0; }
.stats { display:flex; gap:10px; flex-wrap:wrap; margin-top:18px; }
.stat { border:1px solid var(--line); background:rgba(255,255,255,.02); border-radius:12px;
        padding:9px 14px; min-width:120px; }
.stat b { display:block; font:600 1.15rem/1.2 'Space Grotesk', Inter, sans-serif; color:var(--ink); }
.stat span { font-size:.74rem; color:var(--faint); text-transform:uppercase; letter-spacing:.08em; }

/* class legend chips */
.chips { display:flex; gap:8px; flex-wrap:wrap; margin:2px 0 6px; }
.chip { display:inline-flex; align-items:center; gap:7px; padding:5px 11px 5px 9px; border-radius:999px;
        border:1px solid var(--line); background:var(--card); font-size:.82rem; color:var(--ink); }
.chip i { width:9px; height:9px; border-radius:3px; display:inline-block; }
.chip em { font-style:normal; color:var(--faint); font-family:'JetBrains Mono', monospace; font-size:.74rem; }

/* cards */
.card { border:1px solid var(--line); background:var(--card); border-radius:16px; padding:18px 20px; }
.card h4 { margin:0 0 2px; font:600 .78rem/1.2 Inter, sans-serif; color:var(--faint);
           text-transform:uppercase; letter-spacing:.1em; }
.big { font:700 3rem/1 'Space Grotesk', Inter, sans-serif; color:var(--ink); margin:6px 0 2px; }
.sub { color:var(--muted); font-size:.85rem; }
.row { display:grid; grid-template-columns: 14px 1fr auto; gap:10px; align-items:center; margin-top:12px; }
.row i { width:10px; height:10px; border-radius:3px; }
.row .n { font:600 1rem 'JetBrains Mono', monospace; color:var(--ink); }
.row .lbl { font-size:.9rem; color:var(--ink); }
.bar { grid-column: 2 / 4; height:6px; border-radius:6px; background:#20242B; overflow:hidden; margin-top:-4px; }
.bar > div { height:100%; border-radius:6px; }
.meta { display:grid; grid-template-columns:1fr 1fr; gap:10px 16px; margin-top:16px; padding-top:14px;
        border-top:1px solid var(--line); }
.meta div span { display:block; font-size:.7rem; color:var(--faint); text-transform:uppercase; letter-spacing:.08em; }
.meta div b { font:500 .92rem 'JetBrains Mono', monospace; color:var(--ink); }

/* detection list (photo) and gallery captions */
.det { display:flex; gap:12px; align-items:center; padding:10px 12px; border:1px solid var(--line);
       border-radius:12px; background:var(--card); margin-bottom:8px; }
.det .sw { width:4px; align-self:stretch; border-radius:4px; }
.det .t { flex:1; } .det .t b { display:block; font-size:.92rem; color:var(--ink); }
.det .t span { font-size:.78rem; color:var(--muted); }
.det .s { font:600 .95rem 'JetBrains Mono', monospace; color:var(--ink); }
.gcap { display:flex; justify-content:space-between; align-items:center; gap:6px; margin:6px 2px 2px; }
.gcap b { font-size:.82rem; color:var(--ink); white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.gcap span { font:500 .78rem 'JetBrains Mono', monospace; color:var(--muted); }
.tag { display:inline-block; padding:2px 7px; border-radius:6px; font:700 .74rem 'JetBrains Mono', monospace;
       color:#14161A !important; line-height:1.3; vertical-align:middle; }
.gcap b .tag { margin-right:4px; }

/* empty state */
.empty { border:1.5px dashed #2C3139; border-radius:18px; padding:34px 24px; text-align:center;
         background: linear-gradient(180deg, rgba(255,255,255,.015), transparent); }
.empty .ic { font-size:2rem; } .empty b { display:block; font-size:1.05rem; margin:8px 0 4px; color:var(--ink); }
.empty span { color:var(--muted); font-size:.9rem; }
.how { display:grid; grid-template-columns:repeat(3, 1fr); gap:12px; margin-top:14px; }
.how div { border:1px solid var(--line); border-radius:14px; padding:14px 16px; background:var(--card); }
.how b { display:block; font-size:.92rem; color:var(--ink); margin-bottom:3px; }
.how span { color:var(--muted); font-size:.83rem; }
.how em { font-style:normal; font:600 .72rem 'JetBrains Mono', monospace; color:var(--accent); }
@media (max-width: 760px) { .how { grid-template-columns:1fr; } .hero h1 { font-size:1.8rem; } }

/* sidebar brand */
.brand { display:flex; gap:11px; align-items:center; margin:-6px 0 6px; }
.brand b { font:700 1.15rem 'Space Grotesk', Inter, sans-serif; color:var(--ink); display:block; }
.brand span { font-size:.76rem; color:var(--faint); }
.side-h { font:600 .7rem Inter, sans-serif; letter-spacing:.12em; text-transform:uppercase; color:var(--faint);
          margin:14px 0 4px; }
.foot { font-size:.76rem; color:var(--faint); line-height:1.5; }
.foot a { color:var(--muted); }

/* widgets polish */
[data-testid="stFileUploaderDropzone"] { border:1.5px dashed #2C3139; background:var(--card); border-radius:16px;
                                          padding:22px; }
[data-testid="stVideo"], [data-testid="stImage"] img { border-radius:14px; }
[data-testid="stImage"] img { border:1px solid var(--line); }
div[data-testid="stExpander"] details { background:var(--card); border-radius:14px; }
[data-testid="stMetric"] { background:var(--card); border:1px solid var(--line); border-radius:14px; padding:12px 16px; }
</style>
""")

LOGO = """<svg width="38" height="38" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="g" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#FF9A5C"/>
<stop offset="1" stop-color="#FF5A2A"/></linearGradient></defs>
<rect width="40" height="40" rx="11" fill="url(#g)"/>
<path d="M14 33 L18.5 7 M26 33 L21.5 7" stroke="#14161A" stroke-width="2.6" stroke-linecap="round"/>
<path d="M20 30v-3.5M20 21.5v-3M20 13.5v-2.5" stroke="#14161A" stroke-width="2" stroke-linecap="round"/>
<rect x="23.5" y="21" width="9" height="7" rx="2" fill="none" stroke="#fff" stroke-width="1.8"/>
</svg>"""


# ---------------------------------------------------------------------------- models
def device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


@st.cache_resource(show_spinner="Loading VGG16…")
def load_model():
    ckpt = torch.load(CKPT, map_location="cpu")
    model = VGG16Classifier(ckpt["hidden"], ckpt["dropout"], len(ckpt["classes"]), pretrained=False,
                            input_size=ckpt["input_size"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


@st.cache_resource(show_spinner="Loading the detector…")
def load_det():
    from src.detect import load_detector
    return load_detector(DET_CKPT, device(), score_thresh=0.05)


@st.cache_data(show_spinner=False)
def test_scores() -> dict:
    out = {}
    for k, f in (("best", "detector_v2_test_tta_ms.json"), ("balanced", "detector_v2_test_tta.json")):
        p = ROOT / "results" / f
        if p.exists():
            r = json.loads(p.read_text())
            out[k] = (r["mAP50"], r["best_f1"]["f1"])
    return out


def make_detect_fn(det, keep: list[int], preset: str, floor: float = 0.05):
    """list of PIL images -> list of (boxes, 0-based labels, scores) for the kept classes."""
    from src.detect import detect
    dev = next(det.parameters()).device
    cfg = PRESETS[preset]

    @torch.no_grad()
    def fn(imgs):
        xs = [torch.from_numpy(np.asarray(im, dtype=np.uint8).copy()).permute(2, 0, 1).float().div(255).to(dev)
              for im in imgs]
        res = []
        for o in detect(det, xs, cfg["tta"], scales=cfg["scales"]):
            b, s = o["boxes"].cpu().numpy(), o["scores"].cpu().numpy()
            lab = o["labels"].cpu().numpy() - 1  # detector label 0 is background
            k = (s >= floor) & np.isin(lab, keep)
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


# ---------------------------------------------------------------------------- small UI helpers
def mmss(t: float) -> str:
    return f"{int(t // 60)}:{int(t % 60):02d}"


def nice(name: str) -> str:
    return name[:1].upper() + name[1:]


def chips(classes, names, keep=None) -> str:
    return '<div class="chips">' + "".join(
        f'<span class="chip" style="opacity:{1 if keep is None or j in keep else .35}">'
        f'<i style="background:{hexc(c)}"></i>{nice(n)} <em>{c}</em></span>'
        for j, (c, n) in enumerate(zip(classes, names))) + "</div>"


def empty_state(icon: str, title: str, hint: str):
    st.html(f'<div class="empty"><div class="ic">{icon}</div><b>{title}</b><span>{hint}</span></div>')


def summary_card(counts: list[int], classes, names, meta: list[tuple[str, str]], title="Damage found"):
    total, top = sum(counts), max(1, max(counts) if counts else 1)
    rows = "".join(
        f'<div class="row"><i style="background:{hexc(c)}"></i><span class="lbl">{nice(n)}</span>'
        f'<span class="n">{k}</span><div class="bar"><div style="width:{100 * k / top:.0f}%;'
        f'background:{hexc(c)}"></div></div></div>'
        for c, n, k in zip(classes, names, counts))
    m = "".join(f"<div><span>{html.escape(a)}</span><b>{html.escape(b)}</b></div>" for a, b in meta)
    st.html(f'<div class="card"><h4>{title}</h4><div class="big">{total}</div>'
            f'<div class="sub">distinct damages, each counted once</div>{rows}<div class="meta">{m}</div></div>')


def missing(path: Path, how: str, tag: str) -> bool:
    if path.exists():
        return False
    st.error(f"Model file `{path.relative_to(ROOT)}` not found. {how}, or download it from the "
             f"GitHub release:\n\n`curl -L -o {path.relative_to(ROOT)} {RELEASE}/{tag}/{path.name}`")
    return True


def pick_box(img: Image.Image, key: str) -> list[float] | None:
    """Let the user drag a box on `img`; returns [xmin, ymin, xmax, ymax] in full-res pixels."""
    try:
        from streamlit_cropper import st_cropper
        scale = min(1.0, 900 / img.width)  # interact on a downscaled copy, map back
        shown = img.resize((int(img.width * scale), int(img.height * scale))) if scale < 1 else img
        r = st_cropper(shown, realtime_update=True, box_color="#FF7A3D", return_type="box", key=key)
        return [r["left"] / scale, r["top"] / scale, (r["left"] + r["width"]) / scale,
                (r["top"] + r["height"]) / scale]
    except Exception:  # cropper widget unavailable -> numeric box entry
        st.info("Interactive box widget unavailable — enter the box in pixels.")
        st.image(img, width="stretch")
        c = st.columns(4)
        x0 = c[0].number_input("xmin", 0, img.width - 2, 0, key=f"{key}_x0")
        y0 = c[1].number_input("ymin", 0, img.height - 2, 0, key=f"{key}_y0")
        x1 = c[2].number_input("xmax", 1, img.width, min(img.width, 200), key=f"{key}_x1")
        y1 = c[3].number_input("ymax", 1, img.height, min(img.height, 200), key=f"{key}_y1")
        return [x0, y0, x1, y1]


def show_vgg(patch: np.ndarray, probs: np.ndarray, ckpt: dict):
    classes, names = ckpt["classes"], ckpt["class_names"]
    k = int(probs.argmax())
    c1, c2 = st.columns([1, 2], gap="large")
    with c1:
        st.image(patch, caption="128×128 model input", width=230)
    with c2:
        st.html(f'<div class="card"><h4>VGG16 prediction</h4><div class="big" style="font-size:2.1rem">'
                f'{nice(names[k])}</div><div class="sub"><span class="tag" style="background:{hexc(classes[k])}">'
                f'{classes[k]}</span>&nbsp; {probs[k]:.1%} confident</div></div>')
        for c, n, p in sorted(zip(classes, names, probs), key=lambda t: -t[2]):
            st.progress(float(p), text=f"{nice(n)} · {p:.1%}")


def timeline_chart(damages: list[dict], classes, names, duration: float):
    import altair as alt
    import pandas as pd
    df = pd.DataFrame([{"ID": f"#{d['id']}", "order": d["id"], "Damage": nice(names[d["label"]]),
                        "start": d["first"], "end": max(d["last"], d["first"] + 0.12),
                        "Confidence": round(d["conf"], 2), "Seen": f"{d['first']:.1f}s – {d['last']:.1f}s"}
                       for d in damages])
    dom = [nice(n) for n in names]
    rng = [hexc(c) for c in classes]
    h = max(120, min(420, 22 * len(df) + 40))
    ch = alt.Chart(df).mark_bar(cornerRadius=4, height=12).encode(
        x=alt.X("start:Q", title="Time in video (s)", scale=alt.Scale(domain=[0, max(duration, 0.5)])),
        x2="end:Q",
        y=alt.Y("ID:N", sort=alt.EncodingSortField("order"), title=None,
                axis=alt.Axis(labels=len(df) <= 30, ticks=False)),
        color=alt.Color("Damage:N", scale=alt.Scale(domain=dom, range=rng),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["ID", "Damage", "Confidence", "Seen"],
    ).properties(height=h)
    st.altair_chart(ch, width="stretch")


# ---------------------------------------------------------------------------- page
if missing(CKPT, "Train it with `python -m src.run_all`", "v1.0"):
    st.stop()
model, ckpt = load_model()
classes, names = ckpt["classes"], ckpt["class_names"]
scores = test_scores()

# ---- sidebar: settings
with st.sidebar:
    st.html(f'<div class="brand"><img src="data:image/svg+xml;base64,{base64.b64encode(LOGO.encode()).decode()}" width="38" height="38"><div><b>DamageNet</b><span>Road damage finder</span></div></div>')
    st.html('<div class="side-h">Detection</div>')
    preset = st.segmented_control("Accuracy", list(PRESETS), default=DEFAULT_PRESET, key="preset",
                                  help="Best runs the detector on the frame, its mirror image and a larger "
                                       "copy, then merges the boxes: most accurate on held-out images, ~4x "
                                       "slower than Fast.") or DEFAULT_PRESET
    st.caption(PRESETS[preset]["help"])
    thresh = st.slider("Minimum confidence", 0.20, 0.90, 0.50 if preset == "Best" else 0.40, 0.05, key=f"thr_{preset}",
                       help="Show a damage when the detector is at least this sure of it. In videos this is "
                            "the average of its 3 best frames. Default = the best F1 on the validation set.")
    keep_names = st.pills("Damage types", [nice(n) for n in names], default=[nice(n) for n in names],
                          selection_mode="multi", key="types")
    keep = [j for j, n in enumerate(names) if nice(n) in (keep_names or [])]
    st.html('<div class="side-h">Video</div>')
    every = st.select_slider("Check every", options=[1, 2, 3, 4], value=2, key="every",
                             format_func=lambda v: "frame" if v == 1 else f"{v} frames",
                             help="How often the detector runs. Boxes are drawn on every frame either way "
                                  "(in between they are interpolated). Fewer runs = faster.")
    steady = st.toggle("Steady boxes", value=True, key="steady",
                       help="Track each damage over time and smooth its box, label and score. Off shows the "
                            "raw detector output frame by frame (flickers).")
    st.divider()
    sc = scores.get("best") or scores.get("balanced")
    st.html('<div class="foot">Faster R-CNN (MobileNetV3-FPN) finds damage; a fine-tuned VGG16 re-checks it. '
            + (f"Held-out test: mAP@0.5 <b>{sc[0]:.3f}</b>, F1 <b>{sc[1]:.2f}</b>. " if sc else "")
            + f'Trained on RDD2022 (CC BY-SA 4.0).<br><a href="{REPO}">GitHub ↗</a> · runs on '
            f'<b>{device().upper()}</b></div>')

# ---- hero
sc = scores.get("best") or scores.get("balanced")
st.html(f"""
<div class="hero">
  <div class="eyebrow"><span class="dot"></span>Road inspection · computer vision</div>
  <h1>Find road damage in dashcam footage</h1>
  <p>Upload a drive and DamageNet boxes every crack and pothole, follows each one across frames and counts it once,
     with a confidence score you can trust.</p>
  <div class="stats">
    <div class="stat"><b>4</b><span>damage types</span></div>
    {f'<div class="stat"><b>{sc[0]:.3f}</b><span>test mAP@0.5</span></div>' if sc else ''}
    {f'<div class="stat"><b>{sc[1]:.2f}</b><span>test F1</span></div>' if sc else ''}
    <div class="stat"><b>14.8k</b><span>training images</span></div>
  </div>
</div>""")

mode = st.segmented_control("Mode", [":material/videocam: Video", ":material/photo_camera: Photo",
                                     ":material/crop: Patch"], default=":material/videocam: Video",
                            key="mode", label_visibility="collapsed") or ":material/videocam: Video"
st.html(chips(classes, names, keep))

if not keep:
    st.warning("Pick at least one damage type in the sidebar.")
    st.stop()

# ============================================================================ VIDEO
if mode.endswith("Video"):
    if missing(DET_CKPT, "Train it with `python -m src.detect`", "v1.1"):
        st.stop()
    up_col, sample_col = st.columns([4, 1], vertical_alignment="center")
    f = up_col.file_uploader("Upload a road video", type=["mp4", "mov", "m4v", "avi", "mkv", "webm"],
                             label_visibility="collapsed",
                             help="mp4 / mov / avi / mkv / webm · the first 2 minutes are analysed")
    sample = next((p for p in SAMPLES if p.exists()), None)
    if sample and sample_col.button("Try a sample drive", icon=":material/play_circle:", width="stretch"):
        st.session_state["use_sample"] = True
    if f is not None:
        st.session_state["use_sample"] = False
    data, src_name = (f.getvalue(), f.name) if f else \
        (sample.read_bytes(), "sample drive") if sample and st.session_state.get("use_sample") else (None, None)

    if data is None:
        st.html("""<div class="how">
          <div><em>01</em><b>Upload a drive</b><span>Any phone or dashcam clip, filmed facing the road.</span></div>
          <div><em>02</em><b>Find damage</b><span>Every crack and pothole is boxed, tracked and scored.</span></div>
          <div><em>03</em><b>Review &amp; export</b><span>Jump to each damage, download the annotated video.</span></div>
        </div>""")
        st.stop()

    det, dck = load_det()
    run_key = (hashlib.md5(data).hexdigest(), thresh, tuple(keep), every, steady, preset)
    cached = st.session_state.get("vid_key") == run_key
    c1, c2 = st.columns([1, 3], vertical_alignment="center")
    go = c1.button("Find damage", type="primary", icon=":material/search:", width="stretch", disabled=cached)
    c2.caption(f"**{html.escape(src_name)}** · {len(data) / 1e6:.1f} MB · {preset} accuracy · "
               f"detector every {'frame' if every == 1 else f'{every} frames'}"
               + (" · settings changed, run again" if st.session_state.get("vid_key") and not cached else ""))
    if go:
        bar = st.progress(0.0, text="Starting…")
        t0 = time.time()
        vid, log, damages = annotate_video(
            data, make_detect_fn(det, keep, preset, floor=max(0.05, thresh / 2)), classes, names, thresh=thresh,
            detect_every=every, stabilize=steady, merge_iou=MERGE_IOU,
            progress=lambda p, txt: bar.progress(min(1.0, p), text=txt))
        bar.empty()
        st.session_state.update(vid_key=run_key, vid=(vid, log, damages, time.time() - t0), start=0)
        cached = True
    if not cached:
        st.stop()

    vid, log, damages, took = st.session_state["vid"]
    import av, io  # noqa: E401
    with av.open(io.BytesIO(vid)) as c:
        vs = c.streams.video[0]
        dur = float(c.duration / av.time_base) if c.duration else 0.0
        size = f"{vs.width}×{vs.height}"
    left, right = st.columns([2.3, 1], gap="large")
    with left:
        st.video(vid, start_time=int(st.session_state.get("start", 0)))
    with right:
        counts = [sum(d["label"] == j for d in damages) for j in range(len(classes))]
        summary_card(counts, classes, names, [("Video", mmss(dur)), ("Resolution", size),
                                              ("Processed in", f"{took:.0f} s"), ("Boxes drawn", f"{len(log):,}")])
        st.download_button("Download annotated video", vid, file_name="damagenet_detections.mp4",
                           mime="video/mp4", icon=":material/download:", width="stretch")

    if not damages:
        st.info("No damage reached this confidence. Lower **Minimum confidence** in the sidebar and run again.")
        st.stop()

    P = classify_batch(model, [d["patch"] for d in damages])
    st.subheader("Damage timeline")
    st.caption("When each tracked damage is on screen. Hover a bar for details.")
    timeline_chart(damages, classes, names, dur)

    tab_g, tab_t = st.tabs([":material/grid_view: Gallery", ":material/table_rows: Table"])
    with tab_g:
        order = sorted(range(len(damages)), key=lambda i: -damages[i]["conf"])[:24]
        cols = st.columns(4)
        for n, i in enumerate(order):
            d = damages[i]
            c = classes[d["label"]]
            with cols[n % 4]:
                st.image(d["patch"], width="stretch")
                st.html(f'<div class="gcap"><b><span class="tag" style="background:{hexc(c)}">#{d["id"]}</span> '
                        f'{nice(names[d["label"]])}</b><span>{d["conf"]:.0%}</span></div>')
                if st.button(f"{mmss(d['first'])}", key=f"jump_{d['id']}", icon=":material/play_arrow:",
                             width="stretch", type="tertiary"):
                    st.session_state["start"] = max(0, int(d["first"]) - 1)
                    st.rerun()
        if len(damages) > 24:
            st.caption(f"Showing the 24 most confident of {len(damages)} damages; the table lists all.")
    with tab_t:
        st.dataframe(
            [{"ID": f"#{d['id']}", "Damage": f"{nice(names[d['label']])} ({classes[d['label']]})",
              "Confidence": d["conf"], "First seen": mmss(d["first"]), "Last seen": mmss(d["last"]),
              "Detections": d["hits"], "VGG16 check": f"{nice(names[int(p.argmax())])} {p.max():.0%}"}
             for d, p in zip(damages, P)],
            hide_index=True, width="stretch", column_config={
                "Confidence": st.column_config.ProgressColumn("Confidence", min_value=0.0, max_value=1.0,
                                                              format="percent")})

# ============================================================================ PHOTO
elif mode.endswith("Photo"):
    f = st.file_uploader("Upload a road photo", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if not f:
        empty_state("📷", "Drop a road photo", "jpg or png, ideally taken from a car facing the road")
        st.stop()
    img = Image.open(f).convert("RGB")
    if missing(DET_CKPT, "Train it with `python -m src.detect`", "v1.1"):
        st.stop()
    det, dck = load_det()
    with st.spinner("Finding damage…"):
        b, lab, s = make_detect_fn(det, keep, preset, floor=thresh)([img])[0]
        b, lab, s, _ = merge_overlaps(b, lab, s, len(classes), MERGE_IOU)
    dets = [{"box": bb.tolist(), "label": int(ll), "score": float(ss), "id": i + 1}
            for i, (bb, ll, ss) in enumerate(zip(b, lab, s))]
    left, right = st.columns([2.3, 1], gap="large")
    with left:
        st.image(draw_detections(img.copy(), dets, classes, names), width="stretch")
    with right:
        counts = [sum(d["label"] == j for d in dets) for j in range(len(classes))]
        summary_card(counts, classes, names, [("Image", f"{img.width}×{img.height}"),
                                              ("Accuracy", preset)], title="Damage in photo")
        if dets:
            P = classify_batch(model, [cut_patch(img, d["box"], ckpt["input_size"], ckpt["context"]) for d in dets])
            st.html("<div style='height:12px'></div>" + "".join(
                f'<div class="det"><div class="sw" style="background:{hexc(classes[d["label"]])}"></div>'
                f'<div class="t"><b>#{d["id"]} {nice(names[d["label"]])}</b>'
                f'<span>VGG16: {nice(names[int(p.argmax())])} {p.max():.0%}</span></div>'
                f'<div class="s">{d["score"]:.0%}</div></div>' for d, p in zip(dets, P)))
        else:
            st.info("No damage above this confidence.")
    with st.expander("Classify your own box with VGG16", icon=":material/crop_free:"):
        box = pick_box(img, key="photo_box")
        if box and box[2] - box[0] >= 4 and box[3] - box[1] >= 4:
            patch = cut_patch(img, box, ckpt["input_size"], ckpt["context"])
            show_vgg(patch, classify(model, patch), ckpt)
            st.caption(f"Box (px): {[int(v) for v in box]} · square context crop ×{ckpt['context']}")

# ============================================================================ PATCH
else:
    f = st.file_uploader("Upload a patch", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    if not f:
        empty_state("🔍", "Drop a close-up of one damage",
                    "An already-cropped patch; VGG16 tells you which of the 4 types it is")
        st.stop()
    img = Image.open(f).convert("RGB").resize((ckpt["input_size"],) * 2, Image.BILINEAR)
    patch = np.asarray(img, dtype=np.uint8)
    show_vgg(patch, classify(model, patch), ckpt)
