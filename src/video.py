"""Video helpers for the demo app: decode, detect, track, smooth and draw road damage.

Pipeline for an uploaded video (`annotate_video`):

1. Pass 1: run the detector on every `detect_every`-th frame. Overlapping boxes of *different*
   classes are merged (the detector's own NMS is per class, so one pothole can come out as a D40
   box and a D20 box that flicker against each other). Detections are linked into tracks by IoU
   with each track's predicted position (constant-velocity fit over its last detections).
2. Tracks are kept only if they were detected on >= `min_hits` detector runs and their
   confidence (mean of their 3 best scores) reaches the user's threshold. Weaker detections
   (down to half the threshold) may extend a kept track, so a damage does not blink off when its
   score dips for a few frames.
3. Because the whole video is processed before it is shown, each kept track is smoothed with
   both past and future detections (local linear Gaussian-kernel regression of centre and log
   size over time). That removes the frame-to-frame wobble, interpolates the frames between
   detector runs and short misses, and gives each damage one label and one confidence for its
   whole life instead of a label that switches from frame to frame.
4. Pass 2: re-decode the video and draw the smoothed boxes on every frame.

`stabilize=False` reproduces the earlier behaviour (raw per-run detections held until the next
run) and is kept for comparison: see scripts/video_stability.py for the measured difference.
"""
from __future__ import annotations

import io
from functools import lru_cache

import numpy as np


def read_frames(data: bytes, every_s: float = 0.5, max_frames: int = 240,
                max_width: int = 1280) -> tuple[list[np.ndarray], list[float], float]:
    """Decode a video from bytes (no temp files) and keep one frame every `every_s` seconds.

    Returns (frames as RGB uint8 arrays, timestamps in seconds, video duration in seconds).
    Frames wider than `max_width` are downscaled to keep memory small.
    """
    import av  # PyAV: bundled FFmpeg, reads mp4/mov/avi/webm/mkv from a file-like object

    frames, times = [], []
    next_t = 0.0
    with av.open(io.BytesIO(data)) as box:
        stream = box.streams.video[0]
        stream.thread_type = "AUTO"
        duration = float(box.duration / av.time_base) if box.duration else 0.0
        for fr in box.decode(stream):
            t = float(fr.time) if fr.time is not None else len(times) * every_s
            if t + 1e-6 < next_t:
                continue
            if fr.width > max_width:
                fr = fr.reformat(width=max_width, height=round(fr.height * max_width / fr.width / 2) * 2)
            frames.append(fr.to_ndarray(format="rgb24"))
            times.append(round(t, 2))
            while next_t <= t + 1e-6:  # fixed grid 0, every_s, 2*every_s, ... (no drift)
                next_t += every_s
            if len(frames) >= max_frames:
                break
    if not frames:
        raise ValueError("no video frames could be decoded")
    return frames, times, duration or times[-1]


# ------------------------------------------------------------------ colours and drawing
# Bright colours that stand out on grey asphalt (and on the app's dark background).
CLASS_COLORS = {"D00": (79, 195, 247), "D10": (255, 213, 79), "D20": (179, 136, 255), "D40": (255, 83, 112)}
INK = (20, 22, 26)  # tag text: near-black on the bright class colour


@lru_cache(maxsize=8)
def _font(size: int):
    from PIL import ImageFont
    try:  # DejaVu Sans Bold ships with matplotlib (a project dependency): same look everywhere
        import matplotlib
        from pathlib import Path
        return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), size)
    except Exception:
        try:
            return ImageFont.load_default(size=size)
        except TypeError:  # Pillow < 10.1
            return ImageFont.load_default()


def draw_detections(img, dets: list[dict], classes: list[str], names: list[str]):
    """Draw each detection onto a PIL image (in place): a light tint inside the box, a class-colour
    outline with a thin dark edge (readable on bright and dark road), and a rounded tag
    '#3 Pothole 87%'. dets: {"box": [x0, y0, x1, y1], "label": int, "score": float, "id": int?}."""
    from PIL import Image, ImageDraw
    if not dets:
        return img
    W, H = img.size
    lw = max(2, round(W / 420))
    font = _font(max(12, round(W / 64)))
    for det in sorted(dets, key=lambda d: d["score"]):  # most confident drawn last (on top)
        col = CLASS_COLORS.get(classes[det["label"]], (255, 107, 53))
        x0, y0, x1, y1 = (int(round(v)) for v in det["box"])
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W - 1, x1), min(H - 1, y1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue
        region = img.crop((x0, y0, x1, y1))
        img.paste(Image.blend(region, Image.new("RGB", region.size, col), 0.14), (x0, y0))
    d = ImageDraw.Draw(img)
    for det in sorted(dets, key=lambda d: d["score"]):
        col = CLASS_COLORS.get(classes[det["label"]], (255, 107, 53))
        x0, y0, x1, y1 = det["box"]
        d.rectangle([x0 - lw, y0 - lw, x1 + lw, y1 + lw], outline=INK, width=1)
        d.rectangle([x0, y0, x1, y1], outline=col, width=lw)
        name = names[det["label"]]
        tag = (f"#{det['id']} " if det.get("id") else "") + f"{name[:1].upper()}{name[1:]} {det['score']:.0%}"
        tb = d.textbbox((0, 0), tag, font=font)
        pad = max(3, font.size // 4)
        tw, th = tb[2] - tb[0] + 2 * pad, tb[3] - tb[1] + 2 * pad
        tx = min(max(0, x0 - lw), W - tw)
        ty = y0 - lw - th - 2 if y0 - lw - th - 2 >= 0 else y0 + lw + 2  # above the box, else inside
        d.rounded_rectangle([tx, ty, tx + tw, ty + th], radius=th // 3, fill=col, outline=INK, width=1)
        d.text((tx + pad - tb[0], ty + pad - tb[1]), tag, fill=INK, font=font)
    return img


# ------------------------------------------------------------------ raw tracker (stabilize=False)
class IoUTracker:
    """Greedy IoU tracker on raw detections (the earlier, unsmoothed video mode).

    A track survives `max_age` detection steps without a match (covers a missed frame).
    Its class is the majority vote of its detections, its score the best one seen.
    """

    def __init__(self, iou_thr: float = 0.3, max_age: int = 3):
        self.iou_thr, self.max_age = iou_thr, max_age
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def update(self, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, t: float) -> list[int]:
        from .detect import box_iou
        live = [k for k, tr in self.tracks.items() if tr["age"] <= self.max_age]
        ious = box_iou(boxes, np.array([self.tracks[k]["box"] for k in live]).reshape(-1, 4))
        ids = [-1] * len(boxes)
        used = set()
        for flat in np.argsort(-ious, axis=None):
            i, j = np.unravel_index(flat, ious.shape)
            if ious[i, j] < self.iou_thr:
                break
            if ids[i] != -1 or live[j] in used:
                continue
            ids[i] = live[j]
            used.add(live[j])
        for k in self.tracks.values():
            k["age"] += 1
        for i, (b, c, s) in enumerate(zip(boxes, labels, scores)):
            if ids[i] == -1:
                ids[i] = self.next_id
                self.tracks[ids[i]] = {"votes": {}, "best": 0.0, "first": t, "hits": 0}
                self.next_id += 1
            tr = self.tracks[ids[i]]
            tr.update(box=b, age=0, last=t)
            tr["hits"] += 1
            tr["votes"][int(c)] = tr["votes"].get(int(c), 0.0) + float(s)
            if s > tr["best"]:
                tr["best"], tr["best_box"], tr["best_t"] = float(s), b.copy(), t
        return ids

    def label(self, tid: int) -> int:
        return track_label(self.tracks[tid])


def track_label(track: dict) -> int:
    """Class of a track: the class with the highest summed detection score."""
    v = track["votes"]
    return max(v, key=v.get)


# ------------------------------------------------------------------ stabilised tracking
def merge_overlaps(boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, n_cls: int,
                   iou: float = 0.6) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Class-agnostic NMS: keep the best box of each group overlapping at IoU >= `iou`.

    Returns kept (boxes, labels, scores, class_scores) where class_scores[k, c] is the best score
    class c got inside group k, so a track can still weigh the runner-up classes when voting.
    """
    from .detect import box_iou
    n = len(boxes)
    if n == 0:
        return boxes.reshape(0, 4), labels, scores, np.zeros((0, n_cls))
    ious = box_iou(boxes, boxes)
    owner = np.full(n, -1)
    kept = []
    for i in np.argsort(-scores, kind="stable"):
        if owner[i] != -1:
            continue
        owner[(ious[i] >= iou) & (owner == -1)] = len(kept)
        owner[i] = len(kept)
        kept.append(i)
    probs = np.zeros((len(kept), n_cls))
    for j in range(n):
        probs[owner[j], labels[j]] = max(probs[owner[j], labels[j]], scores[j])
    k = np.array(kept)
    return boxes[k], labels[k], scores[k], probs


def _to_z(b: np.ndarray) -> np.ndarray:
    """[x0, y0, x1, y1] -> [cx, cy, log w, log h] (log size: growth while driving is multiplicative)."""
    b = np.asarray(b, float).reshape(-1, 4)
    w, h = np.maximum(b[:, 2] - b[:, 0], 1.0), np.maximum(b[:, 3] - b[:, 1], 1.0)
    return np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2, np.log(w), np.log(h)], 1)


def _from_z(z: np.ndarray) -> np.ndarray:
    z = np.asarray(z, float).reshape(-1, 4)
    w, h = np.exp(z[:, 2]), np.exp(z[:, 3])
    return np.stack([z[:, 0] - w / 2, z[:, 1] - h / 2, z[:, 0] + w / 2, z[:, 1] + h / 2], 1)


class StableTracker:
    """Links detections into tracks using each track's *predicted* box (a linear fit of centre and
    log size over its last 4 detections), so fast-growing damage close to the camera stays on one
    track. Greedy matching by IoU; unmatched detections start new tracks. A track is dropped from
    matching after `max_gap` seconds without a detection."""

    def __init__(self, iou_thr: float = 0.25, max_gap: float = 0.6, history: int = 4):
        self.iou_thr, self.max_gap, self.history = iou_thr, max_gap, history
        self.tracks: dict[int, dict] = {}
        self.next_id = 1

    def predict(self, tr: dict, t: float) -> np.ndarray:
        T = np.array(tr["t"][-self.history:])
        Z = _to_z(np.array(tr["box"][-self.history:]))
        if len(T) < 2 or T[-1] - T[0] < 1e-6:
            return _from_z(Z[-1])[0]
        A = np.stack([np.ones_like(T), T - T[-1]], 1)
        coef = np.linalg.lstsq(A, Z, rcond=None)[0]
        dt = min(t - T[-1], 0.3)  # don't extrapolate a noisy velocity too far
        return _from_z(coef[0] + coef[1] * dt)[0]

    def update(self, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray, probs: np.ndarray,
               t: float, can_start: np.ndarray | None = None) -> list[int]:
        """Returns a track id per detection, or -1 for an unmatched one not allowed to start a track."""
        from .detect import box_iou
        live = [k for k, tr in self.tracks.items() if t - tr["t"][-1] <= self.max_gap]
        pred = np.array([self.predict(self.tracks[k], t) for k in live]).reshape(-1, 4)
        ious = box_iou(np.asarray(boxes, float).reshape(-1, 4), pred)
        ids, used = [-1] * len(boxes), set()
        for flat in np.argsort(-ious, axis=None, kind="stable"):
            i, j = np.unravel_index(flat, ious.shape)
            if ious[i, j] < self.iou_thr:
                break
            if ids[i] == -1 and j not in used:
                ids[i] = live[j]
                used.add(j)
        for i in range(len(boxes)):
            if ids[i] == -1 and can_start is not None and not can_start[i]:
                continue  # weak detection: may extend an existing track, never starts one
            if ids[i] == -1:
                ids[i] = self.next_id
                self.tracks[ids[i]] = {"t": [], "box": [], "score": [], "probs": [], "best": 0.0}
                self.next_id += 1
            tr = self.tracks[ids[i]]
            tr["t"].append(t)
            tr["box"].append(np.asarray(boxes[i], float))
            tr["score"].append(float(scores[i]))
            tr["probs"].append(np.asarray(probs[i], float))
            if scores[i] > tr["best"]:
                tr["best"], tr["best_t"], tr["best_box"] = float(scores[i]), t, np.asarray(boxes[i], float)
        return ids


def track_confidence(scores) -> float:
    """A track's confidence: mean of its 3 best detection scores (robust to a one-frame spike)."""
    s = np.sort(np.asarray(scores, float))[::-1]
    return float(s[:3].mean()) if len(s) else 0.0


def smooth_boxes(T: np.ndarray, B: np.ndarray, q: np.ndarray, bandwidth: float) -> np.ndarray:
    """Local linear Gaussian-kernel regression of (cx, cy, log w, log h) over time, evaluated at
    times `q`. Unlike a moving average it has no lag at the start/end of a track for steady
    motion, and it uses future detections too (the whole video is known before drawing)."""
    T, Z, q = np.asarray(T, float), _to_z(B), np.asarray(q, float)
    W = np.exp(-0.5 * ((q[:, None] - T[None, :]) / bandwidth) ** 2) + 1e-12  # (Q, N)
    sw = W.sum(1, keepdims=True)
    mt = (W * T).sum(1, keepdims=True) / sw
    d = T[None, :] - mt                                       # (Q, N)
    var = (W * d * d).sum(1, keepdims=True) / sw
    mz = W @ Z / sw                                           # (Q, 4)
    cov = np.einsum("qn,qn,nk->qk", W, d, Z) / sw             # = sum w d (z - mz): d has weighted mean 0
    slope = np.where(var > 1e-9, cov / np.maximum(var, 1e-9), 0.0)
    return _from_z(mz + slope * (q[:, None] - mt))


def stabilize_tracks(tracks: dict[int, dict], frame_times: np.ndarray, thresh, min_hits: int = 3,
                     bandwidth: float = 0.15, max_gap: float = 0.6,
                     dedupe: float = 0.5) -> tuple[list[list[dict]], list[dict]]:
    """Turn raw tracks into what is drawn: per frame, a list of smoothed boxes.

    Keeps tracks with >= `min_hits` detections and confidence >= `thresh` (a float, or one value per
    class: the track's class decides which applies); each keeps one label
    (highest summed class score over its life) and one confidence. Frames inside a gap longer
    than `max_gap` seconds are not drawn. Display IDs are renumbered 1, 2, ... by first sighting.
    Returns (per-frame detections, damages sorted by first sighting).
    """
    thr = np.broadcast_to(np.asarray(thresh, float), (len(next(iter(tracks.values()))["probs"][0]),)) \
        if tracks else np.zeros(1)
    kept = [(k, tr) for k, tr in tracks.items()
            if len(tr["t"]) >= min_hits and track_confidence(tr["score"]) >= thr[int(np.sum(tr["probs"], 0).argmax())]]
    kept.sort(key=lambda kt: kt[1]["t"][0])
    per_frame: list[list[dict]] = [[] for _ in frame_times]
    damages = []
    for new_id, (_, tr) in enumerate(kept, 1):
        T, B = np.array(tr["t"]), np.array(tr["box"])
        label = int(np.sum(tr["probs"], 0).argmax())
        conf = track_confidence(tr["score"])
        lo, hi = np.searchsorted(frame_times, T[0] - 1e-6), np.searchsorted(frame_times, T[-1] + 1e-6)
        idx = np.arange(lo, hi)
        q = frame_times[idx]
        nxt = np.clip(np.searchsorted(T, q - 1e-6), 0, len(T) - 1)  # first detection at or after q
        prv = np.clip(nxt - (T[nxt] > q + 1e-6), 0, len(T) - 1)     # last detection at or before q
        ok = (T[nxt] - T[prv]) <= max_gap + 1e-6
        idx, q = idx[ok], q[ok]
        if len(q):
            for n, b in zip(idx, smooth_boxes(T, B, q, bandwidth)):
                per_frame[n].append({"box": b.tolist(), "label": label, "score": conf, "id": new_id})
        damages.append({"id": new_id, "label": label, "conf": conf, "best": tr["best"],
                        "first": float(T[0]), "last": float(T[-1]), "hits": len(T),
                        "best_t": tr["best_t"], "patch": tr.get("patch")})
    if dedupe:  # two tracks on one damage: per frame, hide the less confident overlapping box
        from .detect import box_iou
        for n, dets in enumerate(per_frame):
            if len(dets) < 2:
                continue
            dets.sort(key=lambda d: -d["score"])
            B = np.array([d["box"] for d in dets])
            iou = box_iou(B, B)
            area = (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1])
            x0 = np.maximum(B[:, None, 0], B[None, :, 0]); x1 = np.minimum(B[:, None, 2], B[None, :, 2])
            y0 = np.maximum(B[:, None, 1], B[None, :, 1]); y1 = np.minimum(B[:, None, 3], B[None, :, 3])
            iomin = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None) / np.minimum(area[:, None], area[None, :])
            keep = []
            for i in range(len(dets)):
                if all(iou[i, j] < dedupe and iomin[i, j] < 0.8 for j in keep):
                    keep.append(i)
            per_frame[n] = [dets[i] for i in keep]
        shown = {d["id"] for dets in per_frame for d in dets}
        damages = [d for d in damages if d["id"] in shown]
    return per_frame, damages


# ------------------------------------------------------------------ video in, annotated video out
def _open_frames(data: bytes, max_seconds: float):
    import av
    src = av.open(io.BytesIO(data))
    vs = src.streams.video[0]
    vs.thread_type = "AUTO"
    fps = float(vs.average_rate or vs.guessed_rate or 25)
    n_max = int(max_seconds * fps)
    n_total = min(n_max, vs.frames or int((src.duration or 0) / av.time_base * fps) or n_max)
    return src, vs, fps, n_max, n_total


def _frame_image(fr, max_width: int):
    from PIL import Image
    img = fr.to_image()
    if img.width > max_width:
        img = img.resize((max_width, round(img.height * max_width / img.width)), Image.BILINEAR)
    return img


def track_video(data: bytes, detect_fn, classes: list[str], thresh=0.4, detect_every: int = 2,
                stabilize: bool = True, min_hits: int = 4, merge_iou: float = 0.7, iou_thr: float = 0.25,
                max_gap: float = 0.6, bandwidth: float | None = None, start_ratio: float = 1.0,
                dedupe: float = 0.5, max_seconds: float = 120.0, max_width: int = 1280,
                batch_size: int = 4, progress=None, preview=None, names: list[str] | None = None) -> dict:
    """Pass 1: detect on every `detect_every`-th frame, track, and decide what to draw per frame.

    `detect_fn(list_of_PIL) -> list of (boxes, 0-based labels, scores)` should return detections
    down to at least thresh / 2 (weaker ones are ignored). `thresh` is a float or one value per
    class. Frames go to the detector `batch_size` at a time (much faster on a GPU).
    `progress(fraction, text)` and `preview(PIL image with this run's boxes)` are optional.
    Returns {"per_frame": list (one per frame) of drawn boxes, "damages": list sorted by first
    sighting (id, label, conf, best, first, last, hits, best_t, patch), "fps", "size", "n_frames"}.
    """
    from .patches import cut_patch

    thr = np.broadcast_to(np.asarray(thresh, float), (len(classes),)).copy()
    low = np.maximum(0.05, thr / 2) if stabilize else thr
    start = thr * start_ratio
    stable = StableTracker(iou_thr=iou_thr, max_gap=max_gap)
    raw = IoUTracker(max_age=max(2, 6 // detect_every))
    raw_steps: dict[int, list[dict]] = {}
    src, vs, fps, n_max, n_total = _open_frames(data, max_seconds)
    n_frames, size = 0, None
    pending: list[tuple[int, object]] = []
    n_batches = 0

    def flush():
        nonlocal n_batches
        if not pending:
            return
        outs = detect_fn([im for _, im in pending])
        for (n, img), (boxes, labels, scores) in zip(pending, outs):
            t = n / fps
            boxes, labels, scores = (np.asarray(boxes, float).reshape(-1, 4), np.asarray(labels, int),
                                     np.asarray(scores, float))
            k = scores >= low[labels] if len(labels) else np.zeros(0, bool)
            boxes, labels, scores = boxes[k], labels[k], scores[k]
            if stabilize:
                boxes, labels, scores, probs = merge_overlaps(boxes, labels, scores, len(classes), merge_iou)
                ids = stable.update(boxes, labels, scores, probs, t, can_start=scores >= start[labels])
                tracks = stable.tracks
                boxes = [b for b, i in zip(boxes, ids) if i != -1]
                ids = [i for i in ids if i != -1]
            else:
                ids = raw.update(boxes, labels, scores, t)
                tracks = raw.tracks
                raw_steps[n] = [{"box": b.tolist(), "label": int(c), "score": float(s), "id": i}
                                for b, c, s, i in zip(boxes, labels, scores, ids)]
            for i, b in zip(ids, boxes):  # keep a clean patch of each track's best view
                if tracks[i].get("best_t") == t:
                    tracks[i]["patch"] = cut_patch(img, b, 128, 1.15)
        n_batches += 1
        n_last, img_last = pending[-1]
        if progress:
            progress(0.85 * min(1.0, (n_last + 1) / max(1, n_total)), f"Scanning frame {n_last + 1} of ~{n_total}")
        if preview and names and n_batches % 3 == 1:  # quick look at what is being found
            b, lab, sc = outs[-1]
            lab = np.asarray(lab, int)
            keep = np.asarray(sc) >= thr[lab] if len(lab) else []
            dets = [{"box": list(bb), "label": int(ll), "score": float(ss)}
                    for bb, ll, ss, kk in zip(b, lab, sc, keep) if kk]
            preview(draw_detections(img_last.copy(), dets, classes, names))
        pending.clear()

    with src:
        for n, fr in enumerate(src.decode(vs)):
            if n >= n_max:
                break
            n_frames = n + 1
            if n % detect_every:
                continue
            img = _frame_image(fr, max_width)
            size = img.size
            pending.append((n, img))
            if len(pending) >= batch_size:
                flush()
        flush()
    if size is None:
        raise ValueError("no video frames could be decoded")

    frame_times = np.arange(n_frames) / fps
    if stabilize:
        bw = bandwidth if bandwidth is not None else max(0.15, 1.5 * detect_every / fps)
        per_frame, damages = stabilize_tracks(stable.tracks, frame_times, thr, min_hits, bw, max_gap, dedupe)
    else:  # earlier behaviour: hold each run's raw detections until the next run
        per_frame, cur = [], []
        for n in range(n_frames):
            cur = raw_steps.get(n, cur)
            per_frame.append(cur)
        damages = [{"id": k, "label": track_label(tr), "conf": tr["best"], "best": tr["best"],
                    "first": tr["first"], "last": tr["last"], "hits": tr["hits"], "best_t": tr["best_t"],
                    "patch": tr.get("patch")} for k, tr in raw.tracks.items() if tr["hits"] >= 2]
    return {"per_frame": per_frame, "damages": damages, "fps": fps, "size": size, "n_frames": n_frames}


def render_video(data: bytes, per_frame: list[list[dict]], classes: list[str], names: list[str],
                 size: tuple[int, int], max_width: int = 1280, progress=None) -> bytes:
    """Pass 2: re-decode the video, draw `per_frame` boxes on every frame and encode an H.264
    fragmented mp4 that browsers play directly (fragmented: BytesIO cannot be seeked for faststart)."""
    import av
    out_buf = io.BytesIO()
    n_frames = len(per_frame)
    src, vs, fps, _, _ = _open_frames(data, 1e9)
    with src, av.open(out_buf, mode="w", format="mp4",
                      options={"movflags": "frag_keyframe+empty_moov+default_base_moof"}) as dst:
        os_ = dst.add_stream("libx264", rate=round(fps))
        os_.width, os_.height = size[0] // 2 * 2, size[1] // 2 * 2
        os_.pix_fmt = "yuv420p"
        os_.options = {"crf": "22", "preset": "superfast"}
        for n, fr in enumerate(src.decode(vs)):
            if n >= n_frames:
                break
            img = draw_detections(_frame_image(fr, max_width), per_frame[n], classes, names)
            if img.size != (os_.width, os_.height):
                img = img.crop((0, 0, os_.width, os_.height))
            for p in os_.encode(av.VideoFrame.from_image(img)):
                dst.mux(p)
            if progress and n % 10 == 0:
                progress(0.85 + 0.15 * (n + 1) / n_frames, f"Drawing boxes {n + 1} of {n_frames}")
        for p in os_.encode():
            dst.mux(p)
    return out_buf.getvalue()


def annotate_video(data: bytes, detect_fn, classes: list[str], names: list[str], thresh: float = 0.4,
                   detect_every: int = 2, stabilize: bool = True, max_seconds: float = 120.0,
                   max_width: int = 1280, progress=None, **track_kw) -> tuple[bytes, list[dict], list[dict]]:
    """Find, track and draw road damage on a video (`track_video` then `render_video`).

    Returns (H.264 mp4 bytes, one log row per drawn box per frame, damages sorted by first sighting).
    """
    r = track_video(data, detect_fn, classes, thresh, detect_every, stabilize, max_seconds=max_seconds,
                    max_width=max_width, progress=progress, names=names, **track_kw)
    fps = r["fps"]
    log = [{"time_s": round(n / fps, 3), "frame": n, "id": d["id"], "class": classes[d["label"]],
            "score": round(d["score"], 3), "box": [round(v, 1) for v in d["box"]]}
           for n, dets in enumerate(r["per_frame"]) for d in dets]
    vid = render_video(data, r["per_frame"], classes, names, r["size"], max_width, progress)
    return vid, log, r["damages"]
