"""Video helpers for the demo app: sample frames from an uploaded video in memory.

The model is a *patch classifier*, not a detector: it has no "no damage" class, so it cannot
find damage by itself. For video, the user picks a frame and draws a box around the damage
(or keeps one box fixed across frames), exactly like the photo mode.
"""
from __future__ import annotations

import io

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


# ------------------------------------------------------------------ auto-detection on video
# Bright colours that stand out on grey asphalt (the report palette's charcoal/grey would not).
CLASS_COLORS = {"D00": (79, 195, 247), "D10": (255, 200, 69), "D20": (179, 136, 255), "D40": (255, 107, 53)}


class IoUTracker:
    """Greedy IoU tracker: links detections across frames so each damage gets one ID.

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


def draw_detections(img, dets: list[dict], classes: list[str], names: list[str]):
    """Draw boxes with 'pothole 0.91 #3' tags (class colour = box colour) onto a PIL image (in place)."""
    from PIL import ImageDraw, ImageFont
    d = ImageDraw.Draw(img)
    lw = max(2, round(img.width / 400))
    try:
        font = ImageFont.load_default(size=max(14, round(img.width / 45)))
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    for det in dets:
        c = classes[det["label"]]
        col = CLASS_COLORS.get(c, (255, 107, 53))
        x0, y0, x1, y1 = det["box"]
        d.rectangle([x0, y0, x1, y1], outline=col, width=lw)
        tag = f"{names[det['label']]} {det['score']:.2f}" + (f" #{det['id']}" if det.get("id") else "")
        tb = d.textbbox((0, 0), tag, font=font)
        tw, th = tb[2] - tb[0] + 8, tb[3] - tb[1] + 6
        ty = y0 - th if y0 - th >= 0 else y0
        d.rectangle([x0, ty, x0 + tw, ty + th], fill=col)
        txt = (43, 45, 47)
        d.text((x0 + 4, ty + 3 - tb[1]), tag, fill=txt, font=font)
    return img


def annotate_video(data: bytes, detect_fn, classes: list[str], names: list[str], detect_every: int = 3,
                   max_seconds: float = 120.0, max_width: int = 1280, progress=None) -> tuple[bytes, list[dict], dict]:
    """Run `detect_fn(list_of_PIL) -> list of (boxes, labels, scores)` on every `detect_every`-th
    frame, track the detections, draw boxes + scores on *every* frame (the latest detections
    are kept for frames in between), and re-encode an H.264 mp4 that browsers can play.

    Returns (mp4 bytes, per-frame detection log, tracker.tracks).
    """
    import av
    from PIL import Image

    from .patches import cut_patch

    tracker = IoUTracker(max_age=max(2, 6 // detect_every))
    log, current = [], []
    out_buf = io.BytesIO()
    with av.open(io.BytesIO(data)) as src:
        vs = src.streams.video[0]
        vs.thread_type = "AUTO"
        fps = float(vs.average_rate or vs.guessed_rate or 25)
        n_total = vs.frames or int((src.duration or 0) / av.time_base * fps) or 0
        n_max = int(max_seconds * fps)
        with av.open(out_buf, mode="w", format="mp4", options={"movflags": "frag_keyframe+empty_moov+default_base_moof"}) as dst:
            os_ = None
            for n, fr in enumerate(src.decode(vs)):
                if n >= n_max:
                    break
                img = fr.to_image()
                if img.width > max_width:
                    img = img.resize((max_width, round(img.height * max_width / img.width)), Image.BILINEAR)
                if os_ is None:
                    os_ = dst.add_stream("libx264", rate=round(fps))
                    os_.width, os_.height = img.width // 2 * 2, img.height // 2 * 2
                    os_.pix_fmt = "yuv420p"
                    os_.options = {"crf": "23", "preset": "veryfast"}
                t = n / fps
                if n % detect_every == 0:
                    boxes, labels, scores = detect_fn([img])[0]
                    ids = tracker.update(boxes, labels, scores, t)
                    for i, b in zip(ids, boxes):  # keep a clean patch of each track's best view
                        if tracker.tracks[i]["best_t"] == t:
                            tracker.tracks[i]["patch"] = cut_patch(img, b, 128, 1.15)
                    current = [{"box": b.tolist(), "label": int(c), "score": float(s), "id": i}
                               for b, c, s, i in zip(boxes, labels, scores, ids)]
                    log += [{"time_s": round(t, 2), "frame": n, "id": d["id"], "class": classes[d["label"]],
                             "score": round(d["score"], 3), "box": [round(v) for v in d["box"]]} for d in current]
                draw_detections(img, current, classes, names)
                if img.size != (os_.width, os_.height):
                    img = img.crop((0, 0, os_.width, os_.height))
                for p in os_.encode(av.VideoFrame.from_image(img)):
                    dst.mux(p)
                if progress and n % 5 == 0:
                    progress(min(1.0, (n + 1) / max(1, min(n_total or n_max, n_max))), n + 1)
            if os_ is not None:
                for p in os_.encode():
                    dst.mux(p)
    if os_ is None:
        raise ValueError("no video frames could be decoded")
    return out_buf.getvalue(), log, tracker.tracks
