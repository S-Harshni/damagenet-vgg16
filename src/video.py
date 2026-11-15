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
