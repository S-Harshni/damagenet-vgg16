"""Video helper: frames are decoded from bytes and sampled at the requested interval."""
import io

import numpy as np
import pytest

av = pytest.importorskip("av")

from src.video import read_frames  # noqa: E402


def _make_video(seconds=3, fps=10, w=320, h=240) -> bytes:
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        s = out.add_stream("libx264" if "libx264" in av.codecs_available else "mpeg4", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for i in range(seconds * fps):
            img = np.full((h, w, 3), (i * 8) % 255, np.uint8)
            for p in s.encode(av.VideoFrame.from_ndarray(img, format="rgb24")):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    return buf.getvalue()


def test_read_frames_samples_interval():
    frames, times, dur = read_frames(_make_video(), every_s=0.5)
    assert len(frames) == 6
    assert np.allclose(np.diff(times), 0.5, atol=0.11)
    assert frames[0].shape == (240, 320, 3) and frames[0].dtype == np.uint8
    assert 2.5 <= dur <= 3.5


def test_read_frames_downscales_and_caps():
    frames, _, _ = read_frames(_make_video(w=640, h=480), every_s=0.1, max_frames=5, max_width=320)
    assert len(frames) == 5 and frames[0].shape == (240, 320, 3)


def test_read_frames_rejects_garbage():
    with pytest.raises(Exception):
        read_frames(b"not a video")
