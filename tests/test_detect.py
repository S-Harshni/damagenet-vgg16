"""Detector helpers: IoU, VOC AP, flip augmentation, tracker, and drawing boxes on a video."""
import io

import numpy as np
import pytest
from PIL import Image

from src.detect import DetDataset, average_precision, box_iou, voc_ap
from src.video import IoUTracker, annotate_video, track_label


def test_box_iou():
    a = np.array([[0, 0, 10, 10]], float)
    b = np.array([[0, 0, 10, 10], [5, 0, 15, 10], [20, 20, 30, 30]], float)
    assert np.allclose(box_iou(a, b)[0], [1.0, 50 / 150, 0.0])


def test_voc_ap_perfect_and_half():
    assert voc_ap(np.array([0.5, 1.0]), np.array([1.0, 1.0])) == pytest.approx(1.0)
    assert voc_ap(np.array([0.5]), np.array([1.0])) == pytest.approx(0.5)


def test_average_precision_counts_duplicates_as_false_positives():
    gts = [{"boxes": np.array([[0, 0, 10, 10]], float), "labels": np.array([4])}]
    preds = [{"boxes": np.array([[0, 0, 10, 10], [0, 0, 10, 10], [50, 50, 60, 60]], float),
              "labels": np.array([4, 4, 1]), "scores": np.array([0.9, 0.8, 0.7])}]
    aps = average_precision(gts, preds, num_classes=4)
    assert aps[3] == pytest.approx(1.0)  # first hit at full precision
    assert np.isnan(aps[1])               # no D10 ground truth -> undefined, not 0


def test_hflip_moves_boxes(tmp_path, monkeypatch):
    img = np.zeros((20, 40, 3), np.uint8)
    p = tmp_path / "a.png"
    Image.fromarray(img).save(p)
    rec = {"file": "a.png", "boxes": [{"label": 3, "box": [2, 3, 12, 13]}]}
    ds = DetDataset([rec], tmp_path, max_side=1000, train=True)
    monkeypatch.setattr("random.random", lambda: 0.0)  # always flip
    x, t = ds[0]
    assert x.shape == (3, 20, 40)
    assert t["boxes"].tolist() == [[28, 3, 38, 13]] and t["labels"].tolist() == [4]


def test_tracker_keeps_id_and_votes_class():
    tr = IoUTracker(iou_thr=0.3, max_age=2)
    a = tr.update(np.array([[0, 0, 10, 10]], float), np.array([3]), np.array([0.9]), 0.0)
    b = tr.update(np.array([[1, 1, 11, 11], [50, 50, 60, 60]], float), np.array([3, 0]),
                  np.array([0.8, 0.6]), 0.1)
    assert a[0] == b[0] and b[1] != b[0]
    assert track_label(tr.tracks[a[0]]) == 3 and tr.tracks[a[0]]["hits"] == 2


def _video(n=12, fps=10, w=160, h=120) -> bytes:
    av = pytest.importorskip("av")
    buf = io.BytesIO()
    with av.open(buf, mode="w", format="mp4") as out:
        s = out.add_stream("libx264", rate=fps)
        s.width, s.height, s.pix_fmt = w, h, "yuv420p"
        for _ in range(n):
            for p in s.encode(av.VideoFrame.from_ndarray(np.full((h, w, 3), 90, np.uint8), format="rgb24")):
                out.mux(p)
        for p in s.encode():
            out.mux(p)
    return buf.getvalue()


def test_annotate_video_draws_on_every_frame():
    av = pytest.importorskip("av")
    calls = []

    def fake_detect(imgs):
        calls.append(1)
        return [(np.array([[40, 30, 120, 90]], float), np.array([3]), np.array([0.9]))]

    vid, log, tracks = annotate_video(_video(), fake_detect, ["D00", "D10", "D20", "D40"],
                                      ["a", "b", "c", "pothole"], detect_every=3)
    assert len(calls) == 4 and len(log) == 4 and list(tracks) == [1]
    assert tracks[1]["patch"].shape == (128, 128, 3)
    with av.open(io.BytesIO(vid)) as c:
        frames = [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]
    assert len(frames) == 12
    # the orange D40 box edge is drawn on frames between detector runs too
    for f in frames:
        assert f[30:33, 60:100, 0].mean() > 180 and f[30:33, 60:100, 2].mean() < 120


def test_f1_at_threshold():
    from src.detect import f1_at
    gts = [{"boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30]], float), "labels": np.array([4, 1])}]
    preds = [{"boxes": np.array([[0, 0, 10, 10], [20, 20, 30, 30], [50, 50, 60, 60]], float),
              "labels": np.array([4, 2, 4]), "scores": np.array([0.9, 0.8, 0.3])}]
    r = f1_at(gts, preds, 4, score=0.5)  # 1 TP (D40), 1 FP (wrong class), 1 missed GT
    assert (r["precision"], r["recall"], r["f1"]) == (0.5, 0.5, 0.5)
    r = f1_at(gts, preds, 4, score=0.2)  # extra low-score false positive
    assert r["precision"] == pytest.approx(1 / 3, abs=1e-3)


def test_tta_flips_boxes_back_and_merges():
    import torch

    from src.detect import detect

    class Fake(torch.nn.Module):  # finds a bright square; returns it in image coordinates
        def forward(self, xs):
            out = []
            for x in xs:
                ys, xs_ = torch.nonzero(x[0] > 0.5, as_tuple=True)
                b = torch.tensor([[xs_.min(), ys.min(), xs_.max() + 1, ys.max() + 1]], dtype=torch.float32)
                out.append({"boxes": b, "scores": torch.tensor([0.9]), "labels": torch.tensor([4])})
            return out

    x = torch.zeros(3, 20, 40)
    x[:, 5:10, 2:8] = 1.0  # object near the left edge
    o = detect(Fake(), [x], tta=True)[0]
    assert o["boxes"].tolist() == [[2.0, 5.0, 8.0, 10.0]]  # flipped copy mapped back, then NMS-merged
