"""Detector helpers: IoU, VOC AP, flip augmentation, tracker, and drawing boxes on a video."""
import io

import numpy as np
import pytest
from PIL import Image

from src.detect import DetDataset, average_precision, box_iou, voc_ap
from src.video import IoUTracker, annotate_video, track_label, track_video


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


def _fake_detect(calls):
    def fn(imgs):  # frames arrive in batches: one result per image
        calls.append(len(imgs))
        return [(np.array([[40, 30, 120, 90]], float), np.array([3]), np.array([0.9])) for _ in imgs]
    return fn


def _decode(vid):
    av = pytest.importorskip("av")
    with av.open(io.BytesIO(vid)) as c:
        return [f.to_ndarray(format="rgb24") for f in c.decode(c.streams.video[0])]


def _has_d40_edge(f):  # top edge of the D40 box (pink-red: high R, much lower B), allowing for H.264 blur
    rows = f[27:35, 60:100].astype(float).mean(1)
    return bool(((rows[:, 0] > 170) & (rows[:, 0] - rows[:, 2] > 60)).any())


def test_annotate_video_steady_boxes_span_the_track():
    pytest.importorskip("av")
    calls = []
    vid, log, damages = annotate_video(_video(), _fake_detect(calls), ["D00", "D10", "D20", "D40"],
                                       ["a", "b", "c", "pothole"], detect_every=3)
    assert calls == [4]  # frames 0, 3, 6, 9 in one batch
    assert len(damages) == 1 and damages[0]["id"] == 1 and damages[0]["label"] == 3
    assert damages[0]["patch"].shape == (128, 128, 3)
    assert sorted({r["frame"] for r in log}) == list(range(10))  # drawn between runs, not after the last
    frames = _decode(vid)
    assert len(frames) == 12
    assert all(_has_d40_edge(f) for f in frames[:10]) and not _has_d40_edge(frames[11])


def test_annotate_video_raw_mode_holds_last_detections():
    pytest.importorskip("av")
    vid, log, damages = annotate_video(_video(), _fake_detect([]), ["D00", "D10", "D20", "D40"],
                                       ["a", "b", "c", "pothole"], detect_every=3, stabilize=False)
    assert len(log) == 12 and len(damages) == 1
    assert all(_has_d40_edge(f) for f in _decode(vid))


def test_merge_overlaps_keeps_best_box_and_runner_up_class():
    from src.video import merge_overlaps
    boxes = np.array([[0, 0, 10, 10], [1, 0, 11, 10], [50, 50, 60, 60]], float)
    b, lab, s, probs = merge_overlaps(boxes, np.array([3, 2, 0]), np.array([0.6, 0.7, 0.5]), 4, iou=0.6)
    assert lab.tolist() == [2, 0] and s.tolist() == [0.7, 0.5]
    assert probs[0].tolist() == [0, 0, 0.7, 0.6]  # the D40 runner-up still counts in the track vote


def test_smooth_boxes_exact_on_linear_motion_and_removes_noise():
    from src.video import smooth_boxes
    T = np.arange(20) * 0.08
    true = np.stack([10 + 50 * T, 20 + 0 * T, 40 + 50 * T, 50 + 0 * T], 1)  # moving right, fixed size
    assert np.allclose(smooth_boxes(T, true, T, 0.15), true, atol=1e-6)
    noisy = true + np.random.default_rng(0).normal(0, 2.0, true.shape)
    err_raw = np.abs(noisy - true).mean()
    err_smooth = np.abs(smooth_boxes(T, noisy, T, 0.15) - true).mean()
    assert err_smooth < 0.6 * err_raw


def test_stable_tracker_follows_fast_box_with_prediction():
    from src.video import StableTracker
    p = np.array([[0, 0, 0, 0.9]])
    tr = StableTracker(iou_thr=0.25)
    a = tr.update(np.array([[0.0, 0, 10, 10]]), np.array([3]), np.array([0.9]), p, 0.0)[0]
    b = tr.update(np.array([[4.0, 0, 14, 10]]), np.array([3]), np.array([0.9]), p, 0.1)[0]
    assert a == b
    # the box now jumps 11 px: IoU with the last box is only 0.03, but the track's velocity predicts it
    ids = [tr.update(np.array([[4.0 + 11 * k, 0, 14 + 11 * k, 10]]), np.array([3]), np.array([0.9]), p,
                     0.1 + 0.275 * k)[0] for k in (1, 2)]  # 40 px/s
    assert ids == [a, a]


def test_stabilize_tracks_confirms_and_fixes_label():
    from src.video import stabilize_tracks
    frames = np.arange(10) / 10
    box = np.array([0, 0, 10, 10], float)
    tracks = {
        7: {"t": [0.0, 0.2, 0.4, 0.6], "box": [box] * 4, "score": [0.9, 0.5, 0.8, 0.7], "best": 0.9, "best_t": 0.0,
            "probs": [np.array([0, 0, 0.9, 0.0]), np.array([0, 0, 0, 0.5]), np.array([0, 0, 0.8, 0]),
                      np.array([0, 0, 0.7, 0])]},
        9: {"t": [0.0, 0.1], "box": [box + 50] * 2, "score": [0.95, 0.95], "best": 0.95, "best_t": 0.0,
            "probs": [np.array([0.95, 0, 0, 0])] * 2},  # only 2 detections: a blip, dropped
    }
    per_frame, damages = stabilize_tracks(tracks, frames, thresh=0.5, min_hits=3)
    assert [d["id"] for d in damages] == [1] and damages[0]["label"] == 2
    assert damages[0]["conf"] == pytest.approx((0.9 + 0.8 + 0.7) / 3)
    assert [len(p) for p in per_frame] == [1] * 7 + [0] * 3  # frames 0.0-0.6 incl. the in-between ones
    assert {d["label"] for p in per_frame for d in p} == {2}


def test_detect_multiscale_restores_model_size():
    import torch

    from src.detect import detect

    class Fake(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transform = type("T", (), {"min_size": (576,), "max_size": 960})()
            self.seen = []

        def forward(self, xs):
            self.seen.append(self.transform.min_size[0])
            return [{"boxes": torch.tensor([[1.0, 1, 5, 5]]), "scores": torch.tensor([0.9]),
                     "labels": torch.tensor([1])} for _ in xs]

    m = Fake()
    o = detect(m, [torch.zeros(3, 20, 20)], tta=True, scales=(576, 704))[0]
    assert m.seen == [576, 576, 704, 704] and len(o["boxes"]) == 2  # plain + mirrored box survive NMS
    assert m.transform.min_size == (576,) and m.transform.max_size == 960


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


def test_per_class_threshold_and_preview():
    pytest.importorskip("av")
    seen = []

    def det(imgs):  # a pothole at 0.35 and a crack at 0.35
        return [(np.array([[40, 30, 120, 90], [5, 5, 30, 20]], float), np.array([3, 0]), np.array([0.35, 0.35]))
                for _ in imgs]
    r = track_video(_video(), det, ["D00", "D10", "D20", "D40"], thresh=[0.5, 0.5, 0.5, 0.3], detect_every=1,
                    batch_size=3, preview=seen.append, names=["a", "b", "c", "pothole"])
    assert [d["label"] for d in r["damages"]] == [3]  # only the pothole passes its lower threshold
    assert seen and seen[0].size == (160, 120)
