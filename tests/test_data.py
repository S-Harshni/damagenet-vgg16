"""Unit tests: XML parsing, square cropping and split leakage."""
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from src.data import parse_voc_xml, read_split_list
from src.patches import box_ok, cut_patch, split_images, square_crop_box

CLASSES = ["D00", "D10", "D20", "D40"]
ROOT = Path(__file__).resolve().parent.parent

XML = """<annotation>
  <filename>Norway_000000.jpg</filename>
  <size><width>3650</width><height>2044</height><depth /></size>
  <object><name>D00</name><bndbox><xmin>1138.46</xmin><ymin>1281.93</ymin>
    <xmax>1169.35</xmax><ymax>1336.85</ymax></bndbox>
    <attributes><attribute><name>rotation</name><value>0.0</value></attribute></attributes></object>
  <object><name>D44</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>50</xmax><ymax>50</ymax></bndbox></object>
  <object><name>D40</name><bndbox><xmin>10</xmin><ymin>20</ymin><xmax>110</xmax><ymax>90</ymax></bndbox></object>
  <object><name>Repair</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
  <object><name>D50</name><bndbox><xmin>1</xmin><ymin>1</ymin><xmax>9</xmax><ymax>9</ymax></bndbox></object>
</annotation>"""


# ------------------------------------------------------------------ XML parsing
def test_parse_keeps_only_target_classes():
    ann = parse_voc_xml(XML, CLASSES)
    assert (ann["width"], ann["height"]) == (3650, 2044)
    assert [b["cls"] for b in ann["boxes"]] == ["D00", "D40"]
    assert [b["label"] for b in ann["boxes"]] == [0, 3]
    assert ann["boxes"][0]["box"] == [1138.46, 1281.93, 1169.35, 1336.85]


def test_parse_skips_degenerate_boxes():
    xml = ("<annotation><size><width>10</width><height>10</height></size><object><name>D10</name>"
           "<bndbox><xmin>5</xmin><ymin>5</ymin><xmax>5</xmax><ymax>8</ymax></bndbox></object></annotation>")
    assert parse_voc_xml(xml, CLASSES)["boxes"] == []


def test_parse_rejects_entity_expansion():
    bomb = '<?xml version="1.0"?><!DOCTYPE x [<!ENTITY a "aaaa">]><annotation>&a;</annotation>'
    with pytest.raises(Exception):
        parse_voc_xml(bomb, CLASSES)


def test_read_split_list_strips_crlf(tmp_path):
    p = tmp_path / "list.txt"
    p.write_bytes(b"datasets/raw/rdd2022/RDD2022/India/train/images/India_000000.jpg\r\n"
                  b"datasets/raw/rdd2022/RDD2022/China_Drone/train/images/China_Drone_000001.jpg\r\n\r\n")
    assert read_split_list(p) == ["India/India_000000", "China_Drone/China_Drone_000001"]


# ------------------------------------------------------------------ square crop
@pytest.mark.parametrize("box,w,h", [
    ((100, 100, 140, 120), 600, 600),      # interior box
    ((0, 0, 30, 60), 600, 600),            # top-left corner -> window shifted inside
    ((560, 570, 600, 600), 600, 600),      # bottom-right corner
    ((0, 0, 600, 400), 600, 400),          # box as large as the image -> capped
    ((10, 10, 3000, 50), 3650, 2044),      # very wide box
])
def test_square_crop_is_square_and_inside(box, w, h):
    l, t, r, b = square_crop_box(box, w, h, 1.15)
    assert r - l == b - t > 0
    assert 0 <= l and 0 <= t and r <= w and b <= h


def test_square_crop_centred_and_sized():
    l, t, r, b = square_crop_box((200, 200, 300, 240), 1000, 1000, 1.15)
    assert r - l == 115                     # 1.15 * max(100, 40)
    assert abs((l + r) / 2 - 250) <= 1 and abs((t + b) / 2 - 220) <= 1


def test_cut_patch_shape():
    img = Image.fromarray(np.zeros((300, 400, 3), dtype=np.uint8))
    assert cut_patch(img, (10, 10, 60, 40), 128).shape == (128, 128, 3)


def test_box_filter():
    assert box_ok((0, 0, 30, 30), 16, 900)
    assert not box_ok((0, 0, 15, 200), 16, 900)   # short side < 16
    assert not box_ok((0, 0, 20, 40), 16, 900)    # area 800 < 900


# ------------------------------------------------------------------ split leakage
def test_split_disjoint_complete_and_seeded():
    ids = [f"C/{i:05d}" for i in range(1000)]
    s = split_images(ids, {"train": 0.7, "val": 0.15, "test": 0.15}, 42)
    sets = {k: set(v) for k, v in s.items()}
    assert not (sets["train"] & sets["val"]) and not (sets["train"] & sets["test"]) \
        and not (sets["val"] & sets["test"])
    assert set().union(*sets.values()) == set(ids)
    assert (len(s["train"]), len(s["val"]), len(s["test"])) == (700, 150, 150)
    assert s == split_images(list(reversed(ids)), {"train": 0.7, "val": 0.15, "test": 0.15}, 42)


@pytest.mark.parametrize("base", [ROOT, ROOT / "debug"])
def test_built_patches_have_no_image_leakage(base):
    meta_path = base / "data" / "patches" / "meta.json"
    if not meta_path.exists():
        pytest.skip("patch dataset not built yet")
    meta = json.loads(meta_path.read_text())
    imgs = {k: {m["id"] for m in v} for k, v in meta.items()}
    assert not (imgs["train"] & imgs["val"])
    assert not (imgs["train"] & imgs["test"])
    assert not (imgs["val"] & imgs["test"])
