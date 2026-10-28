"""VisDrone parsing and conversion on SYNTHETIC annotations."""

import numpy as np
import pytest
from PIL import Image

from aerialrange.data import visdrone as vd

MOT_TXT = (
    "1,1,10,20,30,40,1,1,0,0\n"  # pedestrian
    "1,2,100,100,50,30,1,4,0,1\n"  # car
    "1,0,0,0,60,60,0,0,0,0\n"  # ignored region
    "2,1,12,21,30,40,1,1,0,0\n"
    "2,3,300,300,20,20,1,11,0,0\n"  # others
)


def test_parse_mot_annotation_schema():
    a = vd.parse_mot(MOT_TXT)
    assert a.shape == (5,)
    assert a["frame"].tolist() == [1, 1, 1, 2, 2]
    assert a["category"].tolist() == [1, 4, 0, 1, 11]


def test_parse_mot_rejects_wrong_columns():
    with pytest.raises(vd.SchemaError):
        vd.parse_mot("1,1,10,20,30,40,1,1,0\n")


def test_parse_mot_rejects_unknown_category():
    with pytest.raises(vd.SchemaError):
        vd.parse_mot("1,1,10,20,30,40,1,12,0,0\n")


def test_groups_map_categories():
    groups = {"person": [1, 2], "vehicle": [4, 5, 6, 9]}
    a = vd.parse_mot(MOT_TXT)
    g = vd.assign_groups(a["category"], groups)
    assert g.tolist() == ["person", "vehicle", "", "person", ""]


def test_ignored_region_filter_drops_boxes_centred_inside():
    regions = np.array([[0.0, 0.0, 60.0, 60.0]])
    boxes = np.array([[10.0, 10.0, 20.0, 20.0], [50.0, 50.0, 30.0, 30.0], [100, 100, 5, 5.0]])
    keep = vd.outside_ignored(boxes, regions)
    assert keep.tolist() == [False, True, True]  # second box centre (65, 65) is outside


def test_ignored_region_uses_box_centre():
    regions = np.array([[0.0, 0.0, 60.0, 60.0]])
    boxes = np.array([[50.0, 50.0, 30.0, 30.0]])  # centre (65, 65): outside
    assert vd.outside_ignored(boxes, regions).tolist() == [True]


def test_to_motchallenge_gt_per_group():
    a = vd.parse_mot(MOT_TXT)
    rows = vd.to_motchallenge_gt(a, {"person": [1, 2], "vehicle": [4, 5, 6, 9]}, "person")
    assert rows == ["1,1,10,20,30,40,1,1,1", "2,1,12,21,30,40,1,1,1"]


def test_yolo_labels_normalised_and_filtered(tmp_path):
    det = "10,20,30,40,1,4,0,0\n0,0,5,5,0,0,0,0\n1,1,2,2,1,11,0,0\n"
    lines = vd.det_to_yolo(det, width=200, height=100)
    assert lines == ["3 0.125000 0.400000 0.150000 0.400000"]  # car (4) -> class 3


def test_parse_det_rejects_wrong_columns():
    with pytest.raises(vd.SchemaError):
        vd.parse_det("1,2,3\n")


def test_average_hash_detects_duplicate_content(tmp_path):
    # Smooth synthetic scene (photos have low-frequency structure; white noise does not).
    yy, xx = np.mgrid[0:64, 0:96]
    base = 120 + 80 * np.sin(xx / 15.0) * np.cos(yy / 11.0) + 0.5 * xx
    img = np.clip(np.stack([base, base * 0.8, 255 - base], axis=-1), 0, 255).astype(np.uint8)
    a, b = tmp_path / "a.jpg", tmp_path / "b.jpg"
    Image.fromarray(img).save(a, quality=95)
    Image.fromarray(img).resize((48, 32)).save(b, quality=70)  # rescaled re-encode
    other = tmp_path / "c.jpg"
    Image.fromarray(255 - img).save(other)
    ha, hb, hc = (vd.average_hash(p) for p in (a, b, other))
    assert vd.hamming(ha, hb) <= 4
    assert vd.hamming(ha, hc) > 20
