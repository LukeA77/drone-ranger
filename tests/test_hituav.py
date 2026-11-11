"""HIT-UAV reading, class mapping and the split by capture day (D-751)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from aerialrange.data import hituav as h


def _write_repo(root: Path, images: dict[str, list[tuple[str, str, list]]]) -> None:
    """images: official split -> [(file name, date, [(hit category, x, y, w, h)])]."""
    ann_dir = root / "normal_json" / "annotations"
    ann_dir.mkdir(parents=True)
    k = 0
    for split in h.SPLITS:
        imgs, anns = [], []
        for name, date, boxes in images.get(split, []):
            imgs.append(
                {"filename": name, "height": 512, "width": 640, "id": k, "date_captured": date}
            )
            for c, x, y, w, hh in boxes:
                anns.append(
                    {
                        "image_id": k,
                        "category_id": c,
                        "bbox": [x, y, w, hh],
                        "id": len(anns),
                        "area": w * hh,
                    }
                )
            k += 1
        (ann_dir / f"{split}.json").write_text(
            json.dumps({"images": imgs, "annotation": anns, "categories": []})
        )


def test_parse_name_reads_the_documented_fields():
    assert h.parse_name("1_130_90_0_04077.jpg") == (True, 130, 90, 0, 4077)
    assert h.parse_name("0_60_30_0_00000.jpg")[0] is False
    with pytest.raises(ValueError):
        h.parse_name("2_60_30_0_00000.jpg")


def test_load_maps_classes_marks_ignored_and_drops_empty_boxes(tmp_path):
    _write_repo(
        tmp_path,
        {
            "train": [
                (
                    "0_60_30_0_00005.jpg",
                    "d1",
                    [
                        (0, 1, 2, 3, 4),
                        (1, 0, 0, 5, 5),
                        (2, 0, 0, 2, 2),
                        (3, 0, 0, 9, 9),
                        (4, 0, 0, 7, 7),
                        (0, 0, 0, 0, 4),
                    ],
                )
            ]
        },
    )
    (im,) = h.load(tmp_path)
    assert im.altitude_m == 60 and im.serial == 5 and not im.night
    got = [(b[0], b[5]) for b in im.boxes]
    assert got == [(1, False), (4, False), (3, False), (4, True), (1, True)]


def test_coco_writes_ignored_regions_as_crowd_and_visdrone_categories(tmp_path):
    _write_repo(
        tmp_path, {"val": [("1_70_40_0_00001.jpg", "d1", [(0, 1, 2, 3, 4), (4, 0, 0, 7, 7)])]}
    )
    c = h.coco(h.load(tmp_path), file_prefix="images/")
    assert c["images"][0]["file_name"] == "images/1_70_40_0_00001.jpg"
    assert [(a["category_id"], a["iscrowd"]) for a in c["annotations"]] == [(1, 0), (1, 1)]
    assert c["annotations"][0]["area"] == 12.0
    assert [x["id"] for x in c["categories"]] == list(range(1, 11))


def _im(date, night, alt, serial):
    return h.Image(
        file=f"{int(night)}_{alt}_30_0_{serial:05d}.jpg",
        official_split="train",
        date=date,
        night=night,
        altitude_m=alt,
        angle_deg=30,
        serial=serial,
        width=640,
        height=512,
        boxes=(),
    )


def test_day_split_keeps_days_whole_and_meets_the_rule():
    ims, s = [], 0
    # day: (number of images, altitudes, has night and day)
    plan = {
        "a": (40, [60, 70], True),
        "b": (30, [60], True),
        "c": (10, [60, 70], True),
        "d": (10, [70], False),
        "e": (20, [60, 70], True),
    }
    for d, (n, alts, both) in plan.items():
        for i in range(n):
            ims.append(_im(d, both and i % 2 == 1, alts[i % len(alts)], s))
            s += 1
    sp = h.day_split(ims)
    assert sorted(sum(sp.values(), [])) == sorted(plan)
    by = {d: k for k, v in sp.items() for d in v}
    for part in ("val", "test"):
        sel = [im for im in ims if by[im.date] == part]
        assert {im.night for im in sel} == {True, False}
    assert {im.altitude_m for im in ims if by[im.date] == "test"} == {60, 70}
    # closest to 70/10/20 under the rule: val c (10 %), test e (20 %)
    assert sp["val"] == ["c"] and sp["test"] == ["e"]


def test_day_split_fails_loudly_when_no_split_qualifies():
    ims = [_im("a", False, 60, 0), _im("b", False, 60, 1), _im("c", False, 60, 2)]
    with pytest.raises(ValueError):
        h.day_split(ims)


def test_nearest_serial_gap_uses_training_images_of_the_same_setting():
    ims = [
        _im("a", False, 60, 10),
        _im("a", False, 70, 11),
        _im("b", False, 60, 14),
        _im("b", False, 80, 15),
    ]
    split_of = {
        ims[0].file: "train",
        ims[1].file: "train",
        ims[2].file: "test",
        ims[3].file: "test",
    }
    assert h.nearest_serial_gap(ims, split_of, "test") == [4]
