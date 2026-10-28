"""VisDrone parsing and conversion.

MOT annotation columns (verified on the real files by scripts/check_data.py): frame, target
id, left, top, width, height, score, category, truncation, occlusion. DET annotations have the
same columns without frame and id. Category 0 marks ignored regions; 1 to 11 are pedestrian,
people, bicycle, car, van, truck, tricycle, awning-tricycle, bus, motor, others.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from PIL import Image

CATEGORIES = {
    0: "ignored",
    1: "pedestrian",
    2: "people",
    3: "bicycle",
    4: "car",
    5: "van",
    6: "truck",
    7: "tricycle",
    8: "awning-tricycle",
    9: "bus",
    10: "motor",
    11: "others",
}
IGNORED = 0
OTHERS = 11
# Detector training classes: categories 1..10 map to class indices 0..9 ("others" and
# ignored regions are not trained as classes; the 10 original classes are kept).
TRAIN_CLASSES = list(range(1, 11))

MOT_DTYPE = np.dtype(
    [
        ("frame", int),
        ("id", int),
        ("left", float),
        ("top", float),
        ("width", float),
        ("height", float),
        ("score", int),
        ("category", int),
        ("truncation", int),
        ("occlusion", int),
    ]
)
DET_DTYPE = np.dtype([(n, MOT_DTYPE[n]) for n in MOT_DTYPE.names[2:]])


class SchemaError(ValueError):
    """A VisDrone file does not match the expected column layout."""


def _parse(text: str, dtype: np.dtype) -> NDArray:
    rows = []
    n = len(dtype.names)
    for i, line in enumerate(text.splitlines(), 1):
        f = line.strip().split(",")
        if f == [""]:
            continue
        if f[-1] == "":
            f = f[:-1]
        if len(f) != n:
            raise SchemaError(f"line {i}: {len(f)} columns, expected {n}")
        try:
            vals = tuple(int(x) for x in f)
        except ValueError as e:
            raise SchemaError(f"line {i}: non-integer field") from e
        if vals[dtype.names.index("category")] not in CATEGORIES:
            raise SchemaError(f"line {i}: unknown category")
        rows.append(vals)
    return np.array(rows, dtype=dtype)


def parse_mot(text: str) -> NDArray:
    """Parse one MOT sequence annotation file into a structured array."""
    return _parse(text, MOT_DTYPE)


def parse_det(text: str) -> NDArray:
    """Parse one DET image annotation file into a structured array."""
    return _parse(text, DET_DTYPE)


def assign_groups(categories: NDArray, groups: Mapping[str, Sequence[int]]) -> NDArray:
    """Evaluation group name per box ("" for none), from the config's category lists."""
    out = np.full(len(categories), "", dtype=object)
    for name, cats in groups.items():
        out[np.isin(categories, list(cats))] = name
    return out.astype(str)


def boxes_ltwh(a: NDArray) -> NDArray:
    return np.stack([a["left"], a["top"], a["width"], a["height"]], axis=1).astype(float)


def outside_ignored(boxes_ltwh: NDArray, regions_ltwh: NDArray) -> NDArray[np.bool_]:
    """True for boxes whose centre is outside every ignored region."""
    if len(regions_ltwh) == 0:
        return np.ones(len(boxes_ltwh), bool)
    cx = boxes_ltwh[:, 0] + boxes_ltwh[:, 2] / 2
    cy = boxes_ltwh[:, 1] + boxes_ltwh[:, 3] / 2
    r = regions_ltwh
    inside = (
        (cx[:, None] >= r[None, :, 0])
        & (cx[:, None] <= r[None, :, 0] + r[None, :, 2])
        & (cy[:, None] >= r[None, :, 1])
        & (cy[:, None] <= r[None, :, 1] + r[None, :, 3])
    )
    return ~inside.any(axis=1)


def to_motchallenge_gt(a: NDArray, groups: Mapping[str, Sequence[int]], group: str) -> list[str]:
    """MOTChallenge ground-truth lines (frame,id,x,y,w,h,mark,class,visibility) for one group.
    Ignored regions are handled separately (`outside_ignored`), not written as boxes."""
    sel = a[np.isin(a["category"], list(groups[group]))]
    sel = np.sort(sel, order=["frame", "id"])
    return [
        f"{r['frame']},{r['id']},{int(r['left'])},{int(r['top'])},{int(r['width'])},"
        f"{int(r['height'])},1,1,1"
        for r in sel
    ]


def det_to_yolo(text: str, width: int, height: int) -> list[str]:
    """YOLO label lines (class cx cy w h, normalised) for one DET image; drops ignored regions
    and "others"."""
    a = parse_det(text)
    out = []
    for r in a:
        if r["category"] not in TRAIN_CLASSES or r["width"] <= 0 or r["height"] <= 0:
            continue
        cls = TRAIN_CLASSES.index(int(r["category"]))
        cx = (r["left"] + r["width"] / 2) / width
        cy = (r["top"] + r["height"] / 2) / height
        out.append(f"{cls} {cx:.6f} {cy:.6f} {r['width'] / width:.6f} {r['height'] / height:.6f}")
    return out


def average_hash(path: Path, size: int = 8) -> int:
    """64-bit average hash (greyscale, `size` x `size`), robust to rescaling and re-encoding.
    Used to test the claim that DET images do not overlap the MOT video frames."""
    g = np.asarray(
        Image.open(path).convert("L").resize((size, size), Image.Resampling.BOX), dtype=float
    )
    bits = (g > g.mean()).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return int(bin(a ^ b).count("1"))
