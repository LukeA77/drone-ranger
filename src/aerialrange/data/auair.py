"""AU-AIR loading.

Schema and units were read from the file (scripts/inspect_auair.py, results/auair/schema.json;
D-501): `frame_<start stamp>_<x|xx>_<index>.jpg`, where `_xx` continues the `_x` recording;
`time.ms` is milliseconds since the recording start (frames 200 ms apart, 5 fps); altitude is
in millimetres; angles are radians. The longitude key is spelt `longtitude` and the width key
`image_width:` in the file, and are read as spelt. Which angle is roll, pitch or yaw is not
assumed here; the angles are passed through under their file names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from aerialrange.geometry.frames import geodetic_to_ned

_NAME = re.compile(r"frame_(\d{14})_(x{1,2})_(\d{7})\.jpg")


def parse_name(name: str) -> tuple[str, int, int]:
    """(recording start stamp, part 1 or 2, frame index) from an AU-AIR image name."""
    m = _NAME.fullmatch(name)
    if m is None:
        raise ValueError(f"unexpected AU-AIR image name: {name}")
    return m.group(1), len(m.group(2)), int(m.group(3))


def segment_clips(recording: NDArray, t_ms: NDArray, gap_ms: float) -> NDArray[np.int64]:
    """Clip id per frame: a new clip starts at each recording and wherever consecutive frames
    of a recording (in time order) are more than `gap_ms` apart. Ids follow (recording, time)."""
    recording, t_ms = np.asarray(recording), np.asarray(t_ms, float)
    order = np.lexsort((t_ms, recording))
    r, t = recording[order], t_ms[order]
    new = np.ones(len(r), bool)
    new[1:] = (r[1:] != r[:-1]) | (np.diff(t) > gap_ms)
    out = np.empty(len(r), np.int64)
    out[order] = np.cumsum(new) - 1
    return out


def frames_table(root: Path, gap_ms: float) -> pd.DataFrame:
    """One row per annotated frame, sorted by recording and time, with clip ids, altitude in
    metres, and a local NED position (metres) relative to each clip's first frame. Down is
    minus the height gain since that frame, from the recorded altitude. Boxes are kept as JSON
    (top, left, height, width in pixels, class index into the file's categories)."""
    d = json.loads((Path(root) / "annotations.json").read_text())
    rows = []
    for a in d["annotations"]:
        rec, part, idx = parse_name(a["image_name"])
        rows.append(
            {
                "recording": rec,
                "part": part,
                "index": idx,
                "image_name": a["image_name"],
                "image_path": f"images/{a['image_name']}",
                "t_ms": float(a["time"]["ms"]),
                "lat_deg": a["latitude"],
                "lon_deg": a["longtitude"],
                "alt_m": a["altitude"] / 1000.0,
                "angle_phi": a["angle_phi"],
                "angle_theta": a["angle_theta"],
                "angle_psi": a["angle_psi"],
                "linear_x": a["linear_x"],
                "linear_y": a["linear_y"],
                "linear_z": a["linear_z"],
                "n_boxes": len(a["bbox"]),
                "boxes": json.dumps(a["bbox"]),
            }
        )
    df = pd.DataFrame(rows).sort_values(["recording", "t_ms"], kind="stable").reset_index(drop=True)
    df.insert(0, "clip", segment_clips(df["recording"].to_numpy(), df["t_ms"].to_numpy(), gap_ms))
    ned = np.zeros((len(df), 3))
    for _, g in df.groupby("clip"):
        o = g.iloc[0]
        ned[g.index] = geodetic_to_ned(
            g["lat_deg"], g["lon_deg"], g["alt_m"], o["lat_deg"], o["lon_deg"], o["alt_m"]
        )
    df["n_m"], df["e_m"], df["d_m"] = ned[:, 0], ned[:, 1], ned[:, 2]
    return df
