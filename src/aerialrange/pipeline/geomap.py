"""Map layer of a run (Phase 12): GeoJSON of tracks, error ellipses and searched ground.

`write_map` reads a run's tracks.csv (with the ground point and covariance columns that
`pipeline/range_speed.py` writes) and its flight-data CSV, converts local north / east
positions to WGS 84 latitude and longitude through the local tangent plane anchored at the
flight's origin fix (`geometry.frames.ned_to_geodetic`, exact), and writes:

- `map.geojson` (RFC 7946: longitude, latitude order; WGS 84; exterior rings anticlockwise),
  with features of kind
  - `track`: each track's reported positions over time (LineString);
  - `detection`: each valid track-frame (Point): class, track number, range and its interval,
    reliability flag, the error ellipse's semi-axes and orientation, and speed and heading
    only where the display rule shows the speed;
  - `ellipse`: each track's error ellipse at its last position (Polygon);
  - `footprint`: the ground the camera saw in each frame, limited to the depressions at which
    ranging can be reliable (Polygon), which Phase 14's search builds on;
  - `drone`: the drone's path (LineString);
- `map.html`, a self-contained viewer of the same data (`viz/map_viewer.html`): no network
  access unless a tile service is given (see that file).

Error ellipse: the point's raw covariance, scaled where the run has calibrated range intervals
by (calibrated 68 % half-width / raw range sigma)^2, so the ellipse agrees with the run's range
intervals (D-737); drawn at `ELLIPSE_LEVEL` (chi-squared with 2 degrees of freedom).
Privacy (build2 R16): people are a class and a track number only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from scipy.stats import chi2

from aerialrange.evaluate.auair_world import ground_offset_oblique
from aerialrange.geometry.frames import ned_to_geodetic
from aerialrange.pipeline.ranging import RangingModel

ELLIPSE_LEVEL = 0.95
ELLIPSE_POINTS = 36
FOOTPRINT_POINTS_PER_EDGE = 16
COORD_DECIMALS = 7  # about 1 cm in latitude
COV_COLS = ("ground_cov_nn_m2", "ground_cov_ne_m2", "ground_cov_ee_m2")
VIEWER = Path(__file__).resolve().parents[1] / "viz" / "map_viewer.html"


def min_reliable_depression_deg(model: RangingModel) -> float:
    """Lowest depression at which a range can be flagged reliable: the lower edge of the first
    depression band whose relative range sigma is within the reliability threshold."""
    for i, rel in enumerate(model.rel_sigma):
        if rel <= model.reliable_max_rel:
            return float(model.edges_deg[i])
    return 90.0


def ellipse_axes(cov: NDArray, level: float = ELLIPSE_LEVEL) -> tuple[float, float, float]:
    """Semi-major and semi-minor axes (m) of the level-confidence ellipse of a 2 x 2 NED
    covariance, and the major axis's bearing (degrees clockwise from north, 0 to 180)."""
    k = float(np.sqrt(chi2.ppf(level, 2)))
    w, v = np.linalg.eigh(np.asarray(cov, float))
    w = np.clip(w, 0.0, None)
    major = v[:, 1]
    bearing = float(np.degrees(np.arctan2(major[1], major[0])) % 180.0)
    return k * float(np.sqrt(w[1])), k * float(np.sqrt(w[0])), bearing


def ellipse_ring(n: float, e: float, cov: NDArray, level: float = ELLIPSE_LEVEL) -> NDArray:
    """Closed ring (ELLIPSE_POINTS + 1, 2) of (north, east) points on the ellipse, anticlockwise
    seen from above with east as x and north as y (RFC 7946's exterior orientation)."""
    k = float(np.sqrt(chi2.ppf(level, 2)))
    w, v = np.linalg.eigh(np.asarray(cov, float))
    a = np.linspace(0.0, 2 * np.pi, ELLIPSE_POINTS + 1)
    unit = np.c_[np.cos(a), np.sin(a)] * np.sqrt(np.clip(w, 0.0, None))
    ne = unit @ v.T * k + [n, e]
    ne[-1] = ne[0]
    return ne if signed_area_en(ne) > 0 else ne[::-1]


def signed_area_en(ne: NDArray) -> float:
    """Shoelace area of a (north, east) ring with east as x and north as y; > 0 anticlockwise."""
    x, y = ne[:, 1], ne[:, 0]
    return 0.5 * float(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))


def footprint_ring(
    pose: pd.Series, model: RangingModel, width_px: float, height_px: float, min_dep_deg: float
) -> NDArray | None:
    """Ground outline (closed (n, 2) ring of north, east) of the image rows whose viewing ray,
    on the image's centre column, is at least `min_dep_deg` below the horizon; None when no
    such row is in view."""
    tilt = float(pose["tilt_deg"])
    if not np.isfinite(tilt) or not np.isfinite(pose["h_m"]):
        return None
    v_lim = model.cy + model.focal_px * np.tan(np.radians(min_dep_deg - tilt))
    top = float(np.clip(v_lim, 0.0, height_px))
    if top >= height_px:
        return None
    m = FOOTPRINT_POINTS_PER_EDGE
    us = np.linspace(0.0, width_px, m)
    vs = np.linspace(top, height_px, m)
    u = np.r_[us, np.full(m, width_px), us[::-1], np.zeros(m)]
    v = np.r_[np.full(m, top), vs, np.full(m, height_px), vs[::-1]]
    dn, de = ground_offset_oblique(
        u, v, float(pose["h_m"]), model.focal_px, float(pose["yaw_rad"]), tilt, model.cx, model.cy
    )
    ok = np.isfinite(dn) & np.isfinite(de)
    if ok.sum() < 3:
        return None
    ne = np.c_[dn[ok] + pose["n_m"], de[ok] + pose["e_m"]]
    ne = np.vstack([ne, ne[:1]])
    return ne if signed_area_en(ne) > 0 else ne[::-1]


def _lonlat(ne: NDArray, origin: tuple[float, float, float], down: float) -> list[list[float]]:
    ned = np.c_[ne, np.full(len(ne), down)]
    lat, lon, _ = ned_to_geodetic(ned, *origin)
    return [
        [round(float(x), COORD_DECIMALS), round(float(y), COORD_DECIMALS)]
        for x, y in zip(lon, lat, strict=True)
    ]


def _num(x: object) -> float | None:
    """JSON number, or null for NaN."""
    try:
        f = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def _bool(x: object) -> bool:
    return str(x).strip().lower() in ("true", "1")


def build_geojson(
    tracks: pd.DataFrame,
    flight: pd.DataFrame,
    origin: tuple[float, float, float],
    model: RangingModel,
    width_px: float | None = None,
    height_px: float | None = None,
    name: str = "run",
) -> dict:
    """GeoJSON FeatureCollection of a run (see the module docstring). `origin` is the local
    frame's origin fix: latitude and longitude (degrees) and ellipsoidal height (m). Ground
    points are placed at the origin's height unless the flight data has `d_m` (local down of the
    drone), in which case they lie `h_m` below the drone."""
    missing = [c for c in ("ground_n_m", "ground_e_m", *COV_COLS) if c not in tracks.columns]
    if missing:
        raise ValueError(
            f"tracks.csv lacks {missing}: re-run or replay it with the current pipeline"
        )
    w = width_px if width_px is not None else 2 * model.cx
    h = height_px if height_px is not None else 2 * model.cy
    fl = flight.set_index("frame")
    has_down = "d_m" in fl.columns

    def ground_down(frame: int) -> float:
        if has_down and frame in fl.index:
            return float(fl.at[frame, "d_m"] + fl.at[frame, "h_m"])
        return 0.0

    feats: list[dict] = []
    ok = tracks["valid"].map(_bool) & tracks["ground_n_m"].notna()
    t = tracks[ok].copy()
    cov = t[list(COV_COLS)].to_numpy(float)
    covs = np.stack([cov[:, [0, 1]], cov[:, [1, 2]]], axis=1)
    calibrated = "range_half_68_m" in t.columns and t["range_half_68_m"].notna().any()
    if calibrated:
        s = (t["range_half_68_m"] / t["range_sigma_m"]).to_numpy(float)
        s = np.where(np.isfinite(s) & (s > 0), s, 1.0)
        covs = covs * (s**2)[:, None, None]
    t["_i"] = np.arange(len(t))
    for tid, g in t.groupby("track_id", sort=True):
        g = g.sort_values("frame")
        ne = g[["ground_n_m", "ground_e_m"]].to_numpy(float)
        downs = [ground_down(int(f)) for f in g["frame"]]
        pts = [_lonlat(ne[i : i + 1], origin, downs[i])[0] for i in range(len(g))]
        cls = str(g["class"].iloc[0])
        if len(pts) >= 2:
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": pts},
                    "properties": {
                        "kind": "track",
                        "track_id": int(tid),
                        "class": cls,
                        "first_t_s": _num(g["t_s"].iloc[0]),
                        "last_t_s": _num(g["t_s"].iloc[-1]),
                        "points": len(pts),
                    },
                }
            )
        for i, (_, r) in enumerate(g.iterrows()):
            a, b, brg = ellipse_axes(covs[int(r["_i"])])
            shown = _bool(r.get("speed_shown", False))
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": pts[i]},
                    "properties": {
                        "kind": "detection",
                        "frame": int(r["frame"]),
                        "t_s": _num(r["t_s"]),
                        "track_id": int(tid),
                        "class": cls,
                        "range_m": _num(r["range_m"]),
                        "range_half_95_m": _num(r.get("range_half_95_m")),
                        "flag": str(r["flag"]),
                        "ellipse_major_m": a,
                        "ellipse_minor_m": b,
                        "ellipse_bearing_deg": brg,
                        "speed_ms": _num(r["speed_ms"]) if shown else None,
                        "heading_deg": _num(r["heading_deg"]) if shown else None,
                    },
                }
            )
        last = g.iloc[-1]
        ring = ellipse_ring(float(ne[-1, 0]), float(ne[-1, 1]), covs[int(last["_i"])])
        feats.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [_lonlat(ring, origin, downs[-1])],
                },
                "properties": {
                    "kind": "ellipse",
                    "track_id": int(tid),
                    "class": cls,
                    "frame": int(last["frame"]),
                    "t_s": _num(last["t_s"]),
                    "level": ELLIPSE_LEVEL,
                },
            }
        )
    min_dep = min_reliable_depression_deg(model)
    path = []
    for frame, pose in fl.sort_index().iterrows():
        if not (np.isfinite(pose["n_m"]) and np.isfinite(pose["e_m"])):
            continue
        down = float(fl.at[frame, "d_m"]) if has_down else 0.0
        path.append(_lonlat(np.array([[pose["n_m"], pose["e_m"]]]), origin, down)[0])
        ring = footprint_ring(pose, model, w, h, min_dep)
        if ring is not None:
            feats.append(
                {
                    "type": "Feature",
                    "geometry": {
                        "type": "Polygon",
                        "coordinates": [_lonlat(ring, origin, ground_down(int(frame)))],
                    },
                    "properties": {
                        "kind": "footprint",
                        "frame": int(frame),
                        "t_s": _num(pose.get("t_s")),
                    },
                }
            )
    if len(path) >= 2:
        feats.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": path},
                "properties": {"kind": "drone"},
            }
        )
    return {
        "type": "FeatureCollection",
        "features": feats,
        "drone_ranger": {
            "name": name,
            "origin_lat_deg": origin[0],
            "origin_lon_deg": origin[1],
            "origin_height_m": origin[2],
            "ellipse_level": ELLIPSE_LEVEL,
            "ellipse_calibrated": bool(calibrated),
            "footprint_min_depression_deg": min_dep,
        },
    }


def write_map_from_frames(
    tracks: pd.DataFrame,
    flight: pd.DataFrame,
    origin: tuple[float, float, float],
    model: RangingModel,
    out: Path,
    name: str | None = None,
) -> dict:
    """Write map.geojson and map.html into `out` from a run's tables; returns the GeoJSON."""
    out = Path(out)
    gj = build_geojson(tracks, flight, origin, model, name=name or out.name)
    out.mkdir(parents=True, exist_ok=True)
    text = json.dumps(gj, separators=(",", ":"))
    (out / "map.geojson").write_text(text, encoding="utf-8")
    html = VIEWER.read_text(encoding="utf-8")
    # embedded, so the page works from a local file with no requests; "</" is escaped so the
    # data cannot close the script element
    html = html.replace("/*__DATA__*/null", text.replace("</", r"<\/"))
    (out / "map.html").write_text(html, encoding="utf-8")
    return gj


def write_map(
    tracks_csv: Path,
    flight_csv: Path,
    origin: tuple[float, float, float],
    model: RangingModel,
    out: Path,
    name: str | None = None,
) -> dict:
    """Write map.geojson and map.html for a run's tracks.csv and flight-data CSV."""
    return write_map_from_frames(
        pd.read_csv(tracks_csv), pd.read_csv(flight_csv), origin, model, out, name
    )
