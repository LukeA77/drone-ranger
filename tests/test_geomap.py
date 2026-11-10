"""Map layer (Phase 12): positions, ellipses, footprints and the GeoJSON's structure."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import chi2

from aerialrange.evaluate.auair_world import project_ground
from aerialrange.geometry.frames import geodetic_to_ned
from aerialrange.pipeline import geomap
from aerialrange.pipeline.ranging import RangingModel

MODEL = RangingModel(
    1259.0, 960.0, 540.0, np.array([0, 10, 20, 40, 60, 91.0]), np.array([0.2, 0.09, 0.12]), 0.10
)
ORIGIN = (56.2, 10.2, 30.0)


def flight(n_frames: int = 3) -> pd.DataFrame:
    k = np.arange(1, n_frames + 1)
    return pd.DataFrame(
        {
            "frame": k,
            "t_s": (k - 1) * 0.2,
            "n_m": 2.0 * (k - 1),
            "e_m": 0.0,
            "h_m": 30.0,
            "yaw_rad": 0.3,
            "tilt_deg": 45.0,
        }
    )


def tracks(cov=(4.0, 1.0, 2.0), shown=False) -> pd.DataFrame:
    rows = []
    for f in (1, 2, 3):
        for tid, cls, n, e in ((1, "person", 100.0, 0.0), (2, "vehicle", 50.0, 30.0)):
            rows.append(
                {
                    "frame": f,
                    "t_s": (f - 1) * 0.2,
                    "track_id": tid,
                    "class": cls,
                    "valid": True,
                    "flag": "reliable",
                    "range_m": 100.0,
                    "range_sigma_m": 2.0,
                    "speed_ms": 1.5,
                    "heading_deg": 90.0,
                    "speed_shown": shown,
                    "ground_n_m": n + f,
                    "ground_e_m": e,
                    "ground_cov_nn_m2": cov[0],
                    "ground_cov_ne_m2": cov[1],
                    "ground_cov_ee_m2": cov[2],
                }
            )
    rows.append({**rows[0], "frame": 3, "track_id": 3, "valid": False, "ground_n_m": np.nan})
    return pd.DataFrame(rows)


def kinds(gj: dict, kind: str) -> list[dict]:
    return [f for f in gj["features"] if f["properties"]["kind"] == kind]


def test_detection_positions_convert_back_to_the_local_frame() -> None:
    gj = geomap.build_geojson(tracks(), flight(), ORIGIN, MODEL)
    for d in kinds(gj, "detection"):
        lon, lat = d["geometry"]["coordinates"]
        ned = geodetic_to_ned(lat, lon, ORIGIN[2], *ORIGIN)
        tid, f = d["properties"]["track_id"], d["properties"]["frame"]
        want = (100.0 + f, 0.0) if tid == 1 else (50.0 + f, 30.0)
        np.testing.assert_allclose(ned[:2], want, atol=0.02)  # 7 decimals: about 1 cm


def test_invalid_rows_are_left_out() -> None:
    gj = geomap.build_geojson(tracks(), flight(), ORIGIN, MODEL)
    assert {d["properties"]["track_id"] for d in kinds(gj, "detection")} == {1, 2}
    assert len(kinds(gj, "track")) == 2 and len(kinds(gj, "ellipse")) == 2


def test_ellipse_axes_match_the_covariance() -> None:
    cov = np.array([[4.0, 1.0], [1.0, 2.0]])
    a, b, brg = geomap.ellipse_axes(cov)
    w, v = np.linalg.eigh(cov)
    k = np.sqrt(chi2.ppf(0.95, 2))
    assert a == pytest.approx(k * np.sqrt(w[1])) and b == pytest.approx(k * np.sqrt(w[0]))
    major = np.array([np.cos(np.radians(brg)), np.sin(np.radians(brg))])  # north, east
    assert abs(major @ v[:, 1]) == pytest.approx(1.0)


def test_ellipse_ring_is_closed_anticlockwise_and_on_the_ellipse() -> None:
    cov = np.array([[9.0, -2.0], [-2.0, 1.0]])
    ring = geomap.ellipse_ring(10.0, 20.0, cov)
    assert np.allclose(ring[0], ring[-1]) and geomap.signed_area_en(ring) > 0
    d = ring[:-1] - [10.0, 20.0]
    m = np.einsum("ni,ij,nj->n", d, np.linalg.inv(cov), d)  # squared Mahalanobis distance
    np.testing.assert_allclose(m, chi2.ppf(0.95, 2), rtol=1e-9)


def test_calibrated_intervals_scale_the_ellipse() -> None:
    t = tracks()
    t["range_half_68_m"] = 4.0  # twice the raw sigma of 2 m
    raw = geomap.build_geojson(tracks(), flight(), ORIGIN, MODEL)
    cal = geomap.build_geojson(t, flight(), ORIGIN, MODEL)
    a0 = kinds(raw, "detection")[0]["properties"]["ellipse_major_m"]
    a1 = kinds(cal, "detection")[0]["properties"]["ellipse_major_m"]
    assert a1 == pytest.approx(2 * a0)
    assert (
        cal["drone_ranger"]["ellipse_calibrated"] and not raw["drone_ranger"]["ellipse_calibrated"]
    )


def test_speed_and_heading_only_where_the_display_rule_shows_them() -> None:
    hidden = kinds(geomap.build_geojson(tracks(), flight(), ORIGIN, MODEL), "detection")
    shown = kinds(geomap.build_geojson(tracks(shown=True), flight(), ORIGIN, MODEL), "detection")
    assert all(d["properties"]["speed_ms"] is None for d in hidden)
    assert all(d["properties"]["heading_deg"] is None for d in hidden)
    assert all(d["properties"]["speed_ms"] == 1.5 for d in shown)


def test_footprint_starts_at_the_reliable_depression_and_ends_at_the_image_bottom() -> None:
    assert geomap.min_reliable_depression_deg(MODEL) == 10.0
    pose = flight().iloc[0]
    ring = geomap.footprint_ring(pose, MODEL, 1920.0, 1080.0, 10.0)
    assert ring is not None and geomap.signed_area_en(ring) > 0
    u, v = project_ground(
        ring[:-1, 0] - pose["n_m"],
        ring[:-1, 1] - pose["e_m"],
        30.0,
        MODEL.focal_px,
        pose["yaw_rad"],
        45.0,
        MODEL.cx,
        MODEL.cy,
    )
    v_top = MODEL.cy + MODEL.focal_px * np.tan(np.radians(10.0 - 45.0))
    assert np.all((u > -1e-6) & (u < 1920 + 1e-6))
    np.testing.assert_allclose([v.min(), v.max()], [max(v_top, 0.0), 1080.0], atol=1e-6)


def test_footprint_absent_when_every_row_is_too_shallow() -> None:
    pose = flight().iloc[0].copy()
    pose["tilt_deg"] = -60.0  # looking up: no row reaches 10 degrees below the horizon
    assert geomap.footprint_ring(pose, MODEL, 1920.0, 1080.0, 10.0) is None


def test_geojson_follows_rfc7946(tmp_path) -> None:
    gj = geomap.write_map_from_frames(tracks(), flight(), ORIGIN, MODEL, tmp_path)
    assert gj["type"] == "FeatureCollection" and "crs" not in gj
    for f in gj["features"]:
        g = f["geometry"]
        if g["type"] == "Polygon":
            ring = np.array(g["coordinates"][0])
            assert len(ring) >= 4 and np.allclose(ring[0], ring[-1])
            # anticlockwise exterior ring in (longitude, latitude)
            x, y = ring[:, 0], ring[:, 1]
            assert np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]) > 0
        coords = np.array(g["coordinates"], dtype=object)
        assert coords.size  # never empty
    text = (tmp_path / "map.geojson").read_text()
    assert json.loads(text) == gj


def test_viewer_embeds_the_data_and_makes_no_requests_by_default(tmp_path) -> None:
    geomap.write_map_from_frames(tracks(), flight(), ORIGIN, MODEL, tmp_path)
    html = (tmp_path / "map.html").read_text(encoding="utf-8")
    assert "/*__DATA__*/null" not in html and '"kind":"detection"' in html
    script = html[html.index("<script>") :]
    assert "http" not in script.split("const params")[0]  # no fixed URL before the tile option


def test_missing_columns_raise_a_clear_error() -> None:
    with pytest.raises(ValueError, match="re-run or replay"):
        geomap.build_geojson(tracks().drop(columns=["ground_cov_ee_m2"]), flight(), ORIGIN, MODEL)
