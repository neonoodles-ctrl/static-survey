"""Coordinates CSV and AutoCAD DXF on the synthetic network: everything lands on the right layer, at the right
place, and a planted wrong point number is drawn in red while clean data draws nothing in red."""

import math
from types import SimpleNamespace

import ezdxf
import numpy as np
import pytest

from static_survey.adjust import adjust
from static_survey.cad import K2_95, coordinate_rows, ellipse_scale, flagged, write_csv, write_dxf
from static_survey.diagnose import diagnose
from static_survey.io.compnet import read_control

from synthetic import TRUE, build

PAIRS = {frozenset((a, b)) for a in TRUE for b in TRUE if a < b}


def _drawn(path):
    doc = ezdxf.readfile(path)
    assert not doc.audit().has_errors
    msp = doc.modelspace()
    by = lambda t, layer: [e for e in msp if e.dxftype() == t and e.dxf.layer == layer]  # noqa: E731
    return doc, by


def test_csv_reads_back_as_control(tmp_path):
    net = build(seed=1)
    res = adjust(net)
    rows = coordinate_rows(net, res)
    write_csv(tmp_path / "c.csv", rows)
    fixed, weighted = read_control(tmp_path / "c.csv")
    assert set(fixed) == {"A", "C"} and set(weighted) == set(TRUE) - {"A", "C"}
    for r in rows:
        enh = fixed[r["name"]] if r["status"] == "fixed" else weighted[r["name"]][0]
        assert np.allclose(enh, (r["e"], r["n"], r["h"]), atol=6e-5)
        if r["status"] != "fixed":
            assert np.allclose(weighted[r["name"]][1], (r["sd_e"], r["sd_n"], r["sd_h"]), atol=6e-5)


def test_ellipse_axes_and_bearing():
    mm = 1e-3
    net = SimpleNamespace(fixed={}, weighted={})
    # 3 mm east, 1 mm north, uncorrelated: long axis east (90°). Then the same turned to bearing 45°.
    c45 = np.array([[5, 4], [4, 5]]) * mm * mm  # eigenvalues 9 and 1 (mm²) along 45° and 135°
    res = SimpleNamespace(vf=0.5, coords={"P": (0, 0, 0), "Q": (10, 10, 0)},
                          cov={"P": np.diag([9 * mm * mm, mm * mm, 4 * mm * mm]),
                               "Q": np.block([[c45, np.zeros((2, 1))], [np.zeros((1, 2)), np.eye(1) * mm * mm]])})
    p, q = coordinate_rows(net, res)
    for r in (p, q):
        assert math.isclose(r["a"], K2_95 * 3 * mm) and math.isclose(r["b"], K2_95 * mm)
    assert math.isclose(p["bearing"], 90.0) and math.isclose(q["bearing"], 45.0)
    assert math.isclose(p["sd_h"], 2 * mm)   # variance factor below 1: not scaled down
    res.vf = 4.0
    assert math.isclose(coordinate_rows(net, res)[0]["sd_h"], 4 * mm)   # above 1: scaled up by 2


def test_dxf_places_marks_lines_and_ellipses(tmp_path):
    net = build(seed=1)
    res = adjust(net)
    rows = coordinate_rows(net, res)
    drawn = write_dxf(tmp_path / "n.dxf", "Synthetic", net, res, rows, [])
    doc, by = _drawn(tmp_path / "n.dxf")
    assert doc.dxfversion == "AC1015" and doc.header["$INSUNITS"] == 6
    xyz = {r["name"]: (r["e"], r["n"], r["h"]) for r in rows}

    ctrl = {tuple(e.dxf.insert) for e in by("INSERT", "SS-CONTROL") if e.dxf.name == "SS_CONTROL"}
    marks = {tuple(e.dxf.insert) for e in by("INSERT", "SS-MARKS") if e.dxf.name == "SS_MARK"}
    assert ctrl == {xyz["A"], xyz["C"]}
    assert marks == {xyz[n] for n in TRUE if n not in ("A", "C")}
    names = {e.dxf.text for e in by("TEXT", "SS-NAMES")}
    assert set(TRUE) <= names and f"RL {xyz['B'][2]:.3f}" in names

    lines = {frozenset((tuple(e.dxf.start), tuple(e.dxf.end))) for e in by("LINE", "SS-TS")}
    assert lines == {frozenset((xyz[a], xyz[b])) for a, b in map(sorted, PAIRS)}
    assert drawn["ts_lines"] == 15 and drawn["gnss_lines"] == 0 and not by("LINE", "SS-GNSS")

    scale = ellipse_scale(rows)
    ells = {tuple(e.dxf.center): e for e in by("ELLIPSE", "SS-ELLIPSES")}
    assert set(ells) == {xyz[n] for n in TRUE if n not in ("A", "C")}
    for r in rows:
        if r["status"] == "adjusted":
            e = ells[xyz[r["name"]]]
            assert math.isclose(e.dxf.major_axis.magnitude, r["a"] * scale, rel_tol=1e-9)
            assert math.isclose(e.dxf.ratio, r["b"] / r["a"], rel_tol=1e-9)
            bearing = math.degrees(math.atan2(e.dxf.major_axis.x, e.dxf.major_axis.y)) % 180
            assert math.isclose(bearing, r["bearing"], abs_tol=1e-6)
    assert any(f"{scale:,.0f} times actual size" in e.dxf.text for e in by("TEXT", "SS-ELLIPSES"))


@pytest.mark.parametrize("seed", [5, 6, 7, 8])
def test_clean_data_draws_nothing_in_red(seed):
    net = build(seed=seed)
    res, findings = diagnose(net)
    assert flagged(net, res, findings) == {}


def test_planted_wrong_point_numbers_are_drawn_in_red(tmp_path):
    net = build(seed=1)
    at_e = [i for i, s in enumerate(net.sightings) if s.setup == "E"]
    a, b = next((i, j) for i, j in zip(at_e, at_e[1:]) if {net.sightings[i].to, net.sightings[j].to} == {"A", "B"})
    net.sightings[a].to, net.sightings[b].to = net.sightings[b].to, net.sightings[a].to
    res, findings = diagnose(net)
    swap = next(f for f in findings if f.kind == "swap")
    rows = coordinate_rows(net, res)
    write_dxf(tmp_path / "n.dxf", "Synthetic", net, res, rows, findings)
    _, by = _drawn(tmp_path / "n.dxf")
    xyz = {r["name"]: (r["e"], r["n"], r["h"]) for r in rows}
    red = {frozenset((tuple(e.dxf.start), tuple(e.dxf.end))) for e in by("LINE", "SS-FLAGGED")}
    assert red == {frozenset((xyz["E"], xyz["A"])), frozenset((xyz["E"], xyz["B"]))}   # and nothing innocent
    labels = [e.dxf.text for e in by("TEXT", "SS-FLAGGED")]
    assert sum(swap.title.split(":")[0] in t for t in labels) >= 2
