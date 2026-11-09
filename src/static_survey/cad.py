"""Files for the surveyor's own software: adjusted coordinates (CSV) and a network drawing (DXF for AutoCAD).

The drawing follows the conventions of CompNet's own DXF so it reads the same in AutoCAD: real MGA2020 eastings
and northings with the AHD height as Z, metres, a two-circle symbol (radii 0.75 m and 1.5 m) on control marks
and a one-circle symbol on adjusted marks, 1 m names. Layer names start with ``SS-`` so they never clash with
CompNet's layers when both drawings are in one file.
"""

from __future__ import annotations

import csv
import math
from datetime import date
from pathlib import Path

import numpy as np
from scipy import stats

from .adjust import Network, Result
from .diagnose import Finding, data_snooping

# 95% confidence: two dimensions for the horizontal ellipse, one for height.
K2_95 = math.sqrt(stats.chi2.ppf(0.95, 2))   # 2.448
K1_95 = stats.norm.ppf(0.975)                 # 1.960
FLAG_ALPHA = 0.01                             # same limit the wrong-point-number check uses

LAYERS = {  # name: (AutoCAD colour index, description)
    "SS-CONTROL": (3, "control marks (held fixed or weighted)"),
    "SS-MARKS": (7, "adjusted marks"),
    "SS-NAMES": (7, "mark names and heights"),
    "SS-TS": (8, "total station lines"),
    "SS-GNSS": (4, "GNSS baselines"),
    "SS-ELLIPSES": (6, "95% error ellipses, enlarged"),
    "SS-FLAGGED": (1, "readings that fail the w-test or belong to a finding"),
}


def sd_scale(res: Result) -> float:
    """Standard deviations are the a-priori ones, scaled up by √(variance factor) when the factor is above 1.

    Scaling down when the factor is below 1 would make the marks look better than the instrument settings
    allow, so that is not done."""
    return math.sqrt(max(1.0, res.vf))


def coordinate_rows(net: Network, res: Result) -> list[dict]:
    """One row per mark: adjusted E, N, H, standard deviations and the 95% horizontal error ellipse."""
    k = sd_scale(res)
    rows = []
    for n in sorted(res.coords):
        e, nn, h = res.coords[n]
        status = "fixed" if n in net.fixed else ("weighted" if n in net.weighted else "adjusted")
        row = dict(name=n, e=e, n=nn, h=h, status=status, sd_e=None, sd_n=None, sd_h=None, a=None, b=None, bearing=None)
        if status != "fixed":
            c = np.array(res.cov[n]) * k * k
            row.update(sd_e=math.sqrt(c[0, 0]), sd_n=math.sqrt(c[1, 1]), sd_h=math.sqrt(c[2, 2]))
            lam, vec = np.linalg.eigh(c[:2, :2])          # ascending
            lam = np.clip(lam, 0.0, None)
            ve, vn = vec[:, 1]
            row.update(a=K2_95 * math.sqrt(lam[1]), b=K2_95 * math.sqrt(lam[0]),
                       bearing=math.degrees(math.atan2(ve, vn)) % 180.0)
        rows.append(row)
    return rows


def write_csv(path: str | Path, rows: list[dict]) -> None:
    """name,E,N,H,sdE,sdN,sdH,status in metres. Fixed marks have no standard deviations.

    The same layout ``read_control`` reads, so one job's adjusted marks can be the next job's weighted control."""
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "E", "N", "H", "sdE", "sdN", "sdH", "status"])
        for r in rows:
            sds = ["" if r[k] is None else f"{r[k]:.4f}" for k in ("sd_e", "sd_n", "sd_h")]
            w.writerow([r["name"], f"{r['e']:.4f}", f"{r['n']:.4f}", f"{r['h']:.4f}", *sds, r["status"]])


def flagged(net: Network, res: Result, findings: list[Finding]) -> dict[frozenset, dict]:
    """Mark pairs with a reading set aside by data snooping, or named by a wrong-point-number or pointing finding.

    Data snooping (the same run the wrong-point-number check uses) sets the worst failing reading aside and
    re-adjusts until nothing fails, so one bad reading does not turn its innocent neighbours red. The w shown
    is the reading's w when it was set aside. Returns {frozenset((from, to)): dict(w, kind, label, frm, to)},
    keeping the worst reading of each pair."""
    from .report import _explained_by

    snooped = data_snooping(net, res, FLAG_ALPHA)
    hits = {si: (w, kind) for si, w, kind in (snooped[0] if snooped else [])}
    named: dict[int, str] = {}
    for f in findings:
        if f.kind in ("swap", "pointing"):
            for si in f.evidence.get("si", []):
                named.setdefault(si, f.title)
    worst_row: dict[int, dict] = {}
    for r in res.rows:
        if r["kind"] in ("DIR", "ZEN", "DIST") and not math.isnan(r["w"]) and r["w"] > worst_row.get(r["si"], {"w": -1})["w"]:
            worst_row[r["si"]] = r
    out: dict[frozenset, dict] = {}
    for si in set(hits) | set(named):
        s = net.sightings[si]
        if s.frm == s.to:
            continue
        row = worst_row.get(si)
        w, kind = hits.get(si) or ((row["w"], row["kind"]) if row else (float("nan"), ""))
        key = frozenset((s.frm, s.to))
        if key in out and not (w > out[key]["w"]):
            continue
        why = named.get(si) or (next((f.title for f in findings if row and _explained_by(row, [f])), None))
        label = _ascii(f"{kind} w {w:.1f}" + (f" - {why}" if why else ""))
        out[key] = dict(w=w, kind=kind, label=label, frm=s.frm, to=s.to)
    return out


def _ascii(text: str) -> str:
    """Arrows in finding titles as plain characters, so every CAD program shows them."""
    return text.replace("↔", "<->").replace("→", "->").replace("×", "x").replace("—", "-")


def _nice_below(x: float) -> float:
    """Largest 1, 2 or 5 × 10^k not above x."""
    if x <= 0:
        return 1.0
    p = 10 ** math.floor(math.log10(x))
    return max(m * p for m in (1, 2, 5) if m * p <= x)


def ellipse_scale(rows: list[dict]) -> float:
    """Enlargement (a round number) that draws the median ellipse at about half the median distance between
    neighbouring marks, so ellipses are readable along a traverse without swamping it."""
    axes = [r["a"] for r in rows if r["a"]]
    if not axes or len(rows) < 2:
        return 1.0
    p = np.array([[r["e"], r["n"]] for r in rows])
    d = np.hypot(p[:, None, 0] - p[None, :, 0], p[:, None, 1] - p[None, :, 1])
    np.fill_diagonal(d, np.inf)
    spacing = float(np.median(d.min(axis=1)))
    return _nice_below(0.5 * spacing / float(np.median(axes)))


def write_dxf(path: str | Path, job: str, net: Network, res: Result, rows: list[dict],
              findings: list[Finding]) -> dict:
    """AutoCAD 2000 DXF of the adjusted network. Returns what was drawn (counts and the ellipse enlargement)."""
    import ezdxf
    from ezdxf import zoom
    from ezdxf.enums import TextEntityAlignment

    doc = ezdxf.new("R2000", setup=True)
    doc.header["$INSUNITS"] = 6          # metres
    doc.header["$MEASUREMENT"] = 1       # metric
    for name, (colour, desc) in LAYERS.items():
        doc.layers.add(name, color=colour).description = desc
    sym_ctrl = doc.blocks.new("SS_CONTROL")
    sym_ctrl.add_circle((0, 0), 0.75)
    sym_ctrl.add_circle((0, 0), 1.5)
    sym_mark = doc.blocks.new("SS_MARK")
    sym_mark.add_circle((0, 0), 0.75)
    msp = doc.modelspace()
    xyz = {r["name"]: (r["e"], r["n"], r["h"]) for r in rows}

    for r in rows:
        p = xyz[r["name"]]
        layer = "SS-MARKS" if r["status"] == "adjusted" else "SS-CONTROL"
        msp.add_blockref("SS_MARK" if r["status"] == "adjusted" else "SS_CONTROL", p, dxfattribs={"layer": layer})
        msp.add_point(p, dxfattribs={"layer": layer})
        msp.add_text(r["name"], height=1.0, dxfattribs={"layer": "SS-NAMES"}).set_placement(
            (p[0] + 2.0, p[1] + 0.2, p[2]), align=TextEntityAlignment.BOTTOM_LEFT)
        msp.add_text(f"RL {r['h']:.3f}", height=0.6, dxfattribs={"layer": "SS-NAMES"}).set_placement(
            (p[0] + 2.0, p[1] - 0.2, p[2]), align=TextEntityAlignment.TOP_LEFT)

    flags = flagged(net, res, findings)
    ts_pairs = {frozenset((s.frm, s.to)) for s in net.sightings if s.frm != s.to and s.frm in xyz and s.to in xyz}
    for pair in sorted(ts_pairs, key=sorted):
        a, b = sorted(pair)
        msp.add_line(xyz[a], xyz[b], dxfattribs={"layer": "SS-TS"})
    gnss_pairs = {frozenset((v.frm, v.to)) for v in net.vectors if v.frm in xyz and v.to in xyz}
    for pair in sorted(gnss_pairs, key=sorted):
        a, b = sorted(pair)
        msp.add_line(xyz[a], xyz[b], dxfattribs={"layer": "SS-GNSS"})
    for pair, f in sorted(flags.items(), key=lambda kv: -kv[1]["w"]):
        a, b = xyz[f["frm"]], xyz[f["to"]]
        msp.add_line(a, b, dxfattribs={"layer": "SS-FLAGGED"})
        mid = tuple((a[i] + b[i]) / 2 for i in range(3))
        msp.add_text(f"{f['frm']}-{f['to']} {f['label']}", height=1.0, dxfattribs={"layer": "SS-FLAGGED"}).set_placement(
            mid, align=TextEntityAlignment.BOTTOM_CENTER)

    scale = ellipse_scale(rows)
    n_ell = 0
    for r in rows:
        if not r["a"]:
            continue
        t = math.radians(r["bearing"])
        major = (r["a"] * scale * math.sin(t), r["a"] * scale * math.cos(t), 0.0)
        msp.add_ellipse(xyz[r["name"]], major_axis=major, ratio=max(min(r["b"] / r["a"], 1.0), 1e-6),
                        dxfattribs={"layer": "SS-ELLIPSES"})
        n_ell += 1

    es = [r["e"] for r in rows]; ns = [r["n"] for r in rows]
    x0, y0 = min(es), min(ns) - 0.05 * max(max(ns) - min(ns), 50.0)
    msp.add_text(f"{job} - adjusted by static-survey {date.today().isoformat()} - MGA2020 zone 56, AHD heights, metres",
                 height=2.0, dxfattribs={"layer": "SS-NAMES"}).set_placement((x0, y0), align=TextEntityAlignment.BOTTOM_LEFT)
    msp.add_text(f"Error ellipses: 95% confidence, drawn {scale:,.0f} times actual size",
                 height=1.5, dxfattribs={"layer": "SS-ELLIPSES"}).set_placement((x0, y0 - 4.0),
                                                                                 align=TextEntityAlignment.BOTTOM_LEFT)
    # Open zoomed to the marks. Set from coordinates, not zoom.extents, which measures text with the system
    # fonts and writes a font cache on the user's machine.
    pad = 0.05 * max(max(es) - min(es), max(ns) - min(ns), 50.0)
    lo, hi = (min(es) - pad, y0 - 6.0 - pad, min(r["h"] for r in rows)), (max(es) + pad, max(ns) + pad, max(r["h"] for r in rows))
    doc.header["$EXTMIN"], doc.header["$EXTMAX"] = lo, hi
    zoom.window(msp, lo[:2], hi[:2])
    doc.saveas(path)
    return dict(marks=len(rows), ts_lines=len(ts_pairs), gnss_lines=len(gnss_pairs), flagged=len(flags),
                ellipses=n_ell, ellipse_scale=scale)
