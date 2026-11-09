"""Leica Infinity baseline export (``.asc``) reader.

Each processed baseline is a block of lines::

    @+FROM   X Y Z               reference station, ECEF metres
    @-TO     dX dY dZ            vector, ECEF metres
    @=  s0  qXX qXY qXZ qYY qYZ qZZ   a-posteriori sigma and covariance (m^2)
    @:  ant_from  0              antenna height at the reference station
    @;  ant_to    0              antenna height at the rover
    @*  DD.MM.YYYY hh:mm:ss      start time
    @E  ...                      (unused)

``@#NAME X Y Z code`` lines are point positions and ``@%`` lines are header
notes. Vectors are mark to mark: Infinity has already applied the antenna
heights.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class Baseline:
    frm: str
    to: str
    from_xyz: tuple[float, float, float]
    dxyz: tuple[float, float, float]
    s0: float
    cov: tuple[float, float, float, float, float, float]   # XX XY XZ YY YZ ZZ, m^2
    ant_from: float
    ant_to: float
    start: str
    source: str


def read_asc(path: str | Path) -> list[Baseline]:
    path = Path(path)
    out: list[Baseline] = []
    cur: dict = {}
    for ln, raw in enumerate(path.read_text(encoding="latin-1").replace("\r", "").split("\n"), 1):
        if len(raw) < 2 or raw[0] != "@":
            continue
        tag, body = raw[1], raw[2:]
        p = body.split()
        if tag == "+":
            cur = dict(frm=p[0], from_xyz=tuple(float(x) for x in p[1:4]))
        elif tag == "-" and cur:
            cur.update(to=p[0], dxyz=tuple(float(x) for x in p[1:4]), line=ln)
        elif tag == "=" and cur:
            cur.update(s0=float(p[0]), cov=tuple(float(x) for x in p[1:7]))
        elif tag == ":" and cur:
            cur["ant_from"] = float(p[0])
        elif tag == ";" and cur:
            cur["ant_to"] = float(p[0])
        elif tag == "*" and cur:
            cur["start"] = " ".join(p)
            out.append(Baseline(cur["frm"], cur["to"], cur["from_xyz"], cur["dxyz"], cur.get("s0", 1.0),
                                cur.get("cov", (0.0,) * 6), cur.get("ant_from", 0.0), cur.get("ant_to", 0.0),
                                cur["start"], f"{path.name} line {cur['line']}"))
            cur = {}
    return out


def gnss_sigmas(d_m: float, h_const: float = 0.005, v_const: float = 0.010, ppm: float = 2.0) -> tuple[float, float]:
    """A-priori horizontal and vertical standard deviation of a vector of length d_m.

    Defaults are the values recovered from a real CompNet job: 5 mm + 2 ppm horizontal and
    10 mm + 2 ppm vertical."""
    return h_const + ppm * 1e-6 * d_m, v_const + ppm * 1e-6 * d_m
