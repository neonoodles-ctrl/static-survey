"""Leica GSI-16 raw data reader.

Each record is one line of words: ``WI`` (2 digits) + 4 info characters +
sign + 16 data characters. Words used here:

  11 point id         21 horizontal circle   22 vertical circle (zenith)
  31 slope distance   51 ppm / prism const   87 target height
  88 instrument ht    84/85/86 station E/N/H 19 date-time (MMDDhhmm)

The last info character of each word is the unit. Angles: 2 = 400 gon,
3 = 360 decimal degrees, 4 = 360 degrees sexagesimal (dddmmsss, last digit in
tenths of a second). Distances: 0 = mm, 6 = 0.1 mm, 8 = 0.01 mm.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

_WORD = re.compile(r"\*?(\d{2})([0-9.]{4})([+-])(\S+)")


def _angle(unit: str, sign: str, digits: str) -> float:
    v = int(digits)
    if unit == "4":    # 360 degrees sexagesimal: dddmmsss, last digit tenths of a second
        a = v // 100000 + (v // 1000) % 100 / 60 + (v % 1000) / 10 / 3600
    elif unit == "3":  # 360 decimal degrees x 1e5
        a = v / 1e5
    elif unit == "2":  # 400 gon x 1e5
        a = v / 1e5 * 0.9
    else:
        raise ValueError(f"unsupported GSI angle unit {unit}")
    return -a if sign == "-" else a


def _dist(unit: str, sign: str, digits: str) -> float:
    v = int(digits)
    scale = {"0": 1e-3, "6": 1e-4, "8": 1e-5}.get(unit)
    if scale is None:
        raise ValueError(f"unsupported GSI distance unit {unit}")
    return (-v if sign == "-" else v) * scale


@dataclass
class GsiRecord:
    line: int
    point: str
    kind: str                 # 'station' or 'obs'
    hz: float | None = None   # degrees
    zen: float | None = None  # degrees
    slope: float | None = None
    ht: float | None = None
    hi: float | None = None
    ppm: float | None = None
    e: float | None = None
    n: float | None = None
    h: float | None = None
    time: str = ""            # MM-DD hh:mm


def read_gsi(path: str | Path) -> list[GsiRecord]:
    out: list[GsiRecord] = []
    for ln, raw in enumerate(Path(path).read_text(encoding="latin-1").replace("\r", "").split("\n"), 1):
        raw = raw.strip()
        if not raw:
            continue
        rec = GsiRecord(line=ln, point="", kind="obs")
        for m in _WORD.finditer(raw):
            wi, info, sign, data = m.groups()
            unit = info[3] if len(info) >= 4 else "0"
            if wi == "11":
                rec.point = data.lstrip("0") or "0"
            elif wi == "21":
                rec.hz = _angle(unit, sign, data)
            elif wi == "22":
                rec.zen = _angle(unit, sign, data)
            elif wi == "31":
                rec.slope = _dist(unit, sign, data)
            elif wi == "87":
                rec.ht = _dist(unit, sign, data)
            elif wi == "88":
                rec.hi = _dist(unit, sign, data)
            elif wi in ("84", "85", "86"):
                val = _dist(unit, sign, data)
                setattr(rec, {"84": "e", "85": "n", "86": "h"}[wi], val)
            elif wi == "51":
                try:
                    rec.ppm = float(int(data[:-4].lstrip("0") or "0"))
                except ValueError:
                    pass
            elif wi == "19":
                d = f"{int(data):08d}"[-8:]
                rec.time = f"{d[0:2]}-{d[2:4]} {d[4:6]}:{d[6:8]}"
        if rec.e is not None and rec.hz is None:
            rec.kind = "station"
        out.append(rec)
    return out


def attach_times(sightings, records: list[GsiRecord], tol_sec: float = 1.5) -> int:
    """Copy instrument timestamps onto sightings by walking both lists in order.

    A sighting matches the next unused GSI observation to the same target whose
    horizontal reading agrees (face II readings differ by 180 degrees) within
    ``tol_sec`` arc-seconds. Returns how many sightings got a time.
    """
    import math

    obs = [r for r in records if r.kind == "obs" and r.hz is not None]
    k = 0
    hits = 0
    for s in sightings:
        if s.hz is None:
            continue
        hz_deg = math.degrees(s.hz)
        for j in range(k, min(k + 60, len(obs))):
            r = obs[j]
            if r.point != s.to:
                continue
            d = (r.hz - hz_deg) % 180.0
            d = min(d, 180.0 - d) * 3600
            if d <= tol_sec:
                s.time = r.time
                k = j + 1
                hits += 1
                break
    return hits
