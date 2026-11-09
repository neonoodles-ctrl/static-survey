"""Starting coordinates for an adjustment, computed from the observations.

Works outward from the control:

1. GNSS vectors: a vector from a known mark fixes the other end.
2. Oriented setups: a setup at a known mark that sights at least one known
   mark gets its orientation; every other target with a distance is radiated.
3. Resection: a setup at an unknown mark that sights two or more known marks
   with distances is solved for its position.

Repeats until nothing new is found. The adjustment then refines everything,
so centimetre-level starting values are plenty.
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import geodesy as G
from .adjust import Sighting, Vector


def _xyz(c, n_geoid, epsg):
    lat, lon = G.grid_to_latlon(c[0], c[1], epsg)
    return G.geodetic_to_ecef(lat, lon, c[2] + n_geoid), float(lat), float(lon)


def _grid(xyz, n_geoid, epsg):
    lat, lon, h = G.ecef_to_geodetic(xyz)
    e, n = G.latlon_to_grid(lat, lon, epsg)
    return float(np.atleast_1d(e)[0]), float(np.atleast_1d(n)[0]), float(h[0]) - n_geoid


def provisional(known: dict[str, tuple[float, float, float]], sightings: list[Sighting], vectors: list[Vector],
                geoid, epsg: int = 7856, refraction_k: float = 0.16, level_n: float | None = None,
                max_rounds: int = 50) -> tuple[dict[str, tuple[float, float, float]], list[str]]:
    """Return coordinates for every mark that can be reached, and a log of how each was found.

    ``geoid(name, e, n)`` returns AUSGeoid N for a mark. Trig work uses one N for the
    whole job (``level_n``), as the adjustment does."""
    coords = dict(known)
    log = [f"{n}: control" for n in known]
    by_setup: dict[str, list[Sighting]] = defaultdict(list)
    for s in sightings:
        by_setup[s.setup].append(s)
    first = next(iter(known.values()))
    n0 = level_n if level_n is not None else geoid("", first[0], first[1])

    for _ in range(max_rounds):
        found = 0
        for v in vectors:
            for a, b, sign in ((v.frm, v.to, 1.0), (v.to, v.frm, -1.0)):
                if a in coords and b not in coords:
                    xa, _, _ = _xyz(coords[a], geoid(a, *coords[a][:2]), epsg)
                    xb = xa + sign * np.array(v.dxyz)
                    e, n, _ = _grid(xb, 0.0, epsg)
                    _, _, h = G.ecef_to_geodetic(xb)
                    coords[b] = (e, n, float(h[0]) - geoid(b, e, n))
                    log.append(f"{b}: GNSS vector from {a} ({v.ref})")
                    found += 1
        for setup, ss in by_setup.items():
            st = ss[0].frm
            if st not in coords:
                res = _resect(st, ss, coords, n0, epsg, refraction_k)
                if res is None:
                    continue
                coords[st] = res
                log.append(f"{st}: resection in setup {setup}")
                found += 1
            xs, lat, lon = _xyz(coords[st], n0, epsg)
            R = G.enu_matrix(lat, lon)[0]
            ori = []
            for s in ss:
                if s.to in coords and s.hz is not None and s.to != st:
                    xt, _, _ = _xyz(coords[s.to], n0, epsg)
                    d = R @ (xt - xs)
                    ori.append(math.atan2(d[0], d[1]) - s.hz)
            if not ori:
                continue
            omega = math.atan2(np.mean(np.sin(ori)), np.mean(np.cos(ori)))
            for s in ss:
                if s.to in coords or s.slope is None or s.zen is None or s.hz is None:
                    continue
                az = s.hz + omega
                zen = s.zen + refraction_k * s.slope * math.sin(s.zen) / (2 * G.R_MEAN)
                hd, dv = s.slope * math.sin(zen), s.slope * math.cos(zen)
                enu = np.array([hd * math.sin(az), hd * math.cos(az), dv + s.hi - s.ht])
                xt = xs + R.T @ enu
                e, n, h = _grid(xt, n0, epsg)
                coords[s.to] = (e, n, h)
                log.append(f"{s.to}: radiated from {st} ({s.ref})")
                found += 1
        if not found:
            break
    return coords, log


def _resect(st: str, ss: list[Sighting], coords, n0: float, epsg: int, k: float):
    """Position of an unknown station from known targets with distances (2D, then height)."""
    obs = [s for s in ss if s.to in coords and s.slope is not None and s.zen is not None and s.hz is not None]
    tg = {}
    for s in obs:
        tg.setdefault(s.to, s)
    if len(tg) < 2:
        return None
    pts = list(tg.values())
    (a, b) = pts[0], pts[1]
    pa, pb = np.array(coords[a.to][:2]), np.array(coords[b.to][:2])
    ra, rb = a.slope * math.sin(a.zen), b.slope * math.sin(b.zen)
    dab = np.linalg.norm(pb - pa)
    if dab == 0 or dab > ra + rb or dab < abs(ra - rb):
        return None
    x = (ra**2 - rb**2 + dab**2) / (2 * dab)
    y = math.sqrt(max(ra**2 - x**2, 0.0))
    u = (pb - pa) / dab
    perp = np.array([-u[1], u[0]])
    best = None
    for sgn in (1, -1):
        p = pa + x * u + sgn * y * perp
        # the measured angle between the two targets must match (clockwise from a to b)
        ang_obs = (b.hz - a.hz) % (2 * math.pi)
        aza = math.atan2(*(pa - p)); azb = math.atan2(*(pb - p))
        ang = (azb - aza) % (2 * math.pi)
        err = abs((ang - ang_obs + math.pi) % (2 * math.pi) - math.pi)
        if best is None or err < best[0]:
            best = (err, p)
    p = best[1]
    hs = []
    for s in pts:
        zen = s.zen + k * s.slope * math.sin(s.zen) / (2 * G.R_MEAN)
        hs.append(coords[s.to][2] + s.ht - s.hi - s.slope * math.cos(zen))
    return (float(p[0]), float(p[1]), float(np.mean(hs)))
