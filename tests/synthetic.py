"""A small made-up network near (297500 E, 6436000 N) MGA56 with exact observations."""
from __future__ import annotations

import math

import numpy as np

from static_survey import geodesy as G
from static_survey.adjust import SEC, Network, Sighting

TRUE = {
    "A": (297000.0, 6436000.0, 150.0),
    "B": (297400.0, 6436060.0, 162.0),
    "C": (297820.0, 6436010.0, 141.0),
    "D": (297760.0, 6435600.0, 128.0),
    "E": (297310.0, 6435640.0, 147.0),
    "F": (297560.0, 6435810.0, 135.0),
}
N0 = 28.0
K = 0.13


def _geom(coords):
    names = list(coords)
    e = np.array([coords[n][0] for n in names]); n_ = np.array([coords[n][1] for n in names])
    h = np.array([coords[n][2] for n in names]) + N0
    xyz, _, lat, lon = G.grid_ecef(e, n_, h)
    R = G.enu_matrix(lat, lon)
    return {nm: (xyz[i], R[i]) for i, nm in enumerate(names)}


def observe(a, b, hi=1.5, ht=1.6, k=K, geom=None):
    geom = geom or _geom(TRUE)
    xa, Ra = geom[a]
    xb, Rb = geom[b]
    d = Ra @ ((xb + ht * Rb[2]) - (xa + hi * Ra[2]))
    rho = math.hypot(d[0], d[1])
    az = math.atan2(d[0], d[1])
    zen = math.atan2(rho, d[2]) - k * rho / (2 * G.R_MEAN)
    return az, zen, float(np.linalg.norm(d))


def build(seed=1, twist=None, refraction=None, reps=4, noise=True):
    """twist=(station, n_first, arcsec); refraction=((a, b), dk) applied both ways."""
    rng = np.random.default_rng(seed)
    geom = _geom(TRUE)
    sd_dir, sd_zen, sd_dist = 2.0 * SEC, 3.0 * SEC, 0.002
    sightings = []
    for st in TRUE:
        omega = rng.uniform(0, 2 * math.pi)
        targets = [t for t in TRUE if t != st]
        count = 0
        for rnd in range(reps):
            for t in targets:
                k = K
                if refraction and {st, t} == set(refraction[0]):
                    k = K + refraction[1]
                az, zen, dist = observe(st, t, k=k, geom=geom)
                hz = az - omega
                if twist and st == twist[0] and count < twist[1]:
                    hz += twist[2] * SEC
                minute = 8 * 60 + count + (40 if twist and st == twist[0] and count >= twist[1] else 0)
                if noise:
                    hz += rng.normal(0, sd_dir); zen += rng.normal(0, sd_zen); dist += rng.normal(0, sd_dist)
                sightings.append(Sighting(setup=st, frm=st, to=t, hz=hz % (2 * math.pi), zen=zen, slope=dist,
                                          hi=1.5, ht=1.6, sd_dir=sd_dir, sd_zen=sd_zen, sd_dist=sd_dist,
                                          ref=f"{st}-{rnd}", time=f"04-09 {minute // 60:02d}:{minute % 60:02d}"))
                count += 1
    start = {n: (c[0] + rng.normal(0, 0.05), c[1] + rng.normal(0, 0.05), c[2] + rng.normal(0, 0.05))
             for n, c in TRUE.items() if n not in ("A", "C")}
    return Network(points=start, geoid={n: N0 for n in TRUE}, fixed={"A": TRUE["A"], "C": TRUE["C"]},
                   sightings=sightings, refraction_k=K)


def _dms_word(deg: float) -> int:
    tenths = round(deg * 36000)  # tenths of a second
    d, rem = divmod(tenths, 36000)
    m, s = divmod(rem, 600)
    return d * 100000 + m * 1000 + s


def write_gsi(path, stations=("A", "B", "D"), reps=2, hi=1.5, ht=1.6):
    """Leica GSI-16 file of exact observations, alternating face I and face II."""
    geom = _geom(TRUE)
    lines, no = [], 1
    for st in stations:
        e, n, h = TRUE[st]
        lines.append(f"*11{no:04d}+{st:0>16} 84..10+{round(e * 1000):016d} 85..10+{round(n * 1000):016d} "
                     f"86..10+{round(h * 1000):016d} 88..10+{round(hi * 1000):016d} ")
        no += 1
        omega = 0.3 + 0.1 * no
        for rnd in range(reps):
            for t in [x for x in TRUE if x != st]:
                az, zen, dist = observe(st, t, hi, ht, geom=geom)
                hz = math.degrees(az - omega) % 360
                z = math.degrees(zen)
                if rnd % 2:  # face II
                    hz, z = (hz + 180) % 360, 360 - z
                lines.append(f"*11{no:04d}+{t:0>16} 21.324+{_dms_word(hz):016d} 22.324+{_dms_word(z):016d} "
                             f"31..00+{round(dist * 1000):016d} 87..10+{round(ht * 1000):016d} "
                             f"19....+{(4 * 1000000 + 9 * 10000 + 800 + no % 60):016d} ")
                no += 1
    path.write_text("\r\n".join(lines) + "\r\n", encoding="latin-1")


def write_asc(path, pairs=(("A", "E"),), epsg=7856):
    """Infinity .asc export with exact vectors between true marks (TRUE read as grid coordinates in ``epsg``)."""
    names = list(TRUE)
    e = np.array([TRUE[n][0] for n in names]); n_ = np.array([TRUE[n][1] for n in names])
    h = np.array([TRUE[n][2] for n in names]) + N0
    lat, lon = G.grid_to_latlon(e, n_, epsg)
    xyz = dict(zip(names, G.geodetic_to_ecef(lat, lon, h)))
    out = ["@%Unit:                m", "@%Coordinate type:     Cartesian"]
    for a, b in pairs:
        d = xyz[b] - xyz[a]
        out += [f"@+{a:<16}{xyz[a][0]:16.4f}{xyz[a][1]:14.4f}{xyz[a][2]:14.4f}",
                f"@-{b:<16}{d[0]:14.4f}{d[1]:14.4f}{d[2]:14.4f}",
                "@=    0.5000   0.00000010  -0.00000005   0.00000006   0.00000008  -0.00000003   0.00000012",
                "@:       1.5000        0.0000 ", "@;       1.6000        0.0000 ", "@*01.04.2023 09:00:00",
                "@E      0.0000       0.0000       0.0000       0.0000"]
    path.write_text("\r\n".join(out) + "\r\n", encoding="latin-1")
