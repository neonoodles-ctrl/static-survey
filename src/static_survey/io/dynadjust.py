"""Readers for DynAdjust (Geoscience Australia / ICSM) jobs, used as an outside check.

DynAdjust keeps a job as a DNA station file (``.stn``), a DNA measurement file (``.msr``) and an NTv2 geoid
grid (``.gsb``) holding the geoid-ellipsoid separation N and the deflections of the vertical. Its result is
the adjustment listing (``.adj``). Formats follow the DynAdjust User's Guide v1.3, Appendix B.1 (DNA) and
C.8 (ADJ); column numbers below are the guide's, 1-based.

Only what a DynAdjust check needs is read. A measurement type this reader does not handle (direction sets
D, baseline clusters X, ellipsoid arcs and chords, latitudes, longitudes, ellipsoid heights, astronomic
positions) stops it with a message rather than being left out.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

HANDLED = set("ABGHKLMSVYZ")


def hp_to_deg(text: str) -> float:
    """DynAdjust 'HP notation' ddd.mmssss... to decimal degrees, keeping every digit."""
    t = text.strip()
    sign = -1.0 if t.startswith("-") else 1.0
    t = t.lstrip("+-")
    deg, _, frac = t.partition(".")
    frac = (frac + "0000")
    minutes = int(frac[:2])
    seconds = float(frac[2:4] + "." + frac[4:]) if len(frac) > 4 else float(frac[2:4])
    return sign * (int(deg or 0) + minutes / 60 + seconds / 3600)


def _dms(deg: str, minutes: str, seconds: str) -> float:
    d = deg.strip()
    sign = -1.0 if d.startswith("-") else 1.0
    return sign * (abs(int(d)) + int(minutes) / 60 + float(seconds) / 3600)


def _f(s: str) -> float | None:
    s = s.strip()
    return float(s) if s else None


@dataclass
class Station:
    name: str
    constraint: str      # three characters, F free / C constrained, in the coordinate order
    ctype: str           # UTM, LLH, LLh or XYZ
    a: float             # easting / latitude (degrees) / x
    b: float             # northing / longitude (degrees) / y
    c: float             # height (orthometric unless LLh) / z
    zone: int | None
    description: str = ""


@dataclass
class Measurement:
    kind: str
    stations: list[str]
    line: int
    ignore: bool = False
    value: float | None = None          # angles in degrees, linear in metres
    sd: float | None = None             # angles in arc-seconds, linear in metres
    hi: float = 0.0
    ht: float = 0.0
    # GNSS baselines (G) and point clusters (Y)
    ctype: str = ""                     # Y: XYZ or LLH
    values: np.ndarray | None = None    # 3n components (LLH: lat, lon in degrees, H in metres)
    cov: np.ndarray | None = None       # 3n x 3n (LLH: radians^2 for lat/lon, metres^2 for H)
    scale: tuple[float, float, float, float] = (1.0, 1.0, 1.0, 1.0)
    frame: str = ""


@dataclass
class Job:
    stations: dict[str, Station]
    measurements: list[Measurement]
    frame: str = ""


def read_stn(path: str | Path) -> dict[str, Station]:
    out: dict[str, Station] = {}
    for raw in Path(path).read_text(encoding="latin-1").splitlines():
        if not raw.strip() or raw.startswith(("!", "*")):
            continue
        name = raw[0:20].strip()
        ctype = raw[24:27].strip()
        if ctype in ("LLH", "LLh"):
            a, b = hp_to_deg(raw[27:47]), hp_to_deg(raw[47:67])
        else:
            a, b = float(raw[27:47]), float(raw[47:67])
        zone_txt = raw[87:90].strip().lstrip("NnSs")
        out[name] = Station(name, raw[20:23].strip(), ctype, a, b, float(raw[67:87]),
                            int(zone_txt) if zone_txt else None, raw[91:].strip())
    return out


def _cov_lines(lines: list[str], start: int, n: int):
    """n GNSS components (each three lines) with their covariance blocks, from ``start``.
    Returns values (3n), the full symmetric 3n x 3n matrix and the next line index."""
    vals = np.zeros(3 * n)
    cov = np.zeros((3 * n, 3 * n))
    i = start
    names = []
    for p in range(n):
        if p > 0:  # each further point of a cluster starts with a line naming it
            names.append(lines[i][2:22].strip())
            i += 1
        for r in range(3):
            l = lines[i + r]
            vals[3 * p + r] = l[62:82]
            for cidx, (lo, hi) in enumerate(((82, 102), (102, 122), (122, 142))[: r + 1]):
                cov[3 * p + r, 3 * p + cidx] = cov[3 * p + cidx, 3 * p + r] = float(l[lo:hi])
        i += 3
        for q in range(p + 1, n):  # covariance blocks C(p, q), three lines each
            for r in range(3):
                l = lines[i + r]
                for cidx, (lo, hi) in enumerate(((82, 102), (102, 122), (122, 142))):
                    cov[3 * p + r, 3 * q + cidx] = cov[3 * q + cidx, 3 * p + r] = float(l[lo:hi])
            i += 3
    return vals, cov, i, names


def read_msr(path: str | Path) -> list[Measurement]:
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    out: list[Measurement] = []
    i = 0
    while i < len(lines):
        l = lines[i]
        if not l.strip() or l.startswith(("!", "*")):
            i += 1
            continue
        kind, ignore, s1 = l[0], l[1:2] == "*", l[2:22].strip()
        if kind not in HANDLED:
            raise ValueError(f"{Path(path).name} line {i + 1}: measurement type {kind} is not read by static-survey "
                             "yet, so the job can't be adjusted the same way.")
        m = Measurement(kind, [s1], i + 1, ignore)
        if kind == "A":
            m.stations += [l[22:42].strip(), l[42:62].strip()]
            m.value, m.sd = _dms(l[76:80], l[80:82], l[82:90]), float(l[90:99])
            i += 1
        elif kind in "BKVZ":
            m.stations.append(l[22:42].strip())
            m.value, m.sd = _dms(l[76:80], l[80:82], l[82:90]), float(l[90:99])
            m.hi, m.ht = _f(l[99:106]) or 0.0, _f(l[106:113]) or 0.0
            i += 1
        elif kind in "MSL":
            m.stations.append(l[22:42].strip())
            m.value, m.sd = float(l[62:76]), float(l[90:99])
            m.hi, m.ht = _f(l[99:106]) or 0.0, _f(l[106:113]) or 0.0
            i += 1
        elif kind == "H":
            m.value, m.sd = float(l[62:76]), float(l[90:99])
            i += 1
        elif kind == "G":
            m.stations.append(l[22:42].strip())
            m.scale = tuple(_f(l[a:a + 10]) or 1.0 for a in (62, 72, 82, 92))
            m.frame = l[102:122].strip()
            m.values, m.cov, i, _ = _cov_lines(lines, i + 1, 1)
        elif kind == "Y":
            m.ctype = l[22:42].strip()
            n = int(l[42:62])
            m.scale = tuple(_f(l[a:a + 10]) or 1.0 for a in (62, 72, 82, 92))
            m.frame = l[102:122].strip()
            m.values, m.cov, i, more = _cov_lines(lines, i + 1, n)
            m.stations += more
            if m.ctype == "LLH":
                for p in range(n):
                    m.values[3 * p] = hp_to_deg(lines_value(m.values[3 * p]))
                    m.values[3 * p + 1] = hp_to_deg(lines_value(m.values[3 * p + 1]))
        if m.scale[1:] != (1.0, 1.0, 1.0):
            raise ValueError(f"{Path(path).name} line {m.line}: north-south, east-west or vertical scaling of a GNSS "
                             "variance matrix is not handled yet.")
        out.append(m)
    return out


def lines_value(v: float) -> str:
    """A packed dd.mmssss value read as a float, back to text for hp_to_deg (17 significant digits)."""
    return repr(float(v))


# ------------------------------------------------------------------ geoid grid

@dataclass
class GeoidGrid:
    """One NTv2 sub grid of N (m), xi and eta (deflections, north-south and east-west, arc-seconds)."""
    s_lat: float
    n_lat: float
    e_long: float      # NTv2 longitudes are positive west, in the grid's units
    w_long: float
    lat_inc: float
    long_inc: float
    nodes: np.ndarray  # rows from south to north, each from east to west: (rows, cols, 3)
    unit: float        # degrees per grid unit (seconds: 1/3600)

    def at(self, lat_deg: float, lon_deg: float, method: str = "bicubic") -> tuple[float, float, float]:
        """N, xi, eta at a point (degrees, longitude east positive). DynAdjust interpolates bicubically by
        default (User's Guide 5.3.2: Press et al.'s bicubic with central-difference gradients); bilinear is 5.3.1."""
        y = (lat_deg / self.unit - self.s_lat) / self.lat_inc
        x = (-lon_deg / self.unit - self.e_long) / self.long_inc
        rows, cols = self.nodes.shape[:2]
        if not (0 <= y <= rows - 1 and 0 <= x <= cols - 1):
            raise ValueError(f"point {lat_deg:.6f}, {lon_deg:.6f} is outside the geoid grid")
        r, c = min(int(y), rows - 2), min(int(x), cols - 2)
        fy, fx = y - r, x - c
        g = self.nodes
        if method == "bilinear":
            v = (g[r, c] * (1 - fx) * (1 - fy) + g[r, c + 1] * fx * (1 - fy)
                 + g[r + 1, c] * (1 - fx) * fy + g[r + 1, c + 1] * fx * fy)
            return float(v[0]), float(v[1]), float(v[2])
        return self._dynadjust_bicubic(lat_deg / self.unit, -lon_deg / self.unit)

    def _dynadjust_bicubic(self, lat: float, lon: float) -> tuple[float, float, float]:
        """Port of DynAdjust's InterpolateNvalue_BiCubic (dnageoid.cpp, Apache-2.0), with its choice of the 16
        nodes. Within two grid intervals of an edge it swaps outer nodes for inner ones ("virtual cells"),
        which changes the gradients there; a plain bicubic differs from DynAdjust by up to 5 mm in N on the
        sample grid. ``lat`` and ``lon`` are in grid units, longitude positive west."""
        eps = 1e-5
        cols = self.nodes.shape[1]
        flat = self.nodes.reshape(-1, 3).astype(float)
        if abs(lat - self.n_lat) < eps or abs(lon - self.w_long) < eps:
            raise ValueError("point on the north or west limit of the geoid grid is not handled")
        r = int((lat - self.s_lat + eps) / self.lat_inc)
        c = int((lon - self.e_long + eps) / self.long_inc)
        A = r * cols + c + 1  # 1-based record numbers; A lower east, B lower west, C upper east, D upper west
        B, C = A + 1, A + cols
        D = C + 1
        I, J, K, L = A - 1, B + 1, C - 1, D + 1
        E = I - cols
        F, G, H = E + 1, E + 2, E + 3
        M = K + cols
        N, O, P = M + 1, M + 2, M + 3

        def near(v, limit, inc):
            buf = abs(v - limit) - inc
            return buf < eps or buf < inc

        lo_lat, up_lat = near(lat, self.s_lat, self.lat_inc), near(lat, self.n_lat, self.lat_inc)
        lo_lon, up_lon = near(lon, self.e_long, self.long_inc), near(lon, self.w_long, self.long_inc)
        if lo_lat and lo_lon:
            I, K, M = A, C, N
            E, F, G, H = I, A, B, J
        elif lo_lat and up_lon:
            J, L, P = B, D, O
            E, F, G, H = I, A, B, J
        elif lo_lat:
            E, F, G, H = I, A, B, J
        elif up_lat and lo_lon:
            K, I, E = C, A, F
            M, N, O, P = K, C, D, L
        elif up_lat and up_lon:
            L, J, P = D, B, O
            M, N, O, P = K, C, D, L
        elif up_lat:
            M, N, O, P = K, C, D, L
        elif lo_lon:
            E, I, K, M = F, A, C, N
        elif up_lon:
            H, J, L, P = G, B, D, O
        recs = (A, B, C, D, E, F, G, H, I, J, K, L, M, N, O, P)
        if min(recs) < 1 or max(recs) > len(flat):
            raise ValueError("point is too close to the geoid grid edge for bicubic interpolation")
        a, b, c_, d, e, f, g_, h, i, j, k, l_, m, n, o, p = (flat[x - 1] for x in recs)
        lon_lower = self.e_long + c * self.long_inc
        t = abs(lon - (lon_lower + self.long_inc)) / self.long_inc
        u = abs(lat - (self.s_lat + r * self.lat_inc)) / self.lat_inc
        xv = np.array([b, a, c_, d,
                       (a - j) / 2, (i - b) / 2, (k - d) / 2, (c_ - l_) / 2,
                       (d - g_) / 2, (c_ - f) / 2, (n - a) / 2, (o - b) / 2,
                       (c_ - f - l_ + h) / 4, (k - e - d + g_) / 4, (m - i - o + b) / 4, (n - a - p + j) / 4])
        cc = (_BICUBIC @ xv).reshape(4, 4, 3)
        val = np.zeros(3)
        for q in range(3, -1, -1):
            val = t * val + ((cc[q, 3] * u + cc[q, 2]) * u + cc[q, 1]) * u + cc[q, 0]
        return float(val[0]), float(val[1]), float(val[2])


# Press et al.'s bicubic coefficient matrix, as printed in the DynAdjust User's Guide (5.3.2).
_BICUBIC = np.array([
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
    [-3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1, 0, 0, 0, 0],
    [2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0],
    [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
    [0, 0, 0, 0, -3, 0, 0, 3, 0, 0, 0, 0, -2, 0, 0, -1],
    [0, 0, 0, 0, 2, 0, 0, -2, 0, 0, 0, 0, 1, 0, 0, 1],
    [-3, 3, 0, 0, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, -3, 3, 0, 0, -2, -1, 0, 0],
    [9, -9, 9, -9, 6, 3, -3, -6, 6, -6, -3, 3, 4, 2, 1, 2],
    [-6, 6, -6, 6, -4, -2, 2, 4, -3, 3, 3, -3, -2, -1, -1, -2],
    [2, -2, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 2, -2, 0, 0, 1, 1, 0, 0],
    [-6, 6, -6, 6, -3, -3, 3, 3, -4, 4, 2, -2, -2, -2, -1, -1],
    [4, -4, 4, -4, 2, 2, -2, -2, 2, -2, -2, 2, 1, 1, 1, 1]], float)


def read_gsb(path: str | Path) -> GeoidGrid:
    b = Path(path).read_bytes()

    def rec(i):
        return b[i:i + 8].decode("latin-1").strip(), b[i + 8:i + 16]

    i, head = 0, {}
    while i + 16 <= len(b):
        key, val = rec(i)
        i += 16
        head[key] = val
        if key == "GS_COUNT":
            break
    if struct.unpack("<i", head["NUM_FILE"][:4])[0] != 1:
        raise ValueError("geoid grids with more than one sub grid are not read yet")
    d = {k: struct.unpack("<d", head[k])[0] for k in ("S_LAT", "N_LAT", "E_LONG", "W_LONG", "LAT_INC", "LONG_INC")}
    count = struct.unpack("<i", head["GS_COUNT"][:4])[0]
    rows = round((d["N_LAT"] - d["S_LAT"]) / d["LAT_INC"]) + 1
    cols = round((d["W_LONG"] - d["E_LONG"]) / d["LONG_INC"]) + 1
    if rows * cols != count:
        raise ValueError(f"geoid grid: {rows} x {cols} does not match {count} nodes")
    vals = np.frombuffer(b[i:i + 16 * count], "<f4").reshape(rows, cols, 4)[:, :, :3].astype(float)
    unit = 1 / 3600 if head["GS_TYPE"].decode("latin-1").strip().upper().startswith("SECOND") else 1.0
    return GeoidGrid(d["S_LAT"], d["N_LAT"], d["E_LONG"], d["W_LONG"], d["LAT_INC"], d["LONG_INC"], vals, unit)


# ------------------------------------------------------------------ .adj listing

@dataclass
class AdjListing:
    stats: dict[str, float] = field(default_factory=dict)
    rows: list[dict] = field(default_factory=list)          # adjusted measurements, in listing order
    coords: dict[str, dict] = field(default_factory=dict)   # adjusted coordinates per station


_ANG = re.compile(r"^\s*(-?\d+)\s+(\d+)\s+(\d+\.\d+)")


def read_adj(path: str | Path) -> AdjListing:
    lines = Path(path).read_text(encoding="latin-1").splitlines()
    out = AdjListing()
    for l in lines:
        for key, pat in (("unknowns", r"Number of unknown parameters\s+(\d+)"),
                         ("measurements", r"Number of measurements\s+(\d+)"),
                         ("dof", r"Degrees of freedom\s+(\d+)"), ("chi2", r"Chi squared\s+([\d.]+)"),
                         ("sigma0", r"Rigorous Sigma Zero\s+([\d.]+)")):
            m = re.match(pat, l.strip())
            if m and key not in out.stats:
                out.stats[key] = float(m.group(1))
    sec = None
    for l in lines:
        t = l.strip()
        if t == "Adjusted Measurements":
            sec = "msr"
            continue
        if t == "Adjusted Coordinates":
            sec = "crd"
            continue
        if not t or t.startswith(("-", "M Station", "Station ")):
            continue
        if sec == "msr" and l[0] in HANDLED and l[1] == " ":
            rest = l[62:]
            comp = rest[:6].replace("*", "").strip()
            body = rest[6:]
            vals = []
            pos = 0
            for _ in range(2):  # measured, adjusted: an angle (d m s) or one number
                m = _ANG.match(body[pos:])
                if m and l[0] in "ABKVZY" and comp not in ("H",):
                    vals.append(_dms(m.group(1), m.group(2), m.group(3)))
                    pos += m.end()
                else:
                    m = re.match(r"^\s*(-?\d+\.\d+)", body[pos:])
                    vals.append(float(m.group(1)))
                    pos += m.end()
            nums = body[pos:].split()
            out.rows.append(dict(kind=l[0], stations=[l[2:22].strip(), l[22:42].strip(), l[42:62].strip()],
                                 comp=comp, measured=vals[0], adjusted=vals[1], correction=float(nums[0]),
                                 meas_sd=float(nums[1]), adj_sd=float(nums[2]), corr_sd=float(nums[3]),
                                 nstat=float(nums[4]), pelzer=float(nums[5]), pre_corr=float(nums[6]),
                                 outlier=len(nums) > 7))
        elif sec == "crd":
            p = l.split()
            if len(p) >= 15:
                out.coords[p[0]] = dict(constraint=p[1], lat=hp_to_deg(p[2]), lon=hp_to_deg(p[3]), H=float(p[4]),
                                        h=float(p[5]), xyz=(float(p[6]), float(p[7]), float(p[8])),
                                        sd_enu=(float(p[9]), float(p[10]), float(p[11])))
    return out


def read_job(stn: str | Path, msr: str | Path) -> Job:
    return Job(read_stn(stn), read_msr(msr))


def mga_epsg(zone: int) -> int:
    """GDA2020 / MGA zone EPSG code. GDA94 coordinates use the same projection on the same ellipsoid, so a
    GDA94 job is computed with these codes without any datum shift."""
    if not 46 <= zone <= 59:
        raise ValueError(f"zone {zone} is not an MGA zone")
    return 7800 + zone


def _llh_jacobian(lat: float, lon: float, h: float) -> np.ndarray:
    """d(X, Y, Z) / d(latitude, longitude, height), latitude and longitude in radians."""
    from .. import geodesy as G

    sl, cl, so, co = math.sin(lat), math.cos(lat), math.sin(lon), math.cos(lon)
    w2 = 1 - G.E2 * sl * sl
    nu = G.A / math.sqrt(w2)
    rho = G.A * (1 - G.E2) / w2 ** 1.5
    return np.array([[-(rho + h) * sl * co, -(nu + h) * cl * so, cl * co],
                     [-(rho + h) * sl * so, (nu + h) * cl * co, cl * so],
                     [(rho + h) * cl, 0.0, sl]])


def to_network(job: Job, geoid: GeoidGrid, free_sd: float = 10.0, fixed_sd: float = 1e-6,
               axes: tuple[str, str, str] = ("e", "n", "u")):
    """A Network with DynAdjust's conventions: positions on the ellipsoid through each station's own N, angles
    and zeniths related to the ellipsoid normal through the deflections of the vertical, no refraction term,
    no GNSS scale or rotation, GNSS covariances as given, constrained station components held at ``fixed_sd``
    and free ones at ``free_sd`` (DynAdjust defaults 0.001 mm and 10 m). ``axes`` are the local axes the three
    constraint characters refer to."""
    from .. import geodesy as G
    from ..adjust import SEC, Constraint, Network, Obs, Sighting, Vector

    zones = {s.zone for s in job.stations.values() if s.ctype == "UTM"}
    if len(zones) != 1:
        raise ValueError(f"stations are in more than one UTM zone, or none: {sorted(z for z in zones if z)}")
    epsg = mga_epsg(zones.pop())
    points, geoid_n, dov, refs = {}, {}, {}, {}
    for s in job.stations.values():
        if s.ctype == "UTM":
            e, n = s.a, s.b
            lat, lon = (float(x) for x in G.grid_to_latlon(e, n, epsg))
        elif s.ctype in ("LLH", "LLh"):
            lat, lon = math.radians(s.a), math.radians(s.b)
            e, n = (float(x) for x in G.latlon_to_grid(lat, lon, epsg))
        else:
            raise ValueError(f"station {s.name}: {s.ctype} coordinates are not read yet")
        nn, xi, eta = geoid.at(math.degrees(lat), math.degrees(lon))
        height = s.c - nn if s.ctype == "LLh" else s.c
        points[s.name] = (e, n, height)
        geoid_n[s.name] = nn
        dov[s.name] = (xi * SEC, eta * SEC)
        refs[s.name] = tuple(G.geodetic_to_ecef(lat, lon, height + nn))
    sightings, vectors, obs = [], [], []
    rad = math.radians
    for m in job.measurements:
        if m.ignore:
            continue
        ref = f"MSR line {m.line}"
        a = m.stations
        if m.kind == "S":
            sightings.append(Sighting(setup="", frm=a[0], to=a[1], hz=None, zen=None, slope=m.value, hi=m.hi, ht=m.ht,
                                      sd_dir=SEC, sd_zen=SEC, sd_dist=m.sd, ref=ref))
        elif m.kind == "V":
            sightings.append(Sighting(setup="", frm=a[0], to=a[1], hz=None, zen=rad(m.value), slope=None, hi=m.hi,
                                      ht=m.ht, sd_dir=SEC, sd_zen=m.sd * SEC, sd_dist=1.0, ref=ref))
        elif m.kind == "G":
            c = m.cov * m.scale[0]
            vectors.append(Vector(a[0], a[1], tuple(m.values), 0.0, 0.0, ref=ref,
                                  cov_xyz=(c[0, 0], c[0, 1], c[0, 2], c[1, 1], c[1, 2], c[2, 2]), cov_given=True))
        elif m.kind == "Y":
            npt = len(a)
            if m.ctype == "XYZ":
                value, cov = np.array(m.values, float), m.cov * m.scale[0]
            elif m.ctype == "LLH":
                value = np.zeros(3 * npt)
                T = np.zeros((3 * npt, 3 * npt))
                for q in range(npt):
                    la, lo, hh = rad(m.values[3 * q]), rad(m.values[3 * q + 1]), m.values[3 * q + 2]
                    hh = hh + geoid_n[a[q]]  # orthometric to ellipsoidal with the station's N, as DynAdjust does
                    value[3 * q:3 * q + 3] = G.geodetic_to_ecef(la, lo, hh)
                    T[3 * q:3 * q + 3, 3 * q:3 * q + 3] = _llh_jacobian(la, lo, hh)
                cov = T @ (m.cov * m.scale[0]) @ T.T
            else:
                raise ValueError(f"{ref}: GNSS point cluster type {m.ctype} is not read yet")
            obs.append(Obs("POS", a, value, cov=cov, ref=ref))
        elif m.kind in "ABKZ":
            kind = {"A": "ANGLE", "B": "AZ", "K": "ASTRO", "Z": "VANGLE"}[m.kind]
            obs.append(Obs(kind, a, rad(m.value), m.sd * SEC, hi=m.hi, ht=m.ht, ref=ref))
        elif m.kind in "MLH":
            obs.append(Obs({"M": "MSL", "L": "LEVEL", "H": "HEIGHT"}[m.kind], a, m.value, m.sd, ref=ref))
    constraints = []
    for s in job.stations.values():
        for ch, axis in zip(s.constraint, axes):
            held = ch.upper() == "C"
            constraints.append(Constraint(s.name, axis, refs[s.name], fixed_sd if held else free_sd, held))
    return Network(points=points, geoid=geoid_n, fixed={}, sightings=sightings, vectors=vectors, epsg=epsg,
                   refraction_k=0.0, gnss_scale=False, gnss_rotation=False, level_frame=False,
                   measurements=obs, constraints=constraints, dov=dov)
