"""Readers for CompNet 3.x project files.

CompNet keeps one project as many small text files that share a stem, for
example ``JOB.TRV``. The ones read here:

``.TRV``  reduced terrestrial observations, one sighting per line::

              from to HZ(ddd.mmss) slope ZEN(ddd.mmss) 0 HI,HT sd_dir(") sd_dist(m) sd_zen(")

          A line ``0 0 0 0 0 0 0,0 0 0 0`` ends a setup. Lines starting with
          ``*`` are headers (``*GENERAL``, ``*PROJECT name``) or sightings the
          surveyor switched off.
``.GPS``  GNSS vectors: ``from to id dX dY dZ sd_horiz sd_vert`` (metres, ECEF).
``.FIX``  fixed marks: ``num E N H "name"``.
``.STD``  weighted marks: ``num E N H sdE sdN sdH "name"``.
``.WGN``  AUSGeoid N value per point: ``num N``.
``.PTS``  point names: ``num "name"``.
``.VAR``  provisional coordinates (CompNet's starting values).
``.FIN``  final adjusted coordinates.
``.OUT``  the printed adjustment listing.
"""

from __future__ import annotations

import math
import re
import struct
from dataclasses import dataclass, field
from pathlib import Path


def dms_to_deg(value: float | str) -> float:
    """Convert CompNet's packed ddd.mmss angle to decimal degrees."""
    text = f"{float(value):.6f}"
    sign = -1.0 if text.startswith("-") else 1.0
    text = text.lstrip("-")
    deg, frac = text.split(".")
    frac = (frac + "000000")[:6]
    minutes = int(frac[:2])
    seconds = int(frac[2:4]) + int(frac[4:6]) / 100.0
    return sign * (int(deg) + minutes / 60.0 + seconds / 3600.0)


def _lines(path: Path) -> list[str]:
    return path.read_text(encoding="latin-1").replace("\r", "").split("\n")


@dataclass
class Sighting:
    setup: int          # index of the setup this sighting belongs to
    seq: int            # running number of the sighting in the file
    frm: str
    to: str
    hz_deg: float       # horizontal circle reading, degrees
    slope: float        # slope distance, metres
    zen_deg: float      # zenith angle, degrees
    hi: float           # instrument height, metres
    ht: float           # target height, metres
    sd_dir: float       # a-priori sd of direction, arc-seconds
    sd_dist: float      # a-priori sd of slope distance, metres
    sd_zen: float       # a-priori sd of zenith angle, arc-seconds
    line: int           # line number in the .TRV file
    project: str = ""   # the *PROJECT block it came from (one block per imported field file)


@dataclass
class GnssVector:
    seq: int
    frm: str
    to: str
    ident: str
    dxyz: tuple[float, float, float]
    sd_h: float
    sd_v: float
    line: int
    vcv: tuple[float, ...] | None = None   # baseline processing covariance, m^2


@dataclass
class Mark:
    name: str
    e: float
    n: float
    h: float
    sd: tuple[float, float, float] | None = None  # None means held fixed


@dataclass
class CompnetProject:
    stem: Path
    names: dict[int, str] = field(default_factory=dict)
    sightings: list[Sighting] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)   # raw text of switched-off sightings
    vectors: list[GnssVector] = field(default_factory=list)
    control: dict[str, Mark] = field(default_factory=dict)
    geoid: dict[str, float] = field(default_factory=dict)
    provisional: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    final: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    weight_source: dict[str, str] = field(default_factory=dict)

    @property
    def n_setups(self) -> int:
        return len({s.setup for s in self.sightings})


_NAMED = re.compile(r'^\s*(-?\d+)\s+(.*?)\s*"(.*)"\s*$')


def _read_named_coords(path: Path) -> dict[int, tuple[str, list[float]]]:
    out: dict[int, tuple[str, list[float]]] = {}
    for raw in _lines(path):
        m = _NAMED.match(raw)
        if not m:
            continue
        num, body, name = m.groups()
        out[int(num)] = (name, [float(x) for x in body.split()])
    return out


def read_project(stem: str | Path) -> CompnetProject:
    """Read a CompNet project given the path without extension."""
    stem = Path(stem)
    proj = CompnetProject(stem=stem)

    def f(ext: str) -> Path:
        return stem.with_suffix(ext)

    # Point names: .PTS, then any coordinate file that carries names.
    for ext in (".PTS", ".FIX", ".STD", ".VAR", ".FIN"):
        if f(ext).exists():
            for num, (name, _) in _read_named_coords(f(ext)).items():
                proj.names.setdefault(num, name)
    if f(".CSV").exists():
        for raw in _lines(f(".CSV")):
            parts = raw.split(",")
            if len(parts) >= 5 and parts[0].strip().lstrip("-").isdigit():
                proj.names.setdefault(int(parts[0]), parts[4].strip())

    def name(num: str) -> str:
        n = int(num)
        return proj.names.get(n, f"#{n}")

    if f(".FIX").exists():
        for num, (nm, v) in _read_named_coords(f(".FIX")).items():
            proj.control[nm] = Mark(nm, v[0], v[1], v[2], None)
    if f(".STD").exists():
        for num, (nm, v) in _read_named_coords(f(".STD")).items():
            proj.control[nm] = Mark(nm, v[0], v[1], v[2], (v[3], v[4], v[5]))
    for ext, target in ((".VAR", proj.provisional), (".FIN", proj.final)):
        if f(ext).exists():
            for num, (nm, v) in _read_named_coords(f(ext)).items():
                target[nm] = (v[0], v[1], v[2])
    if f(".WGN").exists():
        for raw in _lines(f(".WGN")):
            p = raw.split()
            if len(p) == 2 and p[0].isdigit():
                proj.geoid[name(p[0])] = float(p[1])
    if not proj.geoid and f(".N").exists():
        # .N holds the geoid N at the project's reference mark; use it everywhere.
        for raw in _lines(f(".N")):
            p = raw.split()
            if len(p) == 2 and p[0].lstrip("-").isdigit():
                proj.geoid[name(p[0])] = float(p[1])

    if f(".TRV").exists():
        setup, seq, prev_from, block = 0, 0, None, ""
        for ln, raw in enumerate(_lines(f(".TRV")), 1):
            t = raw.strip()
            if not t:
                continue
            if t.startswith("*"):
                if t[1:2].isdigit():
                    proj.disabled.append(t)
                elif t.upper().startswith("*PROJECT"):
                    block = t.split(None, 1)[1].strip() if len(t.split()) > 1 else ""
                continue
            p = t.split()
            if p[0] == "0" and p[1] == "0":
                setup += 1
                prev_from = None
                continue
            if prev_from is not None and p[0] != prev_from:
                setup += 1
            prev_from = p[0]
            hi, ht = (float(x) for x in p[6].split(","))
            proj.sightings.append(Sighting(
                setup=setup, seq=seq, frm=name(p[0]), to=name(p[1]),
                hz_deg=dms_to_deg(p[2]), slope=float(p[3]), zen_deg=dms_to_deg(p[4]),
                hi=hi, ht=ht, sd_dir=float(p[7]), sd_dist=float(p[8]), sd_zen=float(p[9]),
                line=ln, project=block,
            ))
            seq += 1
        # Renumber setups 0..n-1 in order of appearance.
        remap: dict[int, int] = {}
        for s in proj.sightings:
            s.setup = remap.setdefault(s.setup, len(remap))
        # CompNet adjusts with the standard errors in its per-type files,
        # which are rounded copies of the .TRV values (.DIR to 0.1", .SLP to
        # 1 mm). Use them when they line up with the sightings.
        proj.weight_source = {"direction": ".TRV", "zenith": ".TRV", "distance": ".TRV"}
        for ext, key, col, attr in ((".DIR", "direction", 3, "sd_dir"), (".VRT", "zenith", 5, "sd_zen"),
                                    (".SLP", "distance", 4, "sd_dist")):
            if not f(ext).exists():
                continue
            rows = [l.split() for l in _lines(f(ext)) if l.split() and not l.startswith("*")]
            rows = [r for r in rows if not (r[0] == "0" and r[1] == "0")]
            if len(rows) == len(proj.sightings) and all(
                    proj.names.get(int(r[0])) == s.frm and proj.names.get(int(r[1])) == s.to
                    for r, s in zip(rows, proj.sightings)):
                for r, s in zip(rows, proj.sightings):
                    setattr(s, attr, float(r[col]))
                proj.weight_source[key] = ext

    if f(".GPS").exists():
        seq = 0
        for ln, raw in enumerate(_lines(f(".GPS")), 1):
            t = raw.strip()
            if not t or t.startswith("*"):
                continue
            p = t.split()
            proj.vectors.append(GnssVector(
                seq=seq, frm=name(p[0]), to=name(p[1]), ident=p[2],
                dxyz=(float(p[3]), float(p[4]), float(p[5])),
                sd_h=float(p[6]), sd_v=float(p[7]), line=ln,
            ))
            seq += 1
        # .VCV: the baseline processor's covariance for each vector, in units
        # of 1e-7 m^2 (XX XY XZ YY YZ ZZ), in the same order as .GPS.
        if f(".VCV").exists():
            rows = [l.split() for l in _lines(f(".VCV")) if l.split() and not l.startswith("*")]
            if len(rows) == len(proj.vectors):
                for r, v in zip(rows, proj.vectors):
                    v.vcv = tuple(float(x) * 1e-7 for x in r[2:8])
    return proj


# ---------------------------------------------------------------- .OUT listing

_ANGLE = r"(-?\d+°\s*\d+'\s*\d+\.\d+\")"


def _ang(text: str) -> float:
    m = re.match(r"\s*(-?)(\d+)°\s*(\d+)'\s*(\d+\.\d+)\"", text)
    sign, d, mi, s = m.groups()
    v = int(d) + int(mi) / 60 + float(s) / 3600
    return -v if sign else v


@dataclass
class OutListing:
    directions: list[dict] = field(default_factory=list)
    verticals: list[dict] = field(default_factory=list)
    distances: list[dict] = field(default_factory=list)
    vectors: list[dict] = field(default_factory=list)
    adjusted: dict[str, dict] = field(default_factory=dict)
    fixed: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    # WEIGHTED COORDINATE CORRECTIONS: per weighted mark, the input value and correction (m) per component
    weighted: dict[str, dict] = field(default_factory=dict)
    variance_factor: float | None = None
    group_variance: dict[str, float] = field(default_factory=dict)
    dof: int | None = None
    gnss_params: dict[str, float] = field(default_factory=dict)
    header: list[str] = field(default_factory=list)
    # titles of the observation sections (a title line followed by a FROM ... TO column header)
    obs_sections: list[str] = field(default_factory=list)


def read_out(path: str | Path) -> OutListing:
    """Parse the printed CompNet adjustment listing (.OUT)."""
    lines = _lines(Path(path))
    out = OutListing(header=[l.strip() for l in lines[:9] if l.strip()])
    sec = None
    re_dir = re.compile(rf"\s*(\S+)\s+(\S+)\s+(-?\d+\.\d+)\s+{_ANGLE}\s+{_ANGLE}\s+(-?\d+\.\d+)\s+(\d+\.\d+)(\*?)\s+(\d+\.\d+)")
    re_sd = re.compile(r"\s*(\S+)\s+(\S+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(\d+\.\d+)\s+(-?\d+\.\d+)\s+(\d+\.\d+)")
    i = 0
    while i < len(lines):
        l = lines[i]
        t = l.strip()
        if t and i + 1 < len(lines) and lines[i + 1].strip().startswith("FROM ") and t == t.upper():
            out.obs_sections.append(t)
            sec = None  # a section this reader does not parse ends the one before it
        if t in ("DIRECTIONS", "VERTICAL ANGLES", "SLOPE DISTANCES", "GNSS VECTORS",
                 "FIXED COORDINATES", "ADJUSTED COORDINATES", "WEIGHTED COORDINATE CORRECTIONS"):
            sec = t
            i += 1
            continue
        if t.startswith("GNSS GRAVIMETRIC") or t.startswith("MAXIMUM STANDARDISED"):
            sec = None
        m = re.match(r"(Scale ppm|East slope|North slope|Horizontal rotation):\s*(-?\d+\.\d+)", t)
        if m:
            out.gnss_params[m.group(1)] = float(m.group(2))
        m = re.match(r"([A-Za-z][A-Za-z ]*?):\s*(\d+\.\d+)\s+F TEST", t)
        if m:
            out.group_variance[m.group(1)] = float(m.group(2))
        m = re.match(r"ESTIMATE OF VARIANCE FACTOR:\s*(\d+\.\d+)", t)
        if m:
            out.variance_factor = float(m.group(1))
        m = re.search(r"(\d+) DEGREES OF FREEDOM", t)
        if m:
            out.dof = int(m.group(1))
        if sec in ("DIRECTIONS", "VERTICAL ANGLES"):
            m = re_dir.match(l)
            if m:
                frm, to, corr, obs, fin, res, sres, tag, off = m.groups()
                rec = dict(frm=frm, to=to, corr=float(corr), obs=_ang(obs), fin=_ang(fin),
                           res=float(res), sres=float(sres), offset=float(off), line=i + 1)
                (out.directions if sec == "DIRECTIONS" else out.verticals).append(rec)
        elif sec == "SLOPE DISTANCES":
            m = re_sd.match(l)
            if m:
                frm, to, scale, obs, fin, res, sres = m.groups()
                out.distances.append(dict(frm=frm, to=to, scale=float(scale), obs=float(obs),
                                          fin=float(fin), res=float(res), sres=float(sres), line=i + 1))
        elif sec == "GNSS VECTORS":
            p = l.split()
            if len(p) == 9 and not p[0].replace(".", "").replace("-", "").isdigit():
                q1, q2 = lines[i + 1].split(), lines[i + 2].split()
                out.vectors.append(dict(
                    frm=p[0], to=p[1],
                    obs_xyz=(float(p[2]), float(q1[0]), float(q2[0])),
                    fin_xyz=(float(p[3]), float(q1[1]), float(q2[1])),
                    obs_enh=(float(p[5]), float(q1[3]), float(q2[3])),
                    fin_enh=(float(p[6]), float(q1[4]), float(q2[4])),
                    res_enh=(float(p[7]), float(q1[5]), float(q2[5])),
                    sres_enh=(float(p[8].rstrip("*")), float(q1[6].rstrip("*")), float(q2[6].rstrip("*"))),
                    line=i + 1))
                i += 3
                continue
        elif sec == "WEIGHTED COORDINATE CORRECTIONS":
            # name  E vE SR  N vN SR  H vH SR
            p = t.split()
            if len(p) == 10 and not p[0].lstrip("-").replace(".", "").isdigit():
                try:
                    v = [float(x) for x in p[1:]]
                except ValueError:
                    v = None
                if v:
                    out.weighted[p[0]] = dict(e=v[0], ve=v[1], sr_e=v[2], n=v[3], vn=v[4], sr_n=v[5],
                                              h=v[6], vh=v[7], sr_h=v[8])
        elif sec == "FIXED COORDINATES":
            p = t.split()
            if len(p) == 5 and p[0].isdigit():
                out.fixed[p[4]] = (float(p[1]), float(p[2]), float(p[3]))
        elif sec == "ADJUSTED COORDINATES":
            p = t.split()
            if len(p) == 8 and p[0].isdigit():
                out.adjusted[p[7]] = dict(e=float(p[1]), sd_e=float(p[2]), n=float(p[3]),
                                          sd_n=float(p[4]), h=float(p[5]), sd_h=float(p[6]))
        i += 1
    return out


# ------------------------------------------- what this reader cannot handle

HANDLED_ZONES = (56,)
OBS_SECTIONS = ("DIRECTIONS", "VERTICAL ANGLES", "SLOPE DISTANCES", "GNSS VECTORS")
OBS_GROUPS = ("Direction", "Vertical angle", "Slope distance", "GNSS")
# Project files that are read, and files CompNet writes that hold nothing this reader needs: listings,
# drawings, binary matrices, import copies of the observations, settings. Any other project file that looks
# like observations stops the run, since it may hold levelling, horizontal angles, horizontal distances or
# azimuths, which this reader does not read.
READ_FILES = (".TRV", ".DIR", ".VRT", ".SLP", ".GPS", ".VCV", ".FIX", ".STD", ".WGN", ".N", ".PTS", ".VAR",
              ".FIN", ".CSV", ".OUT")
OTHER_FILES = (".DXF", ".ERR", ".FB", ".GPD", ".INV", ".MRG", ".NC!", ".NCE", ".NCY", ".PAT", ".PT!", ".RES",
               ".VR!", ".WWW")
COPIES = {".GPP": ".GPS"}  # the GNSS vectors again in another form: fine only next to the file that is read
_MGA = re.compile(r"^(MGA\s*(?:94|2020))\s+ZONE\s+(\d+)\s+PROJECTION$")


def listing_projection(path: str | Path) -> str | None:
    """The projection line in a CompNet listing header, e.g. 'MGA2020 ZONE 56 PROJECTION'."""
    for raw in _lines(Path(path))[:20]:
        t = raw.strip()
        if "PROJECTION" in t:
            return t
    return None


def project_zone(stem: str | Path) -> tuple[int | None, str]:
    """The project's MGA zone and where it was found: the .OUT listing header, then the .FB copy of it, then
    the reference position in .GPD (two float32 values, latitude and longitude in radians). A projection that
    is not an MGA zone gives (None, its text); nothing found gives (None, '')."""
    stem = Path(stem)
    for ext in (".OUT", ".FB"):
        p = stem.with_suffix(ext)
        if p.exists():
            text = listing_projection(p)
            if text:
                m = _MGA.match(text)
                return (int(m.group(2)) if m else None), f"{p.name} header: '{text}'"
    g = stem.with_suffix(".GPD")
    if g.exists() and g.stat().st_size >= 8:
        lat, lon = struct.unpack("<2f", g.read_bytes()[:8])
        if abs(lat) <= math.pi / 2 and abs(lon) <= math.pi:
            lon_d = math.degrees(lon)
            return int((lon_d + 180) // 6) + 1, f"{g.name}: reference position at longitude {lon_d:.2f}°"
    return None, ""


def _looks_like_observations(path: Path) -> bool:
    """A text file with lines that start with two point numbers followed by a number, as CompNet's
    observation files do (``from to value ...``)."""
    data = path.read_bytes()
    if b"\0" in data[:4096]:
        return False
    for raw in data.decode("latin-1").splitlines():
        p = raw.replace(",", " ").split()
        if len(p) >= 3 and not p[0].startswith("*") and p[0].lstrip("-").isdigit() and p[1].lstrip("-").isdigit():
            try:
                float(p[2])
                return True
            except ValueError:
                continue
    return False


def unsupported(stem: str | Path, proj: CompnetProject, out: OutListing | None = None) -> tuple[list[str], list[str]]:
    """Reasons not to adjust this project, each a plain sentence, and the project files left unread because
    they hold no observations. ``out`` is the project's .OUT listing when there is one."""
    stem = Path(stem)
    problems: list[str] = []
    unread: list[str] = []
    zone, where = project_zone(stem)
    if zone is None and where:
        problems.append(f"The projection is not an MGA zone ({where}). static-survey only handles MGA zone 56 so far.")
    elif zone is None:
        problems.append("The project does not say which MGA zone it is in: there is no .OUT or .FB listing and no "
                        ".GPD file. static-survey only handles MGA zone 56 so far, and in another zone the "
                        "coordinates would come out wrong.")
    elif zone not in HANDLED_ZONES:
        problems.append(f"The project is in MGA zone {zone} ({where}). static-survey only handles MGA zone 56 so far; "
                        f"in zone {zone} the coordinates would come out wrong.")
    have = {p.suffix.upper(): p for p in stem.parent.glob(stem.name + ".*") if p.is_file()}
    for ext, p in sorted(have.items()):
        if ext in READ_FILES or ext in OTHER_FILES:
            continue
        if ext in COPIES:
            if COPIES[ext] not in have:
                problems.append(f"{p.name} holds observations, but there is no {stem.name}{COPIES[ext]}, "
                                "the file this program reads them from.")
            continue
        if _looks_like_observations(p):
            problems.append(f"{p.name} looks like an observation file (its lines start with two point numbers), and "
                            "this program does not read it. It may hold levelling, horizontal angles, horizontal "
                            "distances or azimuths. Adjusting without them would not be the same job.")
        else:
            unread.append(p.name)
    if out is not None:
        for title in out.obs_sections:
            if title not in OBS_SECTIONS:
                problems.append(f"CompNet's listing has a section of observations this program does not read: "
                                f"{title}. Adjusting without them would not be the same job.")
        for group in out.group_variance:
            if group not in OBS_GROUPS:
                problems.append(f"CompNet adjusted a group of observations this program does not read: {group}.")
        counts = (("directions", len(out.directions), len(proj.sightings)),
                  ("vertical angles", len(out.verticals), len(proj.sightings)),
                  ("slope distances", sum(1 for d in out.distances if d["obs"] > 0),
                   sum(1 for s in proj.sightings if s.slope > 0)),
                  ("GNSS vectors", len(out.vectors), len(proj.vectors)))
        off = [f"{listed} {label} against {read}" for label, listed, read in counts if listed != read]
        if off:
            problems.append(f"CompNet's listing ({stem.name}.OUT) does not match the project files: "
                            + "; ".join(off) + " (listing against files). The listing may be from an earlier run, or "
                            "the project holds observations this program does not read. Run the job in CompNet "
                            "again, or move the .OUT out of the folder.")
    return problems, unread


# ------------------------------------------------------------ to a Network

def refraction_k_from_compnet(sec_per_km: float = 13.6) -> float:
    """CompNet adds a curvature-and-refraction term of ``sec_per_km`` x horizontal
    km to each observed vertical angle. That equals (1 - k) * D / 2R, so k follows."""
    from ..geodesy import R_MEAN, RHO

    curvature = RHO * 1000.0 / (2 * R_MEAN)  # arc-seconds per km
    return 1.0 - sec_per_km / curvature


def to_network(proj: CompnetProject, start: str = "provisional", sec_per_km: float = 13.6):
    """Build an adjustment Network from a CompNet project."""
    import math

    from ..adjust import SEC, Network, Sighting as S2, Vector

    coords = proj.provisional if start == "provisional" else proj.final
    fixed = {n: (m.e, m.n, m.h) for n, m in proj.control.items() if m.sd is None}
    weighted = {n: ((m.e, m.n, m.h), m.sd) for n, m in proj.control.items() if m.sd is not None}
    points = {n: c for n, c in coords.items() if n not in fixed}
    for n, (val, _) in weighted.items():
        points.setdefault(n, val)
    sightings = []
    for s in proj.sightings:
        sightings.append(S2(
            setup=f"{s.setup:03d}:{s.frm}", frm=s.frm, to=s.to,
            hz=math.radians(s.hz_deg), zen=math.radians(s.zen_deg),
            slope=s.slope if s.slope > 0 else None, hi=s.hi, ht=s.ht,
            sd_dir=s.sd_dir * SEC, sd_zen=s.sd_zen * SEC, sd_dist=s.sd_dist,
            ref=f"TRV line {s.line}" + (f" ({s.project})" if s.project else ""),
        ))
    vectors = [Vector(v.frm, v.to, v.dxyz, v.sd_h, v.sd_v, ref=f"GPS line {v.line}", cov_xyz=v.vcv)
               for v in proj.vectors]
    return Network(points=points, geoid=dict(proj.geoid), fixed=fixed, weighted=weighted,
                   sightings=sightings, vectors=vectors, refraction_k=refraction_k_from_compnet(sec_per_km))


def read_control(path: str | Path):
    """Control marks from a CompNet .FIX / .STD file or a CSV of name,E,N,H[,sdE,sdN,sdH][,anything].

    A CSV mark with all three standard deviations above zero is weighted; one with them blank or zero is
    fixed. Text after the numbers (such as the status column of static-survey's own coordinates file) is
    ignored, as are header lines. Returns (fixed, weighted) dictionaries."""
    path = Path(path)
    fixed: dict[str, tuple[float, float, float]] = {}
    weighted: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = {}
    if path.suffix.upper() in (".FIX", ".STD"):
        for _, (name, v) in _read_named_coords(path).items():
            if len(v) >= 6:
                weighted[name] = ((v[0], v[1], v[2]), (v[3], v[4], v[5]))
            else:
                fixed[name] = (v[0], v[1], v[2])
        return fixed, weighted
    for raw in _lines(path):
        parts = [x.strip() for x in raw.split(",")]
        if len(parts) < 4 or raw.lstrip().startswith("#"):
            continue
        try:
            enh = tuple(float(x) for x in parts[1:4])
        except ValueError:
            continue  # header line
        sds = []
        for x in parts[4:7]:
            try:
                sds.append(float(x))
            except ValueError:
                break
        if len(sds) == 3 and min(sds) > 0:
            weighted[parts[0]] = (enh, tuple(sds))
        else:
            fixed[parts[0]] = enh
    return fixed, weighted
