"""Build an adjustment straight from instrument files, without CompNet.

* Total station: Leica GSI. Each station record starts a setup. Face II
  readings are reduced to face I (Hz - 180°, zenith 360° - V), as CompNet does.
* GNSS: Leica Infinity ``.asc`` baselines, with their processing covariance.
* Geoid: AUSGeoid2020 from the official PROJ grid, or a supplied table.
* Standard errors: from the instrument settings below. The defaults are the
  values recovered from a real CompNet job. Each reading's figure was
  reproduced to 0.02" or better, and every distance exactly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import geodesy as G
from .adjust import SEC, Network, Sighting, Vector
from .io.gsi import GsiRecord, read_gsi
from .io.infinity import Baseline, gnss_sigmas, read_asc
from .provisional import provisional


@dataclass
class Instrument:
    pointing_h: float = 2.0        # arc-seconds, one direction reading
    pointing_v: float = 4.0        # arc-seconds, one zenith reading
    centring_inst: float = 0.0007  # metres
    centring_target: float = 0.0007
    height_inst: float = 0.001     # metres, error in measuring the instrument height
    height_target: float = 0.001
    edm_const: float = 0.002       # metres
    edm_ppm: float = 2.0

    def sigmas(self, slope: float, zen: float) -> tuple[float, float, float]:
        """(direction rad, zenith rad, distance m) for one reading."""
        h = slope * math.sin(zen)
        c2 = self.centring_inst**2 + self.centring_target**2
        sd_dir = math.sqrt(self.pointing_h**2 + c2 * (G.RHO / h) ** 2)
        hh = self.height_inst**2 + self.height_target**2
        sd_zen = math.sqrt(self.pointing_v**2 + hh * (G.RHO * math.sin(zen) / slope) ** 2)
        sd_dist = math.sqrt((self.edm_const + self.edm_ppm * 1e-6 * slope) ** 2 + c2)
        return sd_dir * SEC, sd_zen * SEC, sd_dist


@dataclass
class GnssModel:
    h_const: float = 0.005
    v_const: float = 0.010
    ppm: float = 2.0


@dataclass
class RawSighting:
    station: str
    target: str
    hz_deg: float
    zen_deg: float
    slope: float | None
    hi: float
    ht: float
    time: str
    ref: str
    setup: int = 0       # index of the GSI station record this reading belongs to
    source: str = ""     # name of the GSI file it came from


def reduce_gsi(records: list[GsiRecord], source: str = "GSI") -> list[RawSighting]:
    out: list[RawSighting] = []
    st: GsiRecord | None = None
    n_station = -1
    for r in records:
        if r.kind == "station":
            st = r
            n_station += 1
            continue
        if st is None or r.hz is None or r.zen is None:
            continue
        hz, zen = r.hz, r.zen
        if zen > 180.0:  # face II
            hz, zen = (hz - 180.0) % 360.0, 360.0 - zen
        out.append(RawSighting(st.point, r.point, hz, zen, r.slope if r.slope else None,
                               st.hi or 0.0, r.ht or 0.0, r.time, f"{source} line {r.line}", n_station, source))
    return out


@dataclass
class RawJob:
    gsi: list[str | Path] = field(default_factory=list)
    asc: list[str | Path] = field(default_factory=list)
    fixed: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    weighted: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = field(default_factory=dict)
    instrument: Instrument = field(default_factory=Instrument)
    # GSI file name -> settings for that file only, e.g. a second instrument or looser centring.
    file_instruments: dict[str, Instrument] = field(default_factory=dict)
    gnss: GnssModel = field(default_factory=GnssModel)
    geoid_grid: str | Path | None = None       # AUSGeoid2020 .tif
    geoid_table: dict[str, float] | None = None
    refraction_sec_per_km: float = 13.6
    exclude: set[str] = field(default_factory=set)   # refs like "JOB.GSI line 883"
    rename: dict[str, str] = field(default_factory=dict)  # field name -> project name, e.g. {"STN_1": "STN1"}
    extra_sightings: list[Sighting] = field(default_factory=list)  # e.g. setups only available from CompNet
    round_like_compnet: bool = False
    epsg: int = 7856


def geoid_function(grid: str | Path | None, table: dict[str, float] | None, epsg: int = 7856):
    tr = None
    if grid is not None:
        import pyproj
        from pyproj import Transformer

        grid = Path(grid)
        pyproj.datadir.append_data_dir(str(grid.parent.resolve()))
        tr = Transformer.from_pipeline(
            "+proj=pipeline +step +proj=axisswap +order=2,1 +step +proj=unitconvert +xy_in=deg +xy_out=rad "
            f"+step +proj=vgridshift +grids={grid.name} +multiplier=1 "
            "+step +proj=unitconvert +xy_in=rad +xy_out=deg +step +proj=axisswap +order=2,1")

    def n_at(name: str, e: float, n: float) -> float:
        if table and name in table:
            return table[name]
        if tr is None:
            if table:
                return float(np.mean(list(table.values())))
            raise ValueError("no geoid: give an AUSGeoid2020 grid or a table")
        lat, lon = G.grid_to_latlon(e, n, epsg)
        _, _, h = tr.transform(math.degrees(float(lat)), math.degrees(float(lon)), 0.0)
        if not math.isfinite(h):
            raise ValueError(f"{name} is outside the geoid grid")
        return float(h)

    return n_at


def build(job: RawJob) -> tuple[Network, dict]:
    from .io.compnet import refraction_k_from_compnet

    k = refraction_k_from_compnet(job.refraction_sec_per_km)
    unknown = sorted(set(job.file_instruments) - {Path(g).name for g in job.gsi})
    if unknown:
        raise ValueError(f"instrument settings given for a file that is not in the job: {', '.join(unknown)}")
    raw: list[RawSighting] = []
    for fi, g in enumerate(job.gsi):
        for r in reduce_gsi(read_gsi(g), Path(g).name):
            r.setup = fi * 100000 + r.setup
            raw.append(r)
    raw = [r for r in raw if r.ref not in job.exclude]
    baselines: list[Baseline] = []
    for a in job.asc:
        baselines += read_asc(a)
    # The same mark can carry a different name in the field file than in the project.
    rn = job.rename
    for r in raw:
        r.station, r.target = rn.get(r.station, r.station), rn.get(r.target, r.target)
    for b in baselines:
        b.frm, b.to = rn.get(b.frm, b.frm), rn.get(b.to, b.to)

    # One setup per GSI station record.
    order = {k: i for i, k in enumerate(dict.fromkeys(r.setup for r in raw))}
    sightings: list[Sighting] = []
    for r in raw:
        sightings.append(Sighting(
            setup=f"G{order[r.setup]:03d}:{r.station}", frm=r.station, to=r.target,
            hz=math.radians(r.hz_deg), zen=math.radians(r.zen_deg), slope=r.slope, hi=r.hi, ht=r.ht,
            sd_dir=0.0, sd_zen=0.0, sd_dist=0.0, ref=r.ref, time=r.time))
    sightings += job.extra_sightings

    vectors: list[Vector] = []
    for b in baselines:
        d = math.sqrt(sum(x * x for x in b.dxyz))
        sh, sv = gnss_sigmas(d, job.gnss.h_const, job.gnss.v_const, job.gnss.ppm)
        if job.round_like_compnet:
            sh, sv = round(sh, 3), round(sv, 3)
        vectors.append(Vector(b.frm, b.to, b.dxyz, sh, sv, ref=b.source, cov_xyz=b.cov))

    n_at = geoid_function(job.geoid_grid, job.geoid_table, job.epsg)
    known = dict(job.fixed)
    for n, (val, _) in job.weighted.items():
        known.setdefault(n, val)
    level_n = float(np.mean([n_at(n, c[0], c[1]) for n, c in known.items()]))
    coords, log = provisional(known, sightings, vectors, n_at, job.epsg, k, level_n=level_n)

    # Standard errors from the instrument settings (CompNet-supplied extras keep theirs).
    for r, s in zip(raw, sightings[: len(sightings) - len(job.extra_sightings)]):
        inst = job.file_instruments.get(r.source, job.instrument)
        slope, zen = s.slope, s.zen
        if slope is None:  # direction only: use the provisional distance
            a, b = coords.get(s.frm), coords.get(s.to)
            slope = math.dist(a, b) if a and b else 100.0
        sd_dir, sd_zen, sd_dist = inst.sigmas(slope, zen)
        if job.round_like_compnet:
            sd_dir = round(sd_dir / SEC, 1) * SEC
            sd_zen = round(sd_zen / SEC, 2) * SEC
            sd_dist = round(sd_dist, 3)
        s.sd_dir, s.sd_zen, s.sd_dist = sd_dir, sd_zen, sd_dist

    names = {s.frm for s in sightings} | {s.to for s in sightings} | {v.frm for v in vectors} | {v.to for v in vectors}
    missing = sorted(n for n in names if n not in coords)
    if missing:
        raise ValueError(f"no starting coordinates could be found for: {', '.join(missing)}")
    geoid = {n: n_at(n, coords[n][0], coords[n][1]) for n in names | set(known)}
    points = {n: c for n, c in coords.items() if n not in job.fixed}
    net = Network(points=points, geoid=geoid, fixed=dict(job.fixed), weighted=dict(job.weighted),
                  sightings=sightings, vectors=vectors, epsg=job.epsg, refraction_k=k)
    info = dict(n_gsi_sightings=len(raw), n_extra=len(job.extra_sightings), n_vectors=len(vectors),
                provisional_log=log, level_n=level_n)
    return net, info

