"""Rigorous 3D least squares adjustment of a survey control network.

Each mark is carried as MGA grid E, N and AHD height H. Every observation is
modelled in Earth-centred XYZ, so there is no projection or curvature
approximation in the geometry:

* direction:   geodetic azimuth from the instrument, minus a setup orientation
* zenith:      angle from the instrument's ellipsoid normal, minus refraction k*D/2R
* slope:       straight-line distance between instrument and target
* GNSS vector: XYZ difference, optionally with one scale and one rotation
               about the network's vertical (as CompNet estimates)

Heights of instrument and target are applied along each mark's ellipsoid
normal. AHD is turned into ellipsoidal height with each mark's AUSGeoid N.

The solution is Gauss-Newton on whitened observation equations. Everything is
dense: networks of a few hundred marks solve in well under a second.

Extra parameters (``Extra``) let the diagnosis layer test causes: a
refraction coefficient on some sightings, an additive distance constant, a
distance scale, or a target height error.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import geodesy as G

SEC = 1.0 / G.RHO  # one arc-second in radians


@dataclass
class Sighting:
    setup: str
    frm: str
    to: str
    hz: float | None          # circle reading, radians
    zen: float | None         # zenith angle, radians
    slope: float | None       # metres
    hi: float
    ht: float
    sd_dir: float             # radians
    sd_zen: float             # radians
    sd_dist: float            # metres
    ref: str = ""             # where it came from, e.g. "TRV line 112"
    time: str = ""            # instrument timestamp when known


@dataclass
class Vector:
    frm: str
    to: str
    dxyz: tuple[float, float, float]
    sd_h: float
    sd_v: float
    ref: str = ""
    # Processing covariance from the baseline software (m^2, upper triangle
    # XX XY XZ YY YZ ZZ). Only its correlations are used: they are rotated
    # into local east/north/up and rescaled to sd_h and sd_v, as CompNet does.
    cov_xyz: tuple[float, float, float, float, float, float] | None = None


def _vector_cov(v: "Vector", r_enu: np.ndarray) -> np.ndarray:
    target = np.array([v.sd_h, v.sd_h, v.sd_v])
    if v.cov_xyz is None:
        return r_enu.T @ np.diag(target**2) @ r_enu
    xx, xy, xz, yy, yz, zz = v.cov_xyz
    cx = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
    ce = r_enu @ cx @ r_enu.T
    s = np.sqrt(np.diag(ce))
    corr = ce / np.outer(s, s)
    return r_enu.T @ (corr * np.outer(target, target)) @ r_enu


@dataclass
class Extra:
    """An extra unknown that models a possible cause."""
    kind: str                 # 'refraction' | 'dist_const' | 'dist_scale' | 'ht_offset' | 'orient_drift'
    members: list[int]        # sighting indices it applies to
    label: str = ""
    values: list[float] | None = None  # per-member factor, e.g. minutes since setup start (orient_drift)


@dataclass
class Network:
    points: dict[str, tuple[float, float, float]]          # provisional E, N, H
    geoid: dict[str, float]
    fixed: dict[str, tuple[float, float, float]]            # held exactly
    weighted: dict[str, tuple[tuple[float, float, float], tuple[float, float, float]]] = field(default_factory=dict)
    sightings: list[Sighting] = field(default_factory=list)
    vectors: list[Vector] = field(default_factory=list)
    epsg: int = 7856
    refraction_k: float = 0.16
    gnss_scale: bool = True
    gnss_rotation: bool = True
    # Total stations level to the plumb line, so trig heighting measures AHD
    # height differences. True (default): sightings use one N for the whole
    # job (no geoid tilt between marks); GNSS vectors use each mark's own N.
    # False: sightings use per-mark N too (zenith from the ellipsoid normal).
    level_frame: bool = True
    # Values used for GNSS scale (fraction) and rotation (radians) when they
    # are not estimated, e.g. to replay another program's transformation.
    gnss_fixed: dict[str, float] = field(default_factory=dict)


@dataclass
class Result:
    coords: dict[str, tuple[float, float, float]]
    sd: dict[str, tuple[float, float, float]]
    rows: list[dict]
    dof: int
    vf: float
    group_vf: dict[str, float]
    group_vf_by_count: dict[str, float]
    orientations: dict[str, float]
    gnss: dict[str, float]
    extras: list[dict]
    iterations: int
    converged: bool


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def adjust(net: Network, extras: list[Extra] | None = None, max_iter: int = 20, tol: float = 1e-6) -> Result:
    extras = extras or []
    names = sorted(set(net.points) | set(net.fixed) | set(net.weighted))
    idx = {n: i for i, n in enumerate(names)}
    free = [n for n in names if n not in net.fixed]
    col = {}
    c = 0
    for n in free:
        col[n] = c
        c += 3
    setups = list(dict.fromkeys(s.setup for s in net.sightings if s.hz is not None))
    ocol = {s: c + i for i, s in enumerate(setups)}
    c += len(setups)
    gcol = {}
    if net.vectors and net.gnss_scale:
        gcol["scale"] = c
        c += 1
    if net.vectors and net.gnss_rotation:
        gcol["rotation"] = c
        c += 1
    xcol = {i: c + i for i in range(len(extras))}
    c += len(extras)
    nunk = c

    enh = np.array([net.fixed.get(n) or net.points.get(n) or net.weighted[n][0] for n in names], float)
    nvals = np.array([net.geoid.get(n, np.nan) for n in names])
    if np.isnan(nvals).all():
        raise ValueError("no geoid N for any mark; supply at least one")
    if np.isnan(nvals).any():  # borrow N from the nearest mark that has one
        have = ~np.isnan(nvals)
        for i in np.where(~have)[0]:
            d = np.hypot(*(enh[have, :2] - enh[i, :2]).T)
            nvals[i] = nvals[have][np.argmin(d)]

    # orientation start values
    params_o = np.zeros(len(setups))
    gpar = {"scale": net.gnss_fixed.get("scale", 0.0), "rotation": net.gnss_fixed.get("rotation", 0.0)}
    xpar = np.zeros(len(extras))

    sight_extra: dict[int, list[int]] = {}
    extra_val: dict[tuple[int, int], float] = {}
    for k, ex in enumerate(extras):
        for pos, m in enumerate(ex.members):
            sight_extra.setdefault(m, []).append(k)
            extra_val[(k, m)] = ex.values[pos] if ex.values is not None else 1.0

    n_obs = sum((s.hz is not None) + (s.zen is not None) + (s.slope is not None) for s in net.sightings)
    n_obs += 3 * len(net.vectors) + 3 * len(net.weighted)

    converged = False
    first = True
    for it in range(1, max_iter + 1):
        h_ell = enh[:, 2] + nvals
        xyz_g, jac_g, lat, lon = G.grid_ecef(enh[:, 0], enh[:, 1], h_ell, net.epsg)
        if net.level_frame:
            xyz, jac, _, _ = G.grid_ecef(enh[:, 0], enh[:, 1], enh[:, 2] + float(np.mean(nvals)), net.epsg)
        else:
            xyz, jac = xyz_g, jac_g
        R = G.enu_matrix(lat, lon)
        up = R[:, 2]
        centre = xyz_g.mean(0)
        clat, clon, _ = G.ecef_to_geodetic(centre)
        u0 = G.enu_matrix(clat, clon)[0, 2]

        if first:
            acc: dict[str, list[float]] = {}
            for s in net.sightings:
                if s.hz is None:
                    continue
                i, j = idx[s.frm], idx[s.to]
                d = R[i] @ ((xyz[j] + s.ht * up[j]) - (xyz[i] + s.hi * up[i]))
                acc.setdefault(s.setup, []).append(_wrap(np.arctan2(d[0], d[1]) - s.hz))
            for k, sname in enumerate(setups):
                a = np.array(acc[sname])
                params_o[k] = np.arctan2(np.sin(a).mean(), np.cos(a).mean())
            first = False

        A = np.zeros((n_obs, nunk))
        w = np.zeros(n_obs)          # whitened misclosure (obs - model)
        meta: list[dict] = []
        r = 0

        def put(i_pt, grad_xyz, row):
            n = names[i_pt]
            if n in col:
                A[row, col[n]:col[n] + 3] += grad_xyz @ jac[i_pt]

        for si, s in enumerate(net.sightings):
            i, j = idx[s.frm], idx[s.to]
            ht = s.ht
            for k in sight_extra.get(si, []):
                if extras[k].kind == "ht_offset":
                    ht = ht + xpar[k]
            pi_ = xyz[i] + s.hi * up[i]
            pj = xyz[j] + ht * up[j]
            dx = pj - pi_
            e = R[i] @ dx
            rho = np.hypot(e[0], e[1])
            S = np.linalg.norm(e)
            kref = net.refraction_k
            for k in sight_extra.get(si, []):
                if extras[k].kind == "refraction":
                    kref = kref + xpar[k]
            if s.hz is not None:
                az = np.arctan2(e[0], e[1])
                model = az - params_o[setups.index(s.setup)]
                for k in sight_extra.get(si, []):
                    if extras[k].kind == "orient_drift":
                        model = model - xpar[k] * extra_val[(k, si)]
                g_enu = np.array([e[1] / rho**2, -e[0] / rho**2, 0.0])
                g = g_enu @ R[i]
                sig = s.sd_dir
                put(j, g, r); put(i, -g, r)
                A[r, ocol[s.setup]] = -1.0
                for k in sight_extra.get(si, []):
                    if extras[k].kind == "ht_offset":
                        A[r, xcol[k]] = g @ up[j]
                    elif extras[k].kind == "orient_drift":
                        A[r, xcol[k]] = -extra_val[(k, si)]
                misc = _wrap(s.hz - model)
                A[r] /= sig
                w[r] = misc / sig
                meta.append(dict(kind="DIR", si=si, frm=s.frm, to=s.to, obs=s.hz, model=model, sigma=sig, ref=s.ref, setup=s.setup, time=s.time))
                r += 1
            if s.zen is not None:
                zg = np.arctan2(rho, e[2])
                model = zg - kref * rho / (2 * G.R_MEAN)
                g_enu = np.array([e[2] * e[0] / rho, e[2] * e[1] / rho, -rho]) / S**2
                g = g_enu @ R[i]
                sig = s.sd_zen
                put(j, g, r); put(i, -g, r)
                for k in sight_extra.get(si, []):
                    if extras[k].kind == "refraction":
                        A[r, xcol[k]] = -rho / (2 * G.R_MEAN)
                    elif extras[k].kind == "ht_offset":
                        A[r, xcol[k]] = g @ up[j]
                A[r] /= sig
                w[r] = (s.zen - model) / sig
                meta.append(dict(kind="ZEN", si=si, frm=s.frm, to=s.to, obs=s.zen, model=model, sigma=sig, ref=s.ref, setup=s.setup, time=s.time))
                r += 1
            if s.slope is not None:
                model = S
                g = dx / S
                for k in sight_extra.get(si, []):
                    if extras[k].kind == "dist_const":
                        model = model + xpar[k]
                    elif extras[k].kind == "dist_scale":
                        model = model + xpar[k] * S
                put(j, g, r); put(i, -g, r)
                for k in sight_extra.get(si, []):
                    if extras[k].kind == "dist_const":
                        A[r, xcol[k]] = 1.0
                    elif extras[k].kind == "dist_scale":
                        A[r, xcol[k]] = S
                    elif extras[k].kind == "ht_offset":
                        A[r, xcol[k]] = g @ up[j]
                sig = s.sd_dist
                A[r] /= sig
                w[r] = (s.slope - model) / sig
                meta.append(dict(kind="DIST", si=si, frm=s.frm, to=s.to, obs=s.slope, model=model, sigma=sig, ref=s.ref, setup=s.setup, time=s.time))
                r += 1

        th, sc = gpar["rotation"], gpar["scale"]
        K = np.array([[0, -u0[2], u0[1]], [u0[2], 0, -u0[0]], [-u0[1], u0[0], 0]])
        Rot = np.eye(3) + np.sin(th) * K + (1 - np.cos(th)) * K @ K
        for vi, v in enumerate(net.vectors):
            i, j = idx[v.frm], idx[v.to]
            dX = xyz_g[j] - xyz_g[i]
            model = (1 + sc) * Rot @ dX
            Ri = R[i]
            cov = _vector_cov(v, Ri)
            Linv = np.linalg.inv(np.linalg.cholesky(cov))
            rows = slice(r, r + 3)
            M = (1 + sc) * Rot
            for pt, sgn in ((j, 1.0), (i, -1.0)):
                n = names[pt]
                if n in col:
                    A[rows, col[n]:col[n] + 3] += sgn * M @ jac_g[pt]
            if "scale" in gcol:
                A[rows, gcol["scale"]] = Rot @ dX
            if "rotation" in gcol:
                A[rows, gcol["rotation"]] = (1 + sc) * (K @ (Rot @ dX))
            A[rows] = Linv @ A[rows]
            w[rows] = Linv @ (np.array(v.dxyz) - model)
            for comp in range(3):
                meta.append(dict(kind="GNSS", vi=vi, comp="ENH"[comp], frm=v.frm, to=v.to, ref=v.ref,
                                 enu_R=Ri, model=model, obs=np.array(v.dxyz), sd=(v.sd_h, v.sd_h, v.sd_v)[comp]))
            r += 3

        for n, (val, sd) in net.weighted.items():
            for comp in range(3):
                A[r, col[n] + comp] = 1.0 / sd[comp]
                w[r] = (val[comp] - enh[idx[n], comp]) / sd[comp]
                meta.append(dict(kind="CTRL", frm=n, to=n, comp="ENH"[comp], obs=val[comp], model=enh[idx[n], comp], sigma=sd[comp], ref="weighted control"))
                r += 1

        N = A.T @ A
        dxp = np.linalg.solve(N, A.T @ w)
        for n in free:
            enh[idx[n]] += dxp[col[n]:col[n] + 3]
        for k, sname in enumerate(setups):
            params_o[k] += dxp[ocol[sname]]
        for key, cc in gcol.items():
            gpar[key] += dxp[cc]
        for k in range(len(extras)):
            xpar[k] += dxp[xcol[k]]
        lin = np.abs(dxp[: 3 * len(free)]).max() if free else 0.0
        if lin < tol and it > 1:
            converged = True
            break

    # Final pass at the solution: residuals, redundancy, precision.
    # vw is the whitened residual (computed minus observed) after the last step.
    Q = np.linalg.inv(N)
    vw = -w + A @ dxp
    hdiag = np.einsum("ij,jk,ik->i", A, Q, A)
    redund = np.clip(1.0 - hdiag, 0.0, 1.0)
    dof = n_obs - nunk
    vf = float(vw @ vw / dof)

    rows_out: list[dict] = []
    for k, m in enumerate(meta):
        rec = dict(m)
        if m["kind"] in ("DIR", "ZEN"):
            rec.update(res_sec=vw[k] * m["sigma"] / SEC, sigma_sec=m["sigma"] / SEC)
        elif m["kind"] in ("DIST", "CTRL"):
            rec.update(res=vw[k] * m["sigma"])
        if m["kind"] != "GNSS":
            rec["v_over_sigma"] = abs(vw[k])
        rec["r"] = float(redund[k])
        rec["w"] = float(abs(vw[k]) / np.sqrt(redund[k])) if redund[k] > 1e-9 else float("nan")
        rec["vw"] = float(vw[k])
        rows_out.append(rec)
    # GNSS residuals in local E, N, H
    for k, m in enumerate(meta):
        if m["kind"] == "GNSS" and m["comp"] == "E":
            v_enu = m["enu_R"] @ (m["model"] - m["obs"])
            for c3 in range(3):
                rows_out[k + c3]["res"] = float(v_enu[c3])
                rows_out[k + c3]["v_over_sigma"] = abs(v_enu[c3]) / m["sd"]
    for rec in rows_out:
        if rec["kind"] == "GNSS":
            for key in ("enu_R", "model", "obs"):
                rec.pop(key, None)

    groups = {"DIR": "Direction", "ZEN": "Vertical angle", "DIST": "Slope distance", "GNSS": "GNSS", "CTRL": "Control"}
    group_vf, group_vf_n = {}, {}
    for key, label in groups.items():
        sel = [k for k, m in enumerate(meta) if m["kind"] == key]
        if sel:
            s2 = float(np.sum(vw[sel] ** 2))
            group_vf[label] = s2 / float(np.sum(redund[sel]))
            group_vf_n[label] = s2 / len(sel)

    coords = {n: tuple(enh[idx[n]]) for n in names}
    sd = {}
    for n in names:
        if n in col:
            d = np.sqrt(np.diag(Q)[col[n]:col[n] + 3])
            sd[n] = tuple(d)
        else:
            sd[n] = (0.0, 0.0, 0.0)
    ex_out = []
    for k, ex in enumerate(extras):
        ex_out.append(dict(kind=ex.kind, label=ex.label, value=float(xpar[k]), sd=float(np.sqrt(Q[xcol[k], xcol[k]]))))
    gnss = {}
    for key, cc in gcol.items():
        gnss[key] = float(gpar[key])
        gnss[key + "_sd"] = float(np.sqrt(Q[cc, cc]))
    return Result(coords=coords, sd=sd, rows=rows_out, dof=dof, vf=vf, group_vf=group_vf,
                  group_vf_by_count=group_vf_n, orientations={s: float(params_o[k]) for k, s in enumerate(setups)},
                  gnss=gnss, extras=ex_out, iterations=it, converged=converged)
