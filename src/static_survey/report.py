"""Plain-English Markdown report for one adjusted job."""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from .adjust import Network, Result
from .diagnose import Finding, worst_residuals


def compare_residuals(res: Result, out) -> dict:
    """Our residuals against the ones CompNet prints, line by line, per observation type.

    Directions and distances share CompNet's sign (computed minus observed). CompNet prints vertical
    angles, so its vertical residuals are negated to compare with our zenith residuals. A type is left out
    when the two lists do not pair up sighting for sighting."""
    lists = {"DIR": out.directions, "ZEN": out.verticals, "DIST": [x for x in out.distances if x["obs"] > 0]}
    found = {}
    for kind, ref in lists.items():
        ours = [r for r in res.rows if r["kind"] == kind]
        if not ours or len(ours) != len(ref) or any((a["frm"], a["to"]) != (b["frm"], b["to"]) for a, b in zip(ours, ref)):
            continue
        if kind == "DIST":
            d = np.array([(a["res"] - b["res"]) * 1000 for a, b in zip(ours, ref)])
        else:
            sign = -1.0 if kind == "ZEN" else 1.0
            d = np.array([a["res_sec"] - sign * b["res"] for a, b in zip(ours, ref)])
        found[kind] = dict(n=len(d), rms=float(np.sqrt((d ** 2).mean())), max=float(np.abs(d).max()),
                           unit="mm" if kind == "DIST" else '"')
    return found


def compare_weighted(net: Network, res: Result, out) -> dict | None:
    """Corrections to the weighted control marks, ours and CompNet's.

    With nothing held fixed, the observations cannot sense a shift of the whole network, so in least squares
    the weighted mean (weights 1/sd^2) of the corrections to the weighted marks is zero in E, N and H."""
    names = [n for n in net.weighted if n in out.weighted and n in res.coords]
    if not names or net.fixed:
        return None
    mean_ours, mean_cn = [], []
    for c, key in enumerate(("ve", "vn", "vh")):
        use = [n for n in names if net.weighted[n][1][c]]  # components held (0) or free (None) carry no weight
        if not use:
            mean_ours.append(0.0)
            mean_cn.append(0.0)
            continue
        w = np.array([1 / net.weighted[n][1][c] ** 2 for n in use])
        v_ours = np.array([res.coords[n][c] - net.weighted[n][0][c] for n in use])
        v_cn = np.array([out.weighted[n][key] for n in use])
        mean_ours.append(float((w * v_ours).sum() / w.sum() * 1000))
        mean_cn.append(float((w * v_cn).sum() / w.sum() * 1000))
    # coordinate difference against CompNet's own printed standard deviation, worst component
    worst = None
    for n, a in out.adjusted.items():
        for c, (k, s) in enumerate((("e", "sd_e"), ("n", "sd_n"), ("h", "sd_h"))):
            if a[s] > 0:
                ratio = abs(res.coords[n][c] - a[k]) / a[s]
                if worst is None or ratio > worst["ratio"]:
                    worst = dict(mark=n, comp="ENH"[c], diff=(res.coords[n][c] - a[k]) * 1000, sd=a[s] * 1000, ratio=ratio)
    return dict(n=len(names), mean_ours=mean_ours, mean_compnet=mean_cn, worst=worst)


def compare_geoid(net: Network, out, tol: float = 0.001) -> list[dict]:
    """Marks where CompNet's own observed GNSS height differences imply a geoid separation N more than ``tol``
    (m) from ours.

    Each vector, taken from CompNet's final position of its first mark, gives the ellipsoidal height difference;
    our N turns it into the observed AHD difference, which CompNet prints as OBS dH. Any difference between the two
    is CompNet's N difference along the vector minus ours. One N offset per mark is fitted to all vectors by least
    squares and taken against the median offset, so a single odd mark stands out. Returns one dict per mark beyond
    ``tol`` once rounded to 0.1 mm as the report prints it: mark, N ours and implied (m), vectors."""
    from . import geodesy as G

    if not net.vectors or len(out.vectors) != len(net.vectors):
        return []
    pos = {n: (a["e"], a["n"], a["h"]) for n, a in out.adjusted.items()}
    pos.update({n: tuple(v) for n, v in out.fixed.items()})
    n_mean = float(np.mean(list(net.geoid.values()))) if net.geoid else 0.0
    N = lambda m: net.geoid.get(m, n_mean)  # noqa: E731
    marks = sorted({m for v in net.vectors for m in (v.frm, v.to)})
    col = {m: i for i, m in enumerate(marks)}
    rows, rhs, count = [], [], {m: 0 for m in marks}
    for v, o in zip(net.vectors, out.vectors):
        if (v.frm, v.to) != (o["frm"], o["to"]) or v.frm not in pos:
            return []  # the listing does not pair up with the project's vectors
        e, n, h = pos[v.frm]
        lat, lon = G.grid_to_latlon(e, n, net.epsg)
        xa = np.array(G.geodetic_to_ecef(lat, lon, h + N(v.frm)), float).ravel()
        _, _, hb = G.ecef_to_geodetic(xa + np.array(v.dxyz))
        ours = float(hb[0]) - (h + N(v.frm)) - (N(v.to) - N(v.frm))
        a = np.zeros(len(marks))
        a[col[v.to]], a[col[v.frm]] = 1.0, -1.0
        rows.append(a)
        rhs.append(ours - o["obs_enh"][2])
        count[v.frm] += 1
        count[v.to] += 1
    x = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)[0]
    x -= np.median(x)
    return [dict(mark=m, n_ours=N(m), n_compnet=N(m) + float(x[col[m]]), vectors=count[m])
            for m in marks if round(abs(x[col[m]]) * 1000, 1) > tol * 1000]  # as printed, to 0.1 mm


def compare_with_compnet(res: Result, out, net: Network | None = None) -> dict:
    """Coordinate and statistic differences against a CompNet .OUT listing."""
    d = np.array([[(res.coords[n][c] - (a["e"], a["n"], a["h"])[c]) * 1000 for c in range(3)]
                  for n, a in out.adjusted.items()])
    worst = max(out.adjusted, key=lambda n: max(abs((res.coords[n][c] - (out.adjusted[n]["e"], out.adjusted[n]["n"],
                                                                         out.adjusted[n]["h"])[c])) for c in range(3)))
    scale = math.sqrt(res.vf)
    sd_ratio = [out.adjusted[n]["sd_n"] / (res.sd[n][1] * scale) for n in out.adjusted if res.sd[n][1] > 0]
    return dict(n=len(d), max_e=float(np.abs(d[:, 0]).max()), max_n=float(np.abs(d[:, 1]).max()),
                max_h=float(np.abs(d[:, 2]).max()), rms=float(np.sqrt((d ** 2).mean())), worst=worst,
                vf=res.vf, vf_compnet=out.variance_factor, dof=res.dof, dof_compnet=out.dof,
                groups=res.group_vf, groups_compnet=out.group_variance, gnss=res.gnss,
                gnss_compnet=out.gnss_params, sd_ratio_median=float(np.median(sd_ratio)) if sd_ratio else None,
                residuals=compare_residuals(res, out),
                weighted=compare_weighted(net, res, out) if net is not None else None,
                n_weighted=sum(bool(s) for _, sd in net.weighted.values() for s in sd) if net is not None else 0,
                geoid=compare_geoid(net, out) if net is not None and getattr(out, "vectors", None) else [])


def _explained_by(row: dict, findings: list[Finding]) -> str:
    for f in findings:
        e = f.evidence
        if f.kind.startswith("orientation") and row["kind"] == "DIR" and row["setup"] == e.get("setup"):
            return f.title.split(":")[0]
        if f.kind == "swap" and row.get("si") in e.get("si", []):
            return f.title
        if f.kind == "pointing" and row.get("si") in e.get("si", []) and row["kind"] in ("DIR", "ZEN"):
            return f.title
        if f.kind == "prism" and row["kind"] == "DIST" and (
                (e["place"] == "setup" and row["setup"] == e["setup"]) or (e["place"] == "target" and row["to"] == e["target"])):
            return f.title
        if f.kind == "refraction" and row["kind"] == "ZEN":
            on_leg = e.get("b") is not None and {row["frm"], row["to"]} == {e.get("a"), e.get("b")}
            on_setup = row["setup"] == e.get("setup") or (
                e.get("preferred") == "instrument_height" and row["setup"] in e["winner"].get("setups", []))
            if on_leg or on_setup:
                return f.title
    return ""


def render(job: str, net: Network, res: Result, findings: list[Finding], sources: list[str],
           compare: dict | None = None, notes: list[str] | None = None, coords: list[dict] | None = None,
           files: list[str] | None = None) -> str:
    L: list[str] = []
    L.append(f"# {job}: adjustment check and cause-finding")
    L.append("")
    L.append(f"Generated {date.today().isoformat()} by static-survey. Every finding is a proposal for the "
             "surveyor to accept or reject; no observation has been changed or removed.")
    L.append("")
    n_dir = sum(1 for s in net.sightings if s.hz is not None)
    n_zen = sum(1 for s in net.sightings if s.zen is not None)
    n_dist = sum(1 for s in net.sightings if s.slope is not None)
    L.append("## What was adjusted")
    L.append("")
    L.append(f"- {len(net.points)} marks adjusted, {len(net.fixed)} held fixed"
             + (f", {len(net.weighted)} weighted" if net.weighted else "") + ".")
    L.append(f"- {n_dir} directions, {n_zen} vertical angles, {n_dist} slope distances in "
             f"{len({s.setup for s in net.sightings})} setups; {len(net.vectors)} GNSS vectors.")
    L.append(f"- Rigorous 3D model on GRS80 ({net.grid}, projection EPSG:{net.epsg})"
             + (f"; refraction k = {net.refraction_k:.3f}" if net.sightings else "")
             + ("; GNSS scale and rotation estimated" if net.vectors else "") + ".")
    L.append(f"- Result: variance factor {res.vf:.3f}, {res.dof} degrees of freedom, "
             f"{'converged' if res.converged else 'NOT converged'} in {res.iterations} iterations.")
    L.append("- Files read: " + ", ".join(f"`{s}`" for s in sources) + ".")
    L.append("")

    if compare:
        c = compare
        wc = c.get("weighted")
        L.append("## Check against CompNet")
        L.append("")
        detail = (f"(largest differences: E {c['max_e']:.2f}, N {c['max_n']:.2f}, H {c['max_h']:.2f} mm, at {c['worst']}; "
                  f"rms {c['rms']:.2f} mm)")
        if wc:
            L.append(f"Our coordinates differ from CompNet's by up to **{max(c['max_e'], c['max_n'], c['max_h']):.2f} mm** "
                     f"over all {c['n']} marks {detail}. See *Weighted control* below for why.")
        else:
            L.append(f"Independent recomputation of the same job reproduces CompNet's adjusted coordinates for all "
                     f"{c['n']} marks to **{max(c['max_e'], c['max_n'], c['max_h']):.2f} mm** or better {detail}.")
        L.append("")
        names = {"DIR": "directions", "ZEN": "vertical angles", "DIST": "slope distances"}
        rs = c.get("residuals") or {}
        if rs:
            def q(x, u):
                return f"{x:.3f}{u}" if u == '"' else f"{x:.3f} {u}"
            parts = [f"{names[k]} rms {q(v['rms'], v['unit'])} (largest {q(v['max'], v['unit'])}, {v['n']} lines)"
                     for k, v in rs.items()]
            text = "Residuals compared line by line with CompNet's listing: " + "; ".join(parts) + "."
            missing = [names[k] for k in names if k not in rs]
            if missing:
                text += " Not compared, because the lists do not pair up sighting for sighting: " + ", ".join(missing) + "."
            L += [text, ""]
        L.append("| | static-survey | CompNet |")
        L.append("|---|---|---|")
        L.append(f"| Variance factor | {c['vf']:.3f} | {c['vf_compnet']} |")
        L.append(f"| Degrees of freedom | {c['dof']} | {c['dof_compnet']} |")
        dof_note = (c.get("n_weighted") and c["dof_compnet"] is not None
                    and c["dof"] == c["dof_compnet"] + c["n_weighted"])
        for k in ("Direction", "Vertical angle", "Slope distance", "GNSS"):
            if k in c["groups"]:
                L.append(f"| {k} group factor | {c['groups'][k]:.3f} | {c['groups_compnet'].get(k, '')} |")
        if c["gnss"]:
            L.append(f"| GNSS scale | {c['gnss'].get('scale', 0) * 1e6:.2f} ppm | {c['gnss_compnet'].get('Scale ppm', '')} ppm |")
            L.append(f"| GNSS rotation (size) | {abs(c['gnss'].get('rotation', 0)) * 206264.806:.2f}\" | "
                     f"{abs(c['gnss_compnet'].get('Horizontal rotation', 0))}\" |")
        L.append("")
        if dof_note:
            L.append(f"CompNet does not count the {c['n_weighted']} weighted "
                     f"{'coordinate' if c['n_weighted'] == 1 else 'coordinates'} as observations: "
                     f"{c['dof']} = {c['dof_compnet']} + {c['n_weighted']}.")
            L.append("")
        if wc:
            L += _weighted_paragraph(wc)
        if c.get("geoid"):
            items = "; ".join(f"{g['mark']} {(g['n_compnet'] - g['n_ours']) * 1000:+.1f} mm ({g['n_ours']:.3f} m in the "
                              f".WGN, {g['n_compnet']:.3f} m implied; {g['vectors']} vector{'s' if g['vectors'] != 1 else ''})"
                              for g in c["geoid"])
            L.append("CompNet's own observed height differences (OBS dH) imply a geoid separation N more than 1 mm "
                     f"from the project's .WGN at: {items}. The heights here use the .WGN values.")
            L.append("")
        if c["sd_ratio_median"]:
            r = c["sd_ratio_median"]
            if f"{r:.2f}" == "1.00":
                first = ("CompNet's printed standard deviations equal the a-priori values × √(variance factor) "
                         f"(ratio {r:.2f} after that scaling).")
            else:
                first = (f"CompNet's printed standard deviations are {'smaller' if r < 1 else 'larger'} than the "
                         f"a-priori values × √(variance factor): the median ratio is {r:.2f}.")
            L.append(f"{first} With a variance factor of {c['vf']:.2f} "
                     f"that shrinks every printed standard deviation by {100 * (1 - math.sqrt(c['vf'])):.0f}%. "
                     "Scaling down when the factor is below 1 is a choice; the a-priori figures are the cautious ones.")
            L.append("")

    L.append("## Findings")
    L.append("")
    order = {"likely-cause": 0, "check": 1, "info": 2}
    for f in sorted(findings, key=lambda f: order.get(f.severity, 3)):
        tag = {"likely-cause": "Likely cause", "check": "Check", "info": "Info"}[f.severity]
        L.append(f"### {f.title}")
        L.append("")
        L.append(f"*{tag}.* {f.summary}")
        L.append("")
        if f.kind.startswith("orientation") and f.evidence.get("readings"):
            rows = f.evidence["readings"]
            show = rows[: min(len(rows), 8)]
            L.append("| # | Target | Time | Residual |")
            L.append("|---|---|---|---|")
            for i, r in enumerate(show, 1):
                L.append(f"| {i} | {r['to']} | {r['time']} | {r['res_sec']:+.1f}\" |")
            if len(rows) > len(show):
                L.append(f"| … | {len(rows) - len(show)} more | | |")
            L.append("")

    L.append("## Largest residuals (w-test) and what explains them")
    L.append("")
    L.append("| Obs | From → To | w | Residual | Source | Explained by |")
    L.append("|---|---|---|---|---|---|")
    for r in worst_residuals(res, 12, kinds=("DIR", "ZEN", "DIST", "GNSS")):
        val = f"{r['res_sec']:+.1f}\"" if "res_sec" in r else f"{r['res'] * 1000:+.1f} mm"
        kind = f"GNSS {'dE dN dH'.split()['ENH'.index(r['comp'])]}" if r["kind"] == "GNSS" else r["kind"]
        L.append(f"| {kind} | {r['frm']} → {r['to']} | {r['w']:.2f} | {val} | {r['ref']}"
                 f"{' ' + r['time'] if r.get('time') else ''} | {_explained_by(r, findings) or '—'} |")
    L.append("")
    L.append("w is the residual divided by its own standard deviation (Baarda's w-test). CompNet prints the "
             "residual divided by the observation's standard deviation; the two are close when redundancy is high.")
    L.append("")
    if coords:
        L += _coordinates(res, coords, files, net.grid)
    if notes:
        L.append("## Notes and limits")
        L.append("")
        for n in notes:
            L.append(f"- {n}")
        L.append("")
    return "\n".join(L)


def _weighted_paragraph(wc: dict) -> list[str]:
    def mm(x):
        return f"{x:+.2f}" if abs(x) >= 0.005 else "0.00"
    mo = ", ".join(mm(x) for x in wc["mean_ours"])
    mc = ", ".join(mm(x) for x in wc["mean_compnet"])
    L = ["### Weighted control", ""]
    text = (f"Position comes from {wc['n']} weighted marks; none is held fixed. The observations cannot sense a shift "
            "of the whole network, so in least squares the corrections to the weighted marks, weighted by 1/sd², "
            f"average zero in E, N and H. Ours average {mo} mm (E, N, H). CompNet's printed corrections average {mc} mm. ")
    # CompNet prints each correction to 0.1 mm, so its average is only known to +/-0.05 mm.
    if max(abs(x) for x in wc["mean_compnet"]) > 0.05:
        text += ("That is more than its 0.1 mm printing explains, so CompNet's coordinates are not the least squares "
                 "solution for the weights it prints. Where the residuals above match, both programs fit the "
                 "observations alike, and the difference lies mostly in how the whole network is shifted and turned "
                 "onto the weighted marks. ")
    else:
        text += "Within its 0.1 mm printing that is zero too. "
    w = wc["worst"]
    if w:
        text += (f"Measured against CompNet's own standard deviations, the largest difference is {w['diff']:+.1f} mm in "
                 f"{w['comp']} at {w['mark']}: {w['ratio']:.2f} times CompNet's {w['sd']:.1f} mm there.")
    L += [text, ""]
    return L


def _coordinates(res: Result, rows: list[dict], files: list[str] | None, grid: str = "MGA2020 zone 56") -> list[str]:
    from .cad import sd_scale

    k = sd_scale(res)
    L = ["## Adjusted coordinates", ""]
    if files:
        L.append("Also written: " + ", ".join(f"`{f}`" for f in files) + ".")
        L.append("")
    L.append(f"{grid}, AHD heights. Standard deviations are the a-priori ones from the instrument settings"
             + (f", scaled up by √(variance factor) = {k:.2f} because the variance factor is above 1." if k > 1 else
                "; the variance factor is not above 1, so they are not scaled.")
             + " The 95% error ellipse is the horizontal area the mark lies in with 95% confidence: half-length of "
             "the long axis × half-length of the short axis, and the bearing of the long axis.")
    L.append("")
    L.append("| Mark | Status | E (m) | N (m) | H (m) | sd E (mm) | sd N (mm) | sd H (mm) | 95% ellipse (mm) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["status"] == "fixed":
            sds, ell = "| — | — | —", "held fixed"
        else:
            held = r.get("held", "")
            sds = "".join(" | —" if c in held else f" | {r[k] * 1000:.1f}" for c, k in zip("ENH", ("sd_e", "sd_n", "sd_h")))
            sds = "|" + sds[2:]
            ell = "held" if r["a"] is None else f"{r['a'] * 1000:.1f} × {r['b'] * 1000:.1f} at {r['bearing']:.0f}°"
        L.append(f"| {r['name']} | {r['status']} | {r['e']:.4f} | {r['n']:.4f} | {r['h']:.4f} {sds} | {ell} |")
    L.append("")
    return L
