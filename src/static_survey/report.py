"""Plain-English Markdown report for one adjusted job."""

from __future__ import annotations

import math
from datetime import date

import numpy as np

from .adjust import Network, Result
from .diagnose import Finding, worst_residuals


def compare_with_compnet(res: Result, out) -> dict:
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
                gnss_compnet=out.gnss_params, sd_ratio_median=float(np.median(sd_ratio)) if sd_ratio else None)


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
    L.append(f"- Rigorous 3D model on GRS80 (MGA2020 EPSG:{net.epsg}); refraction k = {net.refraction_k:.3f}"
             + ("; GNSS scale and rotation estimated" if net.vectors else "") + ".")
    L.append(f"- Result: variance factor {res.vf:.3f}, {res.dof} degrees of freedom, "
             f"{'converged' if res.converged else 'NOT converged'} in {res.iterations} iterations.")
    L.append("- Files read: " + ", ".join(f"`{s}`" for s in sources) + ".")
    L.append("")

    if compare:
        c = compare
        L.append("## Check against CompNet")
        L.append("")
        L.append(f"Independent recomputation of the same job reproduces CompNet's adjusted coordinates for all "
                 f"{c['n']} marks to **{max(c['max_e'], c['max_n'], c['max_h']):.2f} mm** or better "
                 f"(largest differences: E {c['max_e']:.2f}, N {c['max_n']:.2f}, H {c['max_h']:.2f} mm, at {c['worst']}; "
                 f"rms {c['rms']:.2f} mm).")
        L.append("")
        L.append("| | static-survey | CompNet |")
        L.append("|---|---|---|")
        L.append(f"| Variance factor | {c['vf']:.3f} | {c['vf_compnet']} |")
        L.append(f"| Degrees of freedom | {c['dof']} | {c['dof_compnet']} |")
        for k in ("Direction", "Vertical angle", "Slope distance", "GNSS"):
            if k in c["groups"]:
                L.append(f"| {k} group factor | {c['groups'][k]:.3f} | {c['groups_compnet'].get(k, '')} |")
        if c["gnss"]:
            L.append(f"| GNSS scale | {c['gnss'].get('scale', 0) * 1e6:.2f} ppm | {c['gnss_compnet'].get('Scale ppm', '')} ppm |")
            L.append(f"| GNSS rotation (size) | {abs(c['gnss'].get('rotation', 0)) * 206264.806:.2f}\" | "
                     f"{abs(c['gnss_compnet'].get('Horizontal rotation', 0))}\" |")
        L.append("")
        if c["sd_ratio_median"]:
            L.append(f"CompNet's printed standard deviations equal the a-priori values × √(variance factor) "
                     f"(ratio {c['sd_ratio_median']:.2f} after that scaling). With a variance factor of {c['vf']:.2f} "
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
    for r in worst_residuals(res, 12):
        val = f"{r['res_sec']:+.1f}\"" if "res_sec" in r else f"{r['res'] * 1000:+.1f} mm"
        L.append(f"| {r['kind']} | {r['frm']} → {r['to']} | {r['w']:.2f} | {val} | {r['ref']}"
                 f"{' ' + r['time'] if r.get('time') else ''} | {_explained_by(r, findings) or '—'} |")
    L.append("")
    L.append("w is the residual divided by its own standard deviation (Baarda's w-test). CompNet prints the "
             "residual divided by the observation's standard deviation; the two are close when redundancy is high.")
    L.append("")
    if coords:
        L += _coordinates(res, coords, files)
    if notes:
        L.append("## Notes and limits")
        L.append("")
        for n in notes:
            L.append(f"- {n}")
        L.append("")
    return "\n".join(L)


def _coordinates(res: Result, rows: list[dict], files: list[str] | None) -> list[str]:
    from .cad import sd_scale

    k = sd_scale(res)
    L = ["## Adjusted coordinates", ""]
    if files:
        L.append("Also written: " + ", ".join(f"`{f}`" for f in files) + ".")
        L.append("")
    L.append("MGA2020 zone 56, AHD heights. Standard deviations are the a-priori ones from the instrument settings"
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
            sds = f"| {r['sd_e'] * 1000:.1f} | {r['sd_n'] * 1000:.1f} | {r['sd_h'] * 1000:.1f}"
            ell = f"{r['a'] * 1000:.1f} × {r['b'] * 1000:.1f} at {r['bearing']:.0f}°"
        L.append(f"| {r['name']} | {r['status']} | {r['e']:.4f} | {r['n']:.4f} | {r['h']:.4f} {sds} | {ell} |")
    L.append("")
    return L
