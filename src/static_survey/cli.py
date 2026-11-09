"""Command line: ``static-survey compnet STEM [--gsi FILE ...] [--out REPORT.md]``."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _cad_files(args, job: str, net, res, findings) -> tuple[list[dict], list[str]]:
    """Coordinate rows for the report, and the CSV / DXF when asked for. Returns (rows, names written)."""
    from .cad import coordinate_rows, write_csv, write_dxf

    rows, files = coordinate_rows(net, res), []
    if args.csv:
        write_csv(args.csv, rows)
        files.append(Path(args.csv).name)
        print(f"coordinates written to {args.csv}")
    if args.dxf:
        drawn = write_dxf(args.dxf, job, net, res, rows, findings)
        files.append(Path(args.dxf).name)
        print(f"drawing written to {args.dxf}: {drawn['marks']} marks, {drawn['ts_lines']} total station lines, "
              f"{drawn['gnss_lines']} GNSS lines, {drawn['flagged']} flagged, ellipses x{drawn['ellipse_scale']:,.0f}")
    return rows, files


def _compnet(args) -> int:
    from .diagnose import diagnose
    from .io.compnet import read_out, read_project, to_network
    from .io.gsi import attach_times, read_gsi
    from .report import compare_with_compnet, render

    stem = Path(args.stem)
    proj = read_project(stem)
    net = to_network(proj, sec_per_km=args.refraction)
    sources = [p.name for p in sorted(stem.parent.glob(stem.name + ".*"))
               if p.suffix.upper() in (".TRV", ".DIR", ".VRT", ".SLP", ".GPS", ".VCV", ".FIX", ".STD", ".WGN", ".N",
                                       ".VAR", ".OUT")]
    notes = [f"Weights taken from CompNet's {', '.join(sorted(set(proj.weight_source.values())))} files, "
             "as CompNet uses them (rounded copies of the .TRV values)."]
    if proj.disabled:
        notes.append(f"{len(proj.disabled)} sighting(s) switched off in the .TRV with '*' were left out: "
                     + "; ".join(f"`{d}`" for d in proj.disabled))
    timed = 0
    for g in args.gsi or []:
        timed += attach_times(net.sightings, read_gsi(g))
        sources.append(Path(g).name)
    if args.gsi:
        notes.append(f"Instrument timestamps matched to {timed} of {len(net.sightings)} sightings.")
    res, findings = diagnose(net)
    compare = None
    out_path = stem.with_suffix(".OUT")
    if out_path.exists():
        compare = compare_with_compnet(res, read_out(out_path))
    notes.append("Deflections of the vertical are not modelled; trig heights are treated as AHD differences, "
                 "as CompNet does.")
    notes.append("Vertical residuals here are on zenith angles (computed minus observed). CompNet prints vertical "
                 "angles, so its signs are reversed: +20\" here is -20\" in CompNet's listing.")
    rows, files = _cad_files(args, args.job or stem.name, net, res, findings)
    md = render(args.job or stem.name, net, res, findings, sources, compare, notes, coords=rows, files=files)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        print(md)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(
            coords=res.coords, sd=res.sd, vf=res.vf, dof=res.dof, group_vf=res.group_vf, gnss=res.gnss,
            compare=compare, findings=[f.__dict__ for f in findings]), indent=1, default=str), encoding="utf-8")
    return 0


_FILE_KEYS = {"pointing-h": ("pointing_h",), "pointing-v": ("pointing_v",),
              "centring": ("centring_inst", "centring_target"), "height": ("height_inst", "height_target"),
              "edm-const": ("edm_const",), "edm-ppm": ("edm_ppm",)}


def _file_settings(spec: str, base):
    """'FILE:pointing-h=4,centring=0.002' -> (file name, Instrument based on the job's settings)."""
    from dataclasses import replace

    name, sep, body = spec.partition(":")
    if not sep or not name.strip() or not body.strip():
        raise SystemExit(f"--file-settings wants FILE:key=value,..., got {spec!r}")
    changes = {}
    for item in body.split(","):
        key, sep, val = item.partition("=")
        key = key.strip()
        if not sep or key not in _FILE_KEYS:
            raise SystemExit(f"--file-settings: unknown setting {item!r}; use {', '.join(_FILE_KEYS)}")
        for attr in _FILE_KEYS[key]:
            changes[attr] = float(val)
    return Path(name.strip()).name, replace(base, **changes)


def _settings_text(inst) -> str:
    return (f"pointing {inst.pointing_h:g}\"/{inst.pointing_v:g}\", centring {inst.centring_inst * 1000:.1f} mm, "
            f"height measure {inst.height_inst * 1000:.1f} mm, EDM {inst.edm_const * 1000:.1f} mm + {inst.edm_ppm:g} ppm")


def _raw(args) -> int:
    from .diagnose import diagnose
    from .io.compnet import read_control, read_out, read_project, to_network
    from .rawjob import Instrument, RawJob, build
    from .report import compare_with_compnet, render

    fixed, weighted = {}, {}
    for c in args.control:
        f, w = read_control(c)
        fixed.update(f)
        weighted.update(w)
    extra, extra_note = [], None
    if args.compnet_extra:
        stem, _, block = args.compnet_extra.partition(":")
        cn = to_network(read_project(stem))
        extra = [s for s in cn.sightings if not block or s.ref.endswith(f"({block})")]
        extra_note = (f"{len(extra)} sightings in {len({s.setup for s in extra})} setups taken from CompNet's "
                      f"`{Path(stem).name}.TRV`" + (f" block {block}" if block else "") + " because their raw file "
                      "was not supplied.")
    inst = Instrument(**{k: v for k, v in (
        ("pointing_h", args.pointing_h), ("pointing_v", args.pointing_v), ("centring_inst", args.centring),
        ("centring_target", args.centring), ("edm_const", args.edm_const), ("edm_ppm", args.edm_ppm)) if v is not None})
    geoid_table = read_project(args.geoid_compnet).geoid if args.geoid_compnet else None
    if not args.geoid_grid and not args.geoid_compnet:
        from .app import bundled_geoid  # the Windows program ships the grid inside it

        grid = bundled_geoid()
        if grid is None:
            raise SystemExit("raw needs a geoid: --geoid-grid FILE (AUSGeoid2020) or --geoid-compnet STEM")
        args.geoid_grid = str(grid)
    rename = {}
    for pair in args.rename or []:
        old, sep, new = pair.partition("=")
        if not sep or not old.strip() or not new.strip():
            raise SystemExit(f"--rename wants OLD=NEW, got {pair!r}")
        rename[old.strip()] = new.strip()
    per_file = dict(_file_settings(s, inst) for s in args.file_settings or [])
    job = RawJob(rename=rename, file_instruments=per_file, gsi=args.gsi or [], asc=args.asc or [], fixed=fixed, weighted=weighted, instrument=inst,
                 geoid_grid=args.geoid_grid, geoid_table=geoid_table, refraction_sec_per_km=args.refraction, extra_sightings=extra,
                 exclude=set(args.exclude or []), round_like_compnet=args.round_like_compnet)
    net, info = build(job)

    def pointing(s):
        i = per_file.get(s.ref.split(" line ")[0], inst)
        return i.pointing_h, i.pointing_v

    res, findings = diagnose(net, pointing=pointing)
    compare = compare_with_compnet(res, read_out(args.compare_out)) if args.compare_out else None
    src = {k: sum(1 for l in info["provisional_log"] if k in l) for k in ("control", "GNSS", "radiated", "resection")}
    notes = [f"Built from raw files: {info['n_gsi_sightings']} GSI readings and {info['n_vectors']} Infinity "
             f"baselines. Starting coordinates computed here, not taken from CompNet "
             f"({src['control']} control, {src['GNSS']} by GNSS vector, {src['radiated']} radiated, "
             f"{src['resection']} resected).",
             f"Standard errors from instrument settings: {_settings_text(inst)}"
             + "".join(f"; `{n}` uses its own: {_settings_text(i)}" for n, i in per_file.items())
             + "; GNSS 5 mm + 2 ppm H, 10 mm + 2 ppm V"
             + (", rounded as CompNet rounds them." if args.round_like_compnet else "."),
             "Geoid: " + (f"N values from CompNet's `{Path(args.geoid_compnet).name}.WGN`" if args.geoid_compnet else "")
             + (" and " if args.geoid_compnet and args.geoid_grid else "")
             + (f"AUSGeoid2020 grid `{Path(args.geoid_grid).name}`" if args.geoid_grid else "") + "."]
    if extra_note:
        notes.append(extra_note)
    if rename:
        notes.append("Marks renamed from the field name to the project name: "
                     + ", ".join(f"`{o}` → `{n}`" for o, n in rename.items()) + ".")
    if args.exclude:
        notes.append(f"{len(args.exclude)} reading(s) left out at the user's request: "
                     + "; ".join(f"`{x}`" for x in args.exclude) + ".")
    notes.append("Deflections of the vertical are not modelled; trig heights are treated as AHD differences.")
    notes.append("Vertical residuals are on zenith angles (computed minus observed); CompNet's vertical-angle "
                 "residuals have the opposite sign.")
    sources = [Path(x).name for x in (args.gsi or []) + (args.asc or []) + args.control]
    rows, files = _cad_files(args, args.job or "Raw job", net, res, findings)
    md = render(args.job or "Raw job", net, res, findings, sources, compare, notes, coords=rows, files=files)
    if args.out:
        Path(args.out).write_text(md, encoding="utf-8")
        print(f"report written to {args.out}")
    else:
        print(md)
    if args.json:
        Path(args.json).write_text(json.dumps(dict(coords=res.coords, sd=res.sd, vf=res.vf, dof=res.dof,
                                                   compare=compare, findings=[f.__dict__ for f in findings]),
                                              indent=1, default=str), encoding="utf-8")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="static-survey", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compnet", help="adjust and diagnose a CompNet project")
    c.add_argument("stem", help="project path without extension, e.g. data/JOB")
    c.add_argument("--gsi", action="append", help="Leica GSI raw file(s) for instrument timestamps")
    c.add_argument("--refraction", type=float, default=13.6, help="CompNet curvature+refraction, sec/km (13.6)")
    c.add_argument("--job", help="job title for the report")
    c.add_argument("--out", help="write the Markdown report here")
    c.add_argument("--json", help="also write machine-readable results here")
    c.add_argument("--csv", help="write adjusted coordinates here (name,E,N,H,sdE,sdN,sdH,status)")
    c.add_argument("--dxf", help="write the network drawing here (AutoCAD 2000 DXF)")
    c.set_defaults(func=_compnet)
    r = sub.add_parser("raw", help="adjust and diagnose straight from GSI and Infinity files")
    r.add_argument("--gsi", action="append", help="Leica GSI raw file (repeatable)")
    r.add_argument("--asc", action="append", help="Leica Infinity baseline export .asc (repeatable)")
    r.add_argument("--control", action="append", required=True,
                   help="control marks: CompNet .FIX/.STD or CSV name,E,N,H[,sdE,sdN,sdH] (repeatable)")
    r.add_argument("--geoid-grid", help="AUSGeoid2020 grid, e.g. au_ga_AUSGeoid2020_20180201.tif")
    r.add_argument("--geoid-compnet", help="STEM: take geoid N per mark from a CompNet project's .WGN")
    r.add_argument("--compnet-extra", help="STEM[:PROJECT] take these CompNet sightings where raw files are missing")
    r.add_argument("--exclude", action="append", help='skip a reading, e.g. "File.GSI line 883"')
    r.add_argument("--rename", action="append", metavar="OLD=NEW",
                   help="use the project name for a mark named differently in the field file (repeatable)")
    r.add_argument("--refraction", type=float, default=13.6, help="curvature+refraction, sec/km (13.6)")
    r.add_argument("--pointing-h", type=float); r.add_argument("--pointing-v", type=float)
    r.add_argument("--centring", type=float); r.add_argument("--edm-const", type=float)
    r.add_argument("--edm-ppm", type=float)
    r.add_argument("--file-settings", action="append", metavar="FILE:KEY=VALUE,...",
                   help="instrument settings for one GSI file only (repeatable); keys: pointing-h, pointing-v "
                        "(arc-seconds), centring, height, edm-const (metres), edm-ppm")
    r.add_argument("--round-like-compnet", action="store_true", help="round standard errors as CompNet does")
    r.add_argument("--compare-out", help="CompNet .OUT listing to compare the result with")
    r.add_argument("--job", help="job title for the report")
    r.add_argument("--out", help="write the Markdown report here")
    r.add_argument("--json", help="also write machine-readable results here")
    r.add_argument("--csv", help="write adjusted coordinates here (name,E,N,H,sdE,sdN,sdH,status)")
    r.add_argument("--dxf", help="write the network drawing here (AutoCAD 2000 DXF)")
    r.set_defaults(func=_raw)
    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
