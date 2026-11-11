"""Static Survey for Windows: a small window around the ``static-survey`` command.

The window is a theodolite drawn in pencil (``static_survey.ui``). Click the sighting cross to pick a job folder.
The window finds a CompNet project (``.TRV`` and friends) or the raw files (Leica GSI, Leica Infinity ``.asc`` and
a ``.FIX``/``.STD`` control file). Click the vertical circle to run the adjustment and the cause-finding. It writes
four files next to the job files, named after the job: ``JOB-report.html`` and ``.md``, ``JOB-coordinates.csv`` and
``JOB-network.dxf`` for AutoCAD, and shows the report in the window. The DXF is not ``JOB.dxf`` because CompNet
writes its own drawing under that name. Extra command options (``--rename``, ``--exclude``,
``--file-settings`` ...) go in the options box exactly as on the command line.

Started with arguments (``StaticSurvey.exe raw --gsi ...``) it runs the command line instead, without a
window. Nothing in the job folder is changed except the report files it writes.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from pathlib import Path

GEOID_GRID = "au_ga_AUSGeoid2020_20180201.tif"
CSS = """
body { font-family: Segoe UI, Arial, sans-serif; max-width: 980px; margin: 2em auto; padding: 0 1em;
       color: #1d2733; line-height: 1.45; }
h1 { font-size: 1.6em; border-bottom: 2px solid #2b6cb0; padding-bottom: .3em; }
h2 { font-size: 1.25em; margin-top: 1.6em; color: #2b6cb0; }
h3 { font-size: 1.05em; margin-top: 1.2em; }
table { border-collapse: collapse; margin: .8em 0; font-size: .92em; }
th, td { border: 1px solid #c8d1dc; padding: .3em .6em; text-align: left; vertical-align: top; }
th { background: #eef3f8; }
code { background: #f2f4f7; padding: 0 .25em; border-radius: 3px; }
em { color: #555; }
"""


def bundled_geoid() -> Path | None:
    """The AUSGeoid2020 grid shipped inside the program (``static_survey/data``), if this build has one."""
    import static_survey

    p = Path(static_survey.__file__).resolve().parent / "data" / GEOID_GRID
    return p if p.exists() else None


def find_job(folder: Path) -> dict:
    """What the folder holds: CompNet project stems, GSI and .asc files, control files. A project is a stem with a
    .TRV (total station) or a .GPS (GNSS vectors) file; a GNSS-only project has no .TRV."""
    files = sorted(p for p in folder.iterdir() if p.is_file())
    by_ext: dict[str, list[Path]] = {}
    for p in files:
        by_ext.setdefault(p.suffix.lower(), []).append(p)
    return dict(
        projects=sorted({p.with_suffix("") for p in by_ext.get(".trv", []) + by_ext.get(".gps", [])}),
        gsi=by_ext.get(".gsi", []),
        asc=by_ext.get(".asc", []),
        control=by_ext.get(".fix", []) + by_ext.get(".std", []),
    )


def outputs_for(folder: Path, name: str) -> dict[str, Path]:
    """Where a run writes: the report (Markdown, then HTML), the coordinates and the drawing."""
    return dict(md=folder / f"{name}-report.md", csv=folder / f"{name}-coordinates.csv",
                dxf=folder / f"{name}-network.dxf")


def job_name(folder: Path, job: dict, project: Path | None = None) -> str:
    return (project or job["projects"][0]).name if job["projects"] else folder.name


def command_for(folder: Path, job: dict, options: str, project: Path | None = None) -> list[str]:
    """The ``static-survey`` arguments for this folder: CompNet mode when there is a project, raw otherwise."""
    extra = shlex.split(options) if options.strip() else []
    name = job_name(folder, job, project)
    o = outputs_for(folder, name)
    out = ["--out", str(o["md"]), "--csv", str(o["csv"]), "--dxf", str(o["dxf"])]
    if job["projects"]:
        stem = project or job["projects"][0]
        args = ["compnet", str(stem)]
        for g in job["gsi"]:
            args += ["--gsi", str(g)]
        return args + ["--job", name] + out + extra
    if not job["gsi"] and not job["asc"]:
        raise ValueError("No CompNet project (.TRV or .GPS) and no raw files (.GSI or .asc) in this folder.")
    args = ["raw"]
    for g in job["gsi"]:
        args += ["--gsi", str(g)]
    for a in job["asc"]:
        args += ["--asc", str(a)]
    if "--control" not in extra:
        if not job["control"]:
            raise ValueError("Raw files need control marks: put a .FIX or .STD file in the folder, or give "
                             "--control FILE in the options (CSV of name,E,N,H[,sdE,sdN,sdH]).")
        for c in job["control"]:
            args += ["--control", str(c)]
    if "--geoid-grid" not in extra and "--geoid-compnet" not in extra:
        grid = bundled_geoid()
        if grid is None:
            raise ValueError("This build has no geoid grid: give --geoid-grid FILE in the options.")
        args += ["--geoid-grid", str(grid)]
    return args + ["--job", name] + out + extra


def to_html(md_path: Path) -> Path:
    import markdown

    body = markdown.markdown(md_path.read_text(encoding="utf-8"), extensions=["tables"])
    html = md_path.with_suffix(".html")
    html.write_text(f"<!doctype html><html><head><meta charset='utf-8'><title>{md_path.parent.name}: "
                    f"static-survey report</title><style>{CSS}</style></head><body>{body}</body></html>",
                    encoding="utf-8")
    return html


FROM_WINDOW = "STATIC_SURVEY_FROM_WINDOW"     # set for the command line the window starts
_running: subprocess.Popen | None = None


def self_command() -> list[str]:
    """How to start this program again: the exe itself when built, ``python -m static_survey.app`` otherwise."""
    return [sys.executable] if getattr(sys, "frozen", False) else [sys.executable, "-m", "static_survey.app"]


def run_command(argv: list[str]) -> str:
    """Run the command line in a process of its own and return what it printed.

    In the window's own process the adjustment held up the drawing: the moves stalled for up to half a second.
    """
    global _running
    quiet = getattr(subprocess, "CREATE_NO_WINDOW", 0)       # Windows: no console flashing up
    child = subprocess.Popen(self_command() + argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                             stderr=subprocess.STDOUT, env={**os.environ, FROM_WINDOW: "1"}, creationflags=quiet)
    _running = child
    try:
        printed = child.communicate()[0].decode("utf-8", errors="replace")
    finally:
        _running = None
    if child.returncode:
        raise RuntimeError(printed or f"static-survey exited with {child.returncode}")
    return printed


def stop_running() -> bool:
    """Stop the run in hand, if there is one. For when the window is closed part-way."""
    child = _running
    if child is None:
        return False
    child.terminate()
    return True


def gui() -> None:
    """Open the window (``static_survey.ui.window``). A folder given as the only argument is chosen at once."""
    from static_survey.ui.window import run

    run(sys.argv[1] if len(sys.argv) == 2 and Path(sys.argv[1]).is_dir() else None)


def main() -> int:
    # A windowed build has no console: give print() somewhere to go.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    if len(sys.argv) > 1 and sys.argv[1] in ("compnet", "raw"):
        from static_survey.cli import main as cli_main

        if os.environ.get(FROM_WINDOW):         # the window reads what is printed as UTF-8
            sys.stdout.reconfigure(encoding="utf-8")
            sys.stderr.reconfigure(encoding="utf-8")

        return cli_main(sys.argv[1:])
    gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
