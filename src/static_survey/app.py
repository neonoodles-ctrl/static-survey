"""Static Survey for Windows: a small window around the ``static-survey`` command.

Pick a job folder. The window finds a CompNet project (``.TRV`` and friends) or the raw files (Leica GSI,
Leica Infinity ``.asc`` and a ``.FIX``/``.STD`` control file), runs the adjustment and the cause-finding,
and writes four files next to the job files, named after the job: ``JOB-report.html`` (opened in the
browser) and ``.md``, ``JOB-coordinates.csv`` and ``JOB-network.dxf`` for AutoCAD. The DXF is not
``JOB.dxf`` because CompNet writes its own drawing under that name. Extra command options (``--rename``, ``--exclude``,
``--file-settings`` ...) go in the options box exactly as on the command line.

Started with arguments (``StaticSurvey.exe raw --gsi ...``) it runs the command line instead, without a
window. Nothing in the job folder is changed except the two report files it writes.
"""

from __future__ import annotations

import io
import os
import shlex
import sys
import threading
import traceback
import webbrowser
from contextlib import redirect_stderr, redirect_stdout
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
    """What the folder holds: CompNet project stems, GSI and .asc files, control files."""
    files = sorted(p for p in folder.iterdir() if p.is_file())
    by_ext: dict[str, list[Path]] = {}
    for p in files:
        by_ext.setdefault(p.suffix.lower(), []).append(p)
    return dict(
        projects=[p.with_suffix("") for p in by_ext.get(".trv", [])],
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
        raise ValueError("No CompNet project (.TRV) and no raw files (.GSI or .asc) in this folder.")
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


def run_command(argv: list[str]) -> str:
    """Run the command line in-process and return what it printed."""
    from static_survey.cli import main as cli_main

    buf = io.StringIO()
    with redirect_stdout(buf), redirect_stderr(buf):
        code = cli_main(argv)
    if code:
        raise RuntimeError(buf.getvalue() or f"static-survey exited with {code}")
    return buf.getvalue()


def gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = tk.Tk()
    root.title("Static Survey")
    root.geometry("760x520")
    frm = ttk.Frame(root, padding=12)
    frm.pack(fill="both", expand=True)
    frm.columnconfigure(1, weight=1)

    folder_var = tk.StringVar(value=sys.argv[1] if len(sys.argv) == 2 and Path(sys.argv[1]).is_dir() else "")
    project_var = tk.StringVar()
    options_var = tk.StringVar()
    state = {"job": None, "report": None}

    ttk.Label(frm, text="Job folder").grid(row=0, column=0, sticky="w")
    ttk.Entry(frm, textvariable=folder_var).grid(row=0, column=1, sticky="ew", padx=6)
    ttk.Button(frm, text="Browse...", command=lambda: pick()).grid(row=0, column=2)

    ttk.Label(frm, text="CompNet project").grid(row=1, column=0, sticky="w", pady=(8, 0))
    project_box = ttk.Combobox(frm, textvariable=project_var, state="readonly")
    project_box.grid(row=1, column=1, sticky="ew", padx=6, pady=(8, 0))

    ttk.Label(frm, text="Found").grid(row=2, column=0, sticky="nw", pady=(8, 0))
    found = tk.Text(frm, height=7, wrap="word", relief="solid", borderwidth=1)
    found.grid(row=2, column=1, columnspan=2, sticky="ew", padx=6, pady=(8, 0))

    ttk.Label(frm, text="Options").grid(row=3, column=0, sticky="w", pady=(8, 0))
    ttk.Entry(frm, textvariable=options_var).grid(row=3, column=1, columnspan=2, sticky="ew", padx=6, pady=(8, 0))
    ttk.Label(frm, text='Same as the command line, e.g. --rename STN_1=STN1 --exclude "JOB.GSI line 83"',
              foreground="#666").grid(row=4, column=1, columnspan=2, sticky="w", padx=6)

    buttons = ttk.Frame(frm)
    buttons.grid(row=5, column=0, columnspan=3, sticky="w", pady=10)
    run_btn = ttk.Button(buttons, text="Run", command=lambda: start())
    run_btn.pack(side="left")
    open_btn = ttk.Button(buttons, text="Open report", state="disabled",
                          command=lambda: webbrowser.open(state["report"].as_uri()))
    open_btn.pack(side="left", padx=8)

    log = tk.Text(frm, height=9, wrap="word", relief="solid", borderwidth=1, background="#f7f9fb")
    log.grid(row=6, column=0, columnspan=3, sticky="nsew")
    frm.rowconfigure(6, weight=1)
    ttk.Label(frm, text="AUSGeoid2020 grid © Geoscience Australia, CC BY 4.0. The job files are never changed.",
              foreground="#666").grid(row=7, column=0, columnspan=3, sticky="w", pady=(6, 0))

    def say(text: str) -> None:
        log.insert("end", text.rstrip() + "\n")
        log.see("end")

    def show(*_):
        found.delete("1.0", "end")
        text = folder_var.get().strip().strip('"')
        folder = Path(text).resolve() if text else Path(text)
        if not text or not folder.is_dir():
            state["job"] = None
            return
        job = find_job(folder)
        state["job"] = job
        project_box["values"] = [p.name for p in job["projects"]]
        if job["projects"] and project_var.get() not in project_box["values"]:
            project_var.set(job["projects"][0].name)
        mode = "CompNet project" if job["projects"] else ("Raw files" if job["gsi"] or job["asc"] else "nothing to run")
        lines = [f"Mode: {mode}"]
        if job["projects"]:
            lines.append("Projects: " + ", ".join(p.name for p in job["projects"]))
        for key, label in (("gsi", "GSI"), ("asc", "Infinity .asc"), ("control", "Control")):
            if job[key]:
                lines.append(f"{label}: " + ", ".join(p.name for p in job[key]))
        found.insert("end", "\n".join(lines))

    def pick():
        d = filedialog.askdirectory(title="Choose the job folder")
        if d:
            folder_var.set(d)

    def start():
        text = folder_var.get().strip().strip('"')
        folder = Path(text).resolve() if text else Path(text)
        if not text or not folder.is_dir() or not state["job"]:
            messagebox.showwarning("Static Survey", "Choose a job folder first.")
            return
        project = next((p for p in state["job"]["projects"] if p.name == project_var.get()), None)
        try:
            argv = command_for(folder, state["job"], options_var.get(), project)
        except ValueError as e:
            messagebox.showwarning("Static Survey", str(e))
            return
        run_btn["state"] = "disabled"
        say("Running: static-survey " + " ".join(shlex.quote(a) for a in argv))

        def work():
            try:
                printed = run_command(argv)
                o = outputs_for(folder, job_name(folder, state["job"], project))
                html = to_html(o["md"])
                root.after(0, done, printed, html, [o["csv"], o["dxf"]], None)
            except RuntimeError as e:  # the command stopped and said why: show all of it
                root.after(0, done, "", None, [], str(e), str(e).strip())
            except Exception:  # shown to the user, not swallowed
                root.after(0, done, "", None, [], traceback.format_exc())

        threading.Thread(target=work, daemon=True).start()

    def done(printed, html, others, error, message=None):
        run_btn["state"] = "normal"
        if error:
            say(error)
            messagebox.showerror("Static Survey", message or error.strip().splitlines()[-1])
            return
        say(printed)
        say(f"Report: {html}")
        for f in others:
            say(f"Written: {f}")
        state["report"] = html
        open_btn["state"] = "normal"
        webbrowser.open(html.as_uri())

    folder_var.trace_add("write", show)
    show()
    root.mainloop()


def main() -> int:
    # A windowed build has no console: give print() somewhere to go.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")
    if len(sys.argv) > 1 and sys.argv[1] in ("compnet", "raw"):
        from static_survey.cli import main as cli_main

        return cli_main(sys.argv[1:])
    gui()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
