"""The window: click the sighting cross to choose a job folder, click the vertical circle to run.

The drawing is pre-made, frame by frame (see ``tools/theodolite``); this module places it, plays the moves and
wires the clicks. What happens in which state lives in ``stage.py``.

The rig stands in one of three places: "start" (the target is up), "ready" (the theodolite is out, telescope down
and left) and "done" (turned half round, telescope up and right). The fold move joins the first two, the run move
the last two, and the wait move is the small loop played while an adjustment is still going.
"""

from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path

from static_survey.ui import stage as st

HERE = Path(__file__).resolve().parent
INK, FAINT, PAPER = "#161618", "#77736e", "#ffffff"
CREDIT = "AUSGeoid2020 grid © Geoscience Australia, CC BY 4.0. The job files are never changed."
OVERLAY = {st.START: "overlay_start", st.READY: "overlay_ready", st.RUNNING: "overlay_running",
           st.DONE: "overlay_panel", st.FAILED: "overlay_panel"}


def open_path(path: Path) -> None:
    """Open a file or folder with whatever the computer uses for it."""
    if sys.platform == "win32":
        os.startfile(path)  # noqa: S606 - the user asked to open their own file
    else:
        subprocess.Popen(["open" if sys.platform == "darwin" else "xdg-open", str(path)])


def shorten(text: str, font, room: int) -> str:
    """The text, or its tail behind an ellipsis when it is wider than `room` pixels."""
    if font.measure(text) <= room:
        return text
    while len(text) > 1 and font.measure("…" + text) > room:
        text = text[1:]
    return "…" + text


def run(folder: str | None = None) -> None:
    import tkinter as tk
    from tkinter import filedialog
    from tkinter import font as tkfont

    from static_survey import app

    layout = json.loads((HERE / "layout.json").read_text(encoding="utf-8"))
    width, height = layout["window"]
    root = tk.Tk()
    root.title("Static Survey")
    root.geometry(f"{width}x{height}")
    root.resizable(False, False)
    canvas = tk.Canvas(root, width=width, height=height, background=PAPER, highlightthickness=0)
    canvas.pack()

    have = set(tkfont.families())
    serif = next((f for f in ("Georgia", "Noto Serif", "DejaVu Serif", "Times New Roman") if f in have), "serif")
    mono = next((f for f in ("Consolas", "DejaVu Sans Mono", "Courier New") if f in have), "monospace")

    def sized(family: str, line: int, **style):
        """A font whose lines are `line` pixels high. Tk's own pixel sizes come out wrong on some screens, where
        it and the font library disagree about the resolution, so the size is found by measuring."""
        font = tkfont.Font(family=family, size=-6, **style)
        for size in range(6, 80):
            font.configure(size=-size)
            if font.metrics("linespace") >= line:
                break
        return font

    f_title = sized(serif, 26, slant="italic")
    f_body = sized(serif, 19)
    f_ital = sized(serif, 20, slant="italic")
    f_small = sized(serif, 16, slant="italic")
    f_h2 = sized(serif, 23, weight="bold")
    f_h3 = sized(serif, 20, weight="bold")
    f_mono = sized(mono, 15)

    images = {name[:-4]: tk.PhotoImage(file=str(HERE / name)) for name in layout["images"]}
    count, delay = layout["moves"], round(1000 / layout["fps"])
    still = {st.START: ("fold", 0), st.READY: ("fold", count["fold"] - 1)}      # every other state: the run's end
    frames: dict[str, tk.PhotoImage] = {}
    motion = {"job": None, "at": "start"}       # the next frame's timer, and where the rig stands
    rig_item = canvas.create_image(*layout["rig_xy"], anchor="nw")
    overlay_item = canvas.create_image(0, 0, anchor="nw")
    canvas.create_text(40, height - 10, anchor="sw", text=CREDIT, font=f_small, fill=FAINT)

    stage = st.Stage()
    project_var = tk.StringVar()
    options_var = tk.StringVar()
    view = {"log": [], "details": False, "report": "", "closed": False}

    # ---- the strip under the rig: folder, project, options
    fx0, fx1, fy = layout["fields"]["folder"]
    px0, px1, py = layout["fields"]["project"]
    ox0, ox1, oy = layout["fields"]["options"]
    folder_text = canvas.create_text(fx0, fy, anchor="w", font=f_ital, fill=INK, tags="strip")
    project_text = canvas.create_text(px0, py, anchor="w", font=f_ital, fill=INK, tags=("strip", "project"))
    options_box = tk.Entry(canvas, textvariable=options_var, relief="flat", borderwidth=0, highlightthickness=0,
                           background=PAPER, foreground=INK, insertbackground=INK, font=f_body,
                           disabledbackground=PAPER, disabledforeground=FAINT)
    options_item = canvas.create_window(ox0, oy, anchor="w", width=ox1 - ox0, height=22, window=options_box,
                                        tags="strip")
    caption_text = canvas.create_text(*layout["caption"], anchor="nw", font=f_ital, fill=INK, width=520, tags="ready")
    note_text = canvas.create_text(width - 60, 420, anchor="e", font=f_ital, fill=INK, width=520, tags="start")

    # ---- the panel in the telescope's line of sight
    x0, y0, x1, y1 = layout["panel"]
    title_text = canvas.create_text(x0 + 18, y0 + 26, anchor="w", font=f_title, fill=INK, tags="panel")
    details_link = canvas.create_text(x1 - 18, y0 + 28, anchor="e", font=f_ital, fill=INK, tags=("panel", "link"))
    body = tk.Text(canvas, wrap="word", relief="flat", borderwidth=0, highlightthickness=0, background=PAPER,
                   foreground=INK, font=f_body, padx=4, pady=4, cursor="arrow", spacing1=1, spacing3=3)
    scroll = tk.Scrollbar(canvas, orient="vertical", command=body.yview, width=10, relief="flat", borderwidth=0,
                          background="#cfcbc4", activebackground="#a8a39b", troughcolor=PAPER, highlightthickness=0,
                          elementborderwidth=0)
    body.configure(yscrollcommand=scroll.set)
    canvas.create_window(x0 + 14, y0 + 50, anchor="nw", width=x1 - x0 - 40, height=y1 - y0 - 100, window=body,
                         tags="panel")
    canvas.create_window(x1 - 24, y0 + 50, anchor="nw", width=10, height=y1 - y0 - 100, window=scroll, tags="panel")
    body.tag_configure("h1", font=f_h2, spacing1=4, spacing3=6)
    body.tag_configure("h2", font=f_h2, spacing1=12, spacing3=4)
    body.tag_configure("h3", font=f_h3, spacing1=10, spacing3=3)
    body.tag_configure("li", lmargin1=10, lmargin2=24)
    body.tag_configure("table", font=f_mono, wrap="none", spacing1=0, spacing3=0)
    body.tag_configure("log", font=f_mono)
    body.bind("<Shift-MouseWheel>", lambda e: body.xview_scroll(-1 if e.delta > 0 else 1, "units"))
    links: dict[str, int] = {}

    def lay_links(opens: list[str], actions: list[str]) -> int:
        """The things to open along the foot of the panel; what to do next beside the title."""
        for item in links.values():
            canvas.delete(item)
        links.clear()
        x = x0 + 18
        for name in opens:
            links[name] = canvas.create_text(x, y1 - 22, anchor="w", text=name, font=f_ital, fill=INK, tags="panel")
            x = canvas.bbox(links[name])[2] + 22
        x = canvas.bbox(details_link)[0] - 22
        for name in reversed(actions):
            links[name] = canvas.create_text(x, y0 + 28, anchor="e", text=name, font=f_ital, fill=INK, tags="panel")
            x = canvas.bbox(links[name])[0] - 22
        return x - (x0 + 18)                               # the room left for the title

    # ---- the rig and its moves
    def put(move: str, i: int) -> None:
        name = f"{move}_{i:03d}"
        if name not in frames:
            if len(frames) > 8:                 # the one on show is always the newest, so the oldest can go
                frames.pop(next(iter(frames)))
            frames[name] = tk.PhotoImage(file=str(HERE / f"{name}.png"))
        canvas.itemconfigure(rig_item, image=frames[name])

    def moving() -> bool:
        return motion["job"] is not None

    def play(move: str, order: list[int], then) -> None:
        """Show a move's frames in `order`, one every `delay` ms by the clock, then call `then`."""
        began = time.monotonic()

        def step(k: int) -> None:
            put(move, order[k])
            if k + 1 < len(order):         # timed from the start, so the time a frame takes to load is not added
                wait = began + (k + 1) * delay / 1000 - time.monotonic()
                motion["job"] = root.after(max(1, round(wait * 1000)), step, k + 1)
            else:
                motion["job"] = None
                then()

        motion["job"] = "first frame"
        step(0)

    def backwards(move: str) -> list[int]:
        """A move in reverse at double speed, ending on its first frame."""
        return list(range(count[move] - 1, 0, -2)) + [0]

    def curtain(strip: bool) -> None:
        """While the rig moves: only the drawing, and the strip when a job is loaded."""
        canvas.itemconfigure(overlay_item, image=images["overlay_strip"] if strip else "")
        for tag in ("start", "ready", "panel"):
            canvas.itemconfigure(tag, state="hidden")
        canvas.itemconfigure("strip", state="normal" if strip else "hidden")
        canvas.configure(cursor="")

    def arrive(at: str, then) -> None:
        motion["at"] = at
        then()

    def settle() -> None:
        """After the run move, and after each turn of the waiting loop: wait on, or show what came back."""
        show()
        if stage.state == st.RUNNING:
            play("wait", list(range(count["wait"])), settle)

    # ---- drawing each state
    def fill_body() -> None:
        body.configure(state="normal")
        body.delete("1.0", "end")
        if view["details"]:
            body.insert("end", "\n".join(view["log"]) + "\n", "log")
        elif stage.state == st.FAILED:
            body.insert("end", stage.error.strip() + "\n")
        else:
            for kind, content in st.blocks(view["report"]):
                if kind == "table":
                    body.insert("end", "\n".join(st.table_lines(content)) + "\n\n", "table")
                elif kind == "li":
                    body.insert("end", "•  " + content + "\n", "li")
                else:
                    body.insert("end", content + "\n", kind if kind != "p" else ())
        body.configure(state="disabled")
        canvas.itemconfigure(details_link, text="report" if view["details"] else "details")

    def show() -> None:
        state = stage.state
        put(*still.get(state, ("run", count["run"] - 1)))
        canvas.itemconfigure(overlay_item, image=images[OVERLAY[state]])
        for tag, on in (("start", state == st.START), ("ready", state == st.READY), ("strip", state != st.START),
                        ("panel", state in (st.DONE, st.FAILED))):
            canvas.itemconfigure(tag, state="normal" if on else "hidden")
        canvas.itemconfigure(note_text, text=stage.note)
        if state == st.START:
            return
        job = stage.job
        canvas.itemconfigure(folder_text, text=shorten(str(stage.folder), f_ital, fx1 - fx0))
        names = [p.name for p in job["projects"]]
        if names and project_var.get() not in names:
            project_var.set(names[0])
        canvas.itemconfigure(project_text, text=shorten(project_var.get(), f_ital, px1 - px0 - 24) if names else "raw files",
                             fill=INK if names else FAINT)
        options_box.configure(state="disabled" if state == st.RUNNING else "normal")
        canvas.itemconfigure(caption_text, text="\n".join(st.found_lines(job)))
        if state == st.DONE:
            fill_body()
            room = lay_links(["open report", "open drawing", "open coordinates", "open job folder"],
                             ["run again", "new job"])
            canvas.itemconfigure(title_text, text=shorten(app.job_name(stage.folder, job, chosen_project()), f_title,
                                                           room))
        elif state == st.FAILED:
            fill_body()
            lay_links([], ["run again", "new job"])
            canvas.itemconfigure(title_text, text="The run stopped")

    # ---- actions
    def chosen_project():
        return next((p for p in stage.job["projects"] if p.name == project_var.get()), None)

    def take(folder_text_value: str) -> None:
        text = folder_text_value.strip().strip('"')
        path = Path(text).resolve() if text else None
        if path is None or not path.is_dir() or moving():
            return
        if not stage.choose(path, app.find_job(path)):
            return show()
        project_var.set("")
        view.update(log=[], details=False, report="")
        curtain(False)
        play("fold", list(range(count["fold"])), lambda: arrive("ready", show))

    def browse() -> None:
        chosen = filedialog.askdirectory(title="Choose the job folder")
        if chosen:
            take(chosen)

    def pick_project(event) -> None:
        names = [p.name for p in stage.job["projects"]] if stage.job else []
        if len(names) < 2 or stage.state == st.RUNNING:
            return
        menu = tk.Menu(root, tearoff=0, font=f_ital, background=PAPER, foreground=INK)
        for name in names:
            menu.add_command(label=name, command=lambda n=name: (project_var.set(n), show()))
        menu.tk_popup(event.x_root, event.y_root)

    def start() -> None:
        folder_path, job, project = stage.folder, stage.job, chosen_project()
        argv, problem = [], ""
        try:
            argv = app.command_for(folder_path, job, options_var.get(), project)
        except ValueError as exc:          # an option the command line turns down: say so in the panel
            problem = str(exc)
        if moving() or not stage.run():
            return

        def work() -> None:
            try:
                printed = app.run_command(argv)
                out = app.outputs_for(folder_path, app.job_name(folder_path, job, project))
                out["html"] = app.to_html(out["md"])
                result = (printed, out, None)
            except RuntimeError as exc:    # the command stopped and said why: show all of it
                result = ("", None, str(exc))
            except Exception:              # shown to the user, not swallowed
                result = ("", None, traceback.format_exc())
            if not view["closed"]:
                root.after(0, finished, *result)

        view.update(details=False, report="", log=["Running: static-survey " + " ".join(shlex.quote(a) for a in argv)])
        options_box.configure(state="disabled")
        if problem:
            view["log"] = [problem]
            stage.fail(problem)
        else:
            threading.Thread(target=work, daemon=True).start()
        if motion["at"] == "ready":
            curtain(True)
            play("run", list(range(count["run"])), lambda: arrive("done", settle))
        else:
            settle()

    def finished(printed: str, out: dict | None, error: str | None) -> None:
        if error:
            view["log"].append(error.rstrip())
            stage.fail(error)
        else:
            view["log"] += [printed.rstrip()] + [f"Written: {out[k]}" for k in ("html", "md", "csv", "dxf")]
            view["report"] = out["md"].read_text(encoding="utf-8")
            stage.finish(out)
        if not moving():                   # otherwise the move in hand ends in settle(), which shows it
            show()

    def follow(name: str) -> None:
        out = stage.outputs
        if name == "open report":
            webbrowser.open(out["html"].as_uri())
        elif name == "open drawing":
            open_path(out["dxf"])
        elif name == "open coordinates":
            open_path(out["csv"])
        elif name == "open job folder":
            open_path(stage.folder)
        elif name == "run again":
            start()
        elif name == "new job" and stage.reset():
            options_var.set("")
            curtain(False)
            fold_away = lambda: play("fold", backwards("fold"), lambda: arrive("start", show))  # noqa: E731
            if motion["at"] == "done":
                play("run", backwards("run"), fold_away)
            else:
                fold_away()

    # ---- clicks and the pointer
    def link_at(x: float, y: float) -> str | None:
        for name, item in list(links.items()) + [("details", details_link)]:
            box = canvas.bbox(item)
            if box and canvas.itemcget(item, "state") != "hidden" and box[0] <= x <= box[2] and box[1] <= y <= box[3]:
                return name
        return None

    def live_at(x: float, y: float) -> bool:
        part = stage.live_part()
        return bool(part) and st.inside(layout["live"][part], x, y)

    def click(event) -> None:
        if moving():
            return
        name = link_at(event.x, event.y) if stage.state in (st.DONE, st.FAILED) else None
        if name == "details":
            view["details"] = not view["details"]
            fill_body()
        elif name:
            follow(name)
        elif live_at(event.x, event.y):
            browse() if stage.state == st.START else start()

    def move(event) -> None:
        if moving():
            return
        over = live_at(event.x, event.y) or (stage.state in (st.DONE, st.FAILED) and link_at(event.x, event.y))
        box = canvas.bbox(project_text)
        over = over or (stage.state not in (st.START, st.RUNNING) and bool(stage.job and len(stage.job["projects"]) > 1)
                        and box[0] <= event.x <= px1 and abs(event.y - py) < 14)
        canvas.configure(cursor="hand2" if over else "")

    def close() -> None:
        view["closed"] = True
        app.stop_running()                 # a run in hand is a process of its own: do not leave it going
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", close)
    canvas.bind("<Button-1>", click)
    canvas.bind("<Motion>", move)
    canvas.tag_bind("project", "<Button-1>", pick_project)

    show()
    if folder:
        take(folder)
    root.mainloop()
