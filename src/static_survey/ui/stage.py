"""What the window is doing and what it shows, kept free of Tk so it can be tested without a screen."""

from __future__ import annotations

import re

START, READY, RUNNING, DONE, FAILED = "start", "ready", "running", "done", "failed"
# The part of the drawing that takes a click in each state.
LIVE_PART = {START: "sighting_cross", READY: "vertical_circle"}


def runnable(job: dict) -> bool:
    """A folder can be run when it holds a CompNet project or raw observation files."""
    return bool(job["projects"] or job["gsi"] or job["asc"])


def found_lines(job: dict) -> list[str]:
    """What the folder holds, one line per kind of file."""
    mode = "CompNet project" if job["projects"] else ("Raw files" if job["gsi"] or job["asc"] else "nothing to run")
    lines = [f"Mode: {mode}"]
    if job["projects"]:
        lines.append("Projects: " + ", ".join(p.name for p in job["projects"]))
    for key, label in (("gsi", "GSI"), ("asc", "Infinity .asc"), ("control", "Control")):
        if job[key]:
            lines.append(f"{label}: " + ", ".join(p.name for p in job[key]))
    return lines


def inside(ellipse: list[float], x: float, y: float) -> bool:
    cx, cy, rx, ry = ellipse
    return ((x - cx) / rx) ** 2 + ((y - cy) / ry) ** 2 <= 1.0


class Stage:
    """start -> ready (a folder is chosen) -> running -> done or failed. New job goes back to start."""

    def __init__(self) -> None:
        self.state = START
        self.folder = None
        self.job: dict | None = None
        self.note = ""            # why a chosen folder was turned down
        self.error = ""
        self.outputs: dict = {}

    def live_part(self) -> str | None:
        return LIVE_PART.get(self.state)

    def choose(self, folder, job: dict) -> bool:
        """Take a job folder. A folder with nothing to run is turned down and says why."""
        if self.state == RUNNING:
            return False
        if not runnable(job):
            self.note = f"Nothing to run in {folder}: no CompNet project (.TRV or .GPS) and no GSI or .asc files."
            return False
        self.folder, self.job, self.note, self.error, self.outputs = folder, job, "", "", {}
        self.state = READY
        return True

    def run(self) -> bool:
        if self.state not in (READY, DONE, FAILED) or self.job is None:
            return False
        self.error, self.outputs = "", {}
        self.state = RUNNING
        return True

    def finish(self, outputs: dict) -> None:
        if self.state == RUNNING:
            self.outputs, self.state = outputs, DONE

    def fail(self, message: str) -> None:
        if self.state == RUNNING:
            self.error, self.state = message, FAILED

    def reset(self) -> bool:
        if self.state == RUNNING:
            return False
        self.__init__()
        return True


def _plain(text: str) -> str:
    """Markdown emphasis and code marks removed; the words stay."""
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"(?<!\*)\*(?!\s)(.+?)(?<!\s)\*(?!\*)", r"\1", text)
    return text.replace("`", "")


def blocks(markdown: str) -> list[tuple]:
    """The report as ('h1'|'h2'|'h3'|'p'|'li', text) and ('table', rows) blocks, in order."""
    out: list[tuple] = []
    para: list[str] = []
    table: list[list[str]] = []

    def flush() -> None:
        if para:
            out.append(("p", _plain(" ".join(para))))
            para.clear()
        if table:
            out.append(("table", [row for row in table]))
            table.clear()

    for raw in markdown.splitlines():
        line = raw.rstrip()
        if line.startswith("|"):
            if para:
                flush()
            cells = [_plain(c.strip()) for c in line.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-{2,}:?", c) for c in cells):      # the rule under the header row
                table.append(cells)
            continue
        if table:
            flush()
        heading = re.match(r"(#{1,3})\s+(.*)", line)
        item = re.match(r"\s*[-*]\s+(.*)", line)
        if heading:
            flush()
            out.append((f"h{len(heading.group(1))}", _plain(heading.group(2))))
        elif item:
            flush()
            out.append(("li", _plain(item.group(1))))
        elif not line.strip():
            flush()
        else:
            para.append(line.strip())
    flush()
    return out


def table_lines(rows: list[list[str]]) -> list[str]:
    """A table as fixed-width text: columns padded to their widest cell, a rule under the first row."""
    if not rows:
        return []
    n = max(len(r) for r in rows)
    rows = [r + [""] * (n - len(r)) for r in rows]
    width = [max(len(r[i]) for r in rows) for i in range(n)]
    lines = ["  ".join(c.ljust(w) for c, w in zip(r, width)).rstrip() for r in rows]
    lines.insert(1, "  ".join("-" * w for w in width))
    return lines
