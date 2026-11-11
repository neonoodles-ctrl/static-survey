"""The window's states, click areas and report text. No screen is needed: Tk is never started here."""
import json
from pathlib import Path

from static_survey.ui import stage as st

UI = Path(st.__file__).resolve().parent
JOB = dict(projects=[Path("/jobs/a/JOB1"), Path("/jobs/a/JOB2")], gsi=[], asc=[Path("/jobs/a/x.asc")],
           control=[Path("/jobs/a/JOB1.FIX")])
EMPTY = dict(projects=[], gsi=[], asc=[], control=[Path("/jobs/b/only.FIX")])


def test_the_live_part_follows_the_state():
    s = st.Stage()
    assert (s.state, s.live_part()) == (st.START, "sighting_cross")
    assert s.choose(Path("/jobs/a"), JOB)
    assert (s.state, s.live_part()) == (st.READY, "vertical_circle")
    assert s.run()
    assert (s.state, s.live_part()) == (st.RUNNING, None)
    s.finish({"md": Path("/jobs/a/JOB1-report.md")})
    assert (s.state, s.live_part()) == (st.DONE, None)
    assert s.outputs["md"].name == "JOB1-report.md"


def test_a_folder_with_nothing_to_run_is_turned_down_and_says_why():
    s = st.Stage()
    assert not s.choose(Path("/jobs/b"), EMPTY)
    assert s.state == st.START and s.job is None
    assert "/jobs/b" in s.note and ".TRV" in s.note


def test_run_needs_a_folder_and_cannot_start_twice():
    s = st.Stage()
    assert not s.run()
    s.choose(Path("/jobs/a"), JOB)
    assert s.run()
    assert not s.run()
    assert not s.choose(Path("/jobs/a"), JOB), "a new folder cannot be taken while a run is going"
    assert not s.reset()


def test_a_failed_run_keeps_its_message_and_can_run_again_or_start_over():
    s = st.Stage()
    s.choose(Path("/jobs/a"), JOB)
    s.run()
    s.fail("The project is not in MGA zone 56.")
    assert (s.state, s.error) == (st.FAILED, "The project is not in MGA zone 56.")
    assert s.run() and s.error == ""
    s.fail("again")
    assert s.reset()
    assert (s.state, s.folder, s.job, s.error) == (st.START, None, None, "")


def test_finish_and_fail_do_nothing_unless_a_run_is_going():
    s = st.Stage()
    s.choose(Path("/jobs/a"), JOB)
    s.finish({"md": Path("x")})
    s.fail("no")
    assert (s.state, s.outputs, s.error) == (st.READY, {}, "")


def test_found_lines_name_what_the_folder_holds():
    assert st.found_lines(JOB) == ["Mode: CompNet project", "Projects: JOB1, JOB2", "Infinity .asc: x.asc",
                                   "Control: JOB1.FIX"]
    assert st.found_lines(EMPTY) == ["Mode: nothing to run", "Control: only.FIX"]
    raw = dict(projects=[], gsi=[Path("a.GSI")], asc=[], control=[])
    assert st.found_lines(raw) == ["Mode: Raw files", "GSI: a.GSI"]


def test_inside_is_an_ellipse_not_its_box():
    e = [100.0, 50.0, 40.0, 20.0]
    assert st.inside(e, 100, 50) and st.inside(e, 139, 50) and st.inside(e, 100, 31)
    assert not st.inside(e, 141, 50)
    assert not st.inside(e, 135, 68), "the corner of the box is outside the ellipse"


REPORT = """# JOB1: adjustment check

Two lines of
one paragraph with **bold** and `code` and 3 * 4.

## Findings
- First *finding*
- Second finding

| Mark | dE (mm) | Note |
|---|---:|:--|
| STN097 | 0.31 | ok |
| T0B7R | 11.18 |
"""


def test_blocks_keep_the_reports_order_and_drop_the_markdown_marks():
    got = st.blocks(REPORT)
    assert got[0] == ("h1", "JOB1: adjustment check")
    assert got[1] == ("p", "Two lines of one paragraph with bold and code and 3 * 4.")
    assert got[2:5] == [("h2", "Findings"), ("li", "First finding"), ("li", "Second finding")]
    assert got[5] == ("table", [["Mark", "dE (mm)", "Note"], ["STN097", "0.31", "ok"], ["T0B7R", "11.18"]])
    assert len(got) == 6


def test_table_lines_are_aligned_and_ragged_rows_are_padded():
    lines = st.table_lines([["Mark", "dE (mm)", "Note"], ["STN097", "0.31", "ok"], ["T0B7R", "11.18"]])
    assert lines == ["Mark    dE (mm)  Note", "------  -------  ----", "STN097  0.31     ok", "T0B7R   11.18"]
    assert st.table_lines([]) == []


def test_the_window_art_is_all_there():
    layout = json.loads((UI / "layout.json").read_text())
    frames = [f"{move}_{i:03d}.png" for move, n in layout["moves"].items() for i in range(n)]
    assert set(layout["moves"]) == {"fold", "run", "wait"} and len(frames) > 100
    for name in layout["images"] + frames:
        assert (UI / name).read_bytes()[:8] == b"\x89PNG\r\n\x1a\n", name
    assert set(layout["live"]) == set(st.LIVE_PART.values())
    w, h = layout["window"]
    for cx, cy, rx, ry in layout["live"].values():
        assert rx > 10 and ry > 10 and 0 < cx - rx and cx + rx < w and 0 < cy - ry and cy + ry < h


def test_an_option_the_command_line_turns_down_comes_back_as_its_message():
    """argparse leaves by SystemExit, which once slipped past the window and left it waiting for ever."""
    import pytest

    from static_survey import app

    with pytest.raises(RuntimeError) as stopped:
        app.run_command(["compnet", "/no/such/job", "--no-such-option"])
    assert "--no-such-option" in str(stopped.value)


def test_the_adjustment_runs_in_its_own_process(monkeypatch):
    """Run inside the window's process, the adjustment held up the drawing: a 5 s move took 8.3 s."""
    from static_survey import app, cli

    def in_this_process(argv):
        raise AssertionError("the command line ran inside the window's process")

    monkeypatch.setattr(cli, "main", in_this_process)
    assert "usage:" in app.run_command(["compnet", "--help"])


def test_closing_the_window_stops_a_run_in_hand(monkeypatch):
    import sys
    import threading
    import time

    from static_survey import app

    monkeypatch.setattr(app, "self_command", lambda: [sys.executable, "-c", "import time; time.sleep(60)"])
    ended = []

    def work():
        try:
            app.run_command(["compnet"])
        except RuntimeError as exc:
            ended.append(str(exc))

    run = threading.Thread(target=work)
    run.start()
    deadline = time.monotonic() + 10
    while not app.stop_running() and time.monotonic() < deadline:      # False until the other process is up
        time.sleep(0.02)
    run.join(10)
    assert not run.is_alive() and len(ended) == 1
    assert app.stop_running() is False                                 # nothing left to stop
