"""Run a built program on a made-up raw job and check its coordinates against the Linux reference.

    python tests/windows_smoke.py dist/StaticSurvey.exe     (the Windows build)
    python tests/windows_smoke.py --reference GRID.tif       (print the reference from this machine)

The job is the synthetic 6-mark network (tests/synthetic.py) with the geoid from the AUSGeoid2020 grid, so the
built program must find the grid it ships with.
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from synthetic import TRUE, write_asc, write_gsi  # noqa: E402

# Adjusted E, N, H (m) of the free marks, from Linux (static-survey raw, AUSGeoid2020 grid).
REFERENCE = {
    "B": (297399.9999041283, 6436059.999881429, 162.0002909303809),
    "D": (297759.99978278537, 6435599.999655233, 128.00040195411694),
    "E": (297309.9998346143, 6435639.999748514, 147.00071384417097),
    "F": (297559.9998699582, 6435809.9997916985, 134.9999613563001),
}
TOLERANCE = 1e-5  # 0.01 mm


def make_job(d: Path) -> list[str]:
    write_gsi(d / "job.gsi", stations=("A", "B", "D"))
    write_asc(d / "job.asc", (("A", "E"),))
    (d / "control.csv").write_text("".join(f"{n},{e},{nn},{h}\n" for n, (e, nn, h) in TRUE.items() if n in "AC"))
    return ["raw", "--gsi", str(d / "job.gsi"), "--asc", str(d / "job.asc"), "--control", str(d / "control.csv"),
            "--json", str(d / "result.json"), "--out", str(d / "report.md")]


def run(cmd: list[str], extra: list[str] = ()) -> dict:
    with tempfile.TemporaryDirectory() as t:
        d = Path(t)
        args = make_job(d) + list(extra)
        subprocess.run(cmd + args, check=True, timeout=600, stdout=subprocess.DEVNULL)
        coords = json.loads((d / "result.json").read_text())["coords"]
        assert (d / "report.md").stat().st_size > 1000, "report missing or empty"
    return {n: coords[n] for n in sorted(coords) if n not in "AC"}


def main() -> int:
    if sys.argv[1] == "--reference":
        print(json.dumps(run([sys.executable, "-m", "static_survey.cli"], ["--geoid-grid", sys.argv[2]]), indent=1))
        return 0
    got = run([sys.argv[1]])
    worst = max(abs(got[n][i] - REFERENCE[n][i]) for n in REFERENCE for i in range(3))
    print(f"{len(got)} marks; largest difference from the Linux reference {worst * 1000:.4f} mm")
    return 0 if worst < TOLERANCE and set(got) == set(REFERENCE) else 1


if __name__ == "__main__":
    raise SystemExit(main())
