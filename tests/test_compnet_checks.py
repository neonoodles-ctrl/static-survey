"""The CompNet reader stops, with a plain message, on projects it cannot adjust faithfully: another MGA zone or no
zone at all, observation files it does not read, and listings with observation types it does not read."""
import math
import shutil
import struct
from pathlib import Path

import pytest

from static_survey.io.compnet import listing_projection, project_zone, read_out, read_project, unsupported

ROOT = Path(__file__).resolve().parents[1]
TRV = """*GENERAL
1 2 0.000000 100.0000 90.000000 0.000000 1.500,1.600 2.0 0.002 3.0
1 3 90.000000 120.0000 90.000000 0.000000 1.500,1.600 2.0 0.002 3.0
0 0 0 0 0 0 0,0 0 0 0
"""
PTS = '1 "A"\n2 "B"\n3 "C"\n'
FIX = '1 300000.000 6400000.000 100.000 "A"\n'
HEADER = """                                   PROJECT: JOB
                  COMPNET V3.1 CONSTRAINED THREE DIMENSIONAL ADJUSTMENT
                              {proj}
"""


@pytest.fixture
def job(request):
    """A tiny CompNet project in the project's own scratch folder (never /tmp)."""
    d = ROOT / "output" / "scratch" / "pytest-compnet-checks" / request.node.name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    stem = d / "JOB"
    for ext, text in ((".TRV", TRV), (".PTS", PTS), (".FIX", FIX)):
        stem.with_suffix(ext).write_text(text, encoding="latin-1")
    yield stem
    shutil.rmtree(d, ignore_errors=True)


def gpd(stem, lat, lon):
    stem.with_suffix(".GPD").write_bytes(struct.pack("<2f", math.radians(lat), math.radians(lon)))


def check(stem):
    out = read_out(stem.with_suffix(".OUT")) if stem.with_suffix(".OUT").exists() else None
    return unsupported(stem, read_project(stem), out)


def test_clean_project_passes(job):
    gpd(job, -32.2, 150.85)
    assert project_zone(job)[0] == 56
    assert check(job) == ([], [])


def test_zone_55_in_the_listing_header_stops(job):
    job.with_suffix(".OUT").write_text(HEADER.format(proj="MGA2020 ZONE 55 PROJECTION"), encoding="latin-1")
    assert listing_projection(job.with_suffix(".OUT")) == "MGA2020 ZONE 55 PROJECTION"
    problems, _ = check(job)
    assert any("MGA zone 55" in p for p in problems)


def test_zone_from_the_reference_position(job):
    gpd(job, -31.5, 146.0)                      # zone 55
    problems, _ = check(job)
    assert project_zone(job)[0] == 55 and any("MGA zone 55" in p and ".GPD" in p for p in problems)


def test_no_zone_stops(job):
    problems, _ = check(job)
    assert any("does not say which MGA zone" in p for p in problems)


def test_a_projection_that_is_not_mga_stops(job):
    job.with_suffix(".OUT").write_text(HEADER.format(proj="ISG ZONE 56/1 PROJECTION"), encoding="latin-1")
    problems, _ = check(job)
    assert any("not an MGA zone" in p for p in problems)


def test_an_unread_observation_file_stops_and_a_note_does_not(job):
    gpd(job, -32.2, 150.85)
    job.with_suffix(".LVL").write_text("1 2 0.5231 100.0 0.002\n", encoding="latin-1")
    job.with_suffix(".TXT").write_text("notes about the job\n", encoding="latin-1")
    problems, unread = check(job)
    assert len(problems) == 1 and "JOB.LVL" in problems[0] and "levelling" in problems[0]
    assert unread == ["JOB.TXT"]


def test_gnss_copy_without_the_gnss_file_stops(job):
    gpd(job, -32.2, 150.85)
    job.with_suffix(".GPP").write_text("1 2 1001 5357.087 1128.920 -60.307 0.016 0.021\n", encoding="latin-1")
    problems, _ = check(job)
    assert len(problems) == 1 and "JOB.GPP" in problems[0] and "JOB.GPS" in problems[0]


def test_listing_with_observations_this_reader_does_not_read_stops(job):
    listing = HEADER.format(proj="MGA2020 ZONE 56 PROJECTION") + """
                                       SLOPE DISTANCES
     FROM             TO          COMBINED   OBS SLOPE   FIN SLOPE   RESIDUAL    STANDARDISED
                                  SCALE     DISTANCE    DISTANCE                 RESIDUAL

           A              B      1.000067    100.0000    100.0000    0.0000       0.00
           A              C      1.000067    120.0000    120.0000    0.0000       0.00
                                     HEIGHT DIFFERENCES
     FROM             TO          OBSERVED      FINAL     RESIDUAL    STANDARDISED
           A              B      1.000067      0.5231      0.5230   -0.0001       0.05

                                OBSERVATIONAL VARIANCES
                   Slope distance: 0.05   F TEST: 1.21   PASS
                        Levelling: 0.50   F TEST: 1.21   PASS
"""
    job.with_suffix(".OUT").write_text(listing, encoding="latin-1")
    out = read_out(job.with_suffix(".OUT"))
    assert len(out.distances) == 2              # the height differences are not read as distances
    problems, _ = check(job)
    assert any("HEIGHT DIFFERENCES" in p for p in problems)
    assert any("Levelling" in p for p in problems)
    assert any("does not match the project files" in p and "0 directions against 2" in p for p in problems)


def test_command_stops_without_writing_a_report(job, capsys):
    from static_survey.cli import main

    report = job.with_name("JOB-report.md")
    assert main(["compnet", str(job), "--out", str(report)]) == 2
    assert not report.exists()
    assert "stopped without adjusting JOB" in capsys.readouterr().err
