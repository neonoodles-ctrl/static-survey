"""The CompNet reader stops, with a plain message, on projects it cannot adjust faithfully: another MGA zone or no
zone at all, observation files it does not read, and listings with observation types it does not read."""
import math
import shutil
import struct
from pathlib import Path

import pytest

from static_survey.io.compnet import (listing_projection, project_zone, read_control, read_out, read_project,
                                     to_network, unsupported)

ROOT = Path(__file__).resolve().parents[1]
TRV = """*GENERAL
1 2 0.000000 100.0000 90.000000 0.000000 1.500,1.600 2.0 0.002 3.0
1 3 90.000000 120.0000 90.000000 0.000000 1.500,1.600 2.0 0.002 3.0
0 0 0 0 0 0 0,0 0 0 0
"""
PTS = '1 "A"\n2 "B"\n3 "C"\n'
FIX = '1 300000.000 6400000.000 100.000 "A"\n'
N = "1 25.000\n"
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
    for ext, text in ((".TRV", TRV), (".PTS", PTS), (".FIX", FIX), (".N", N)):
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


def test_zone_55_in_the_listing_header_is_used(job):
    job.with_suffix(".OUT").write_text(HEADER.format(proj="MGA2020 ZONE 55 PROJECTION"), encoding="latin-1")
    assert listing_projection(job.with_suffix(".OUT")) == "MGA2020 ZONE 55 PROJECTION"
    assert not any("zone" in p for p in check(job)[0])   # (a header-only listing fails the count check)
    net = to_network(read_project(job))
    assert (net.epsg, net.grid) == (7855, "MGA2020 zone 55")


def test_mga94_header_uses_the_same_projection(job):
    job.with_suffix(".OUT").write_text(HEADER.format(proj="MGA94 ZONE 54 PROJECTION"), encoding="latin-1")
    assert not any("zone" in p for p in check(job)[0])
    net = to_network(read_project(job))
    assert (net.epsg, net.grid) == (7854, "MGA94 zone 54")


def test_zone_outside_49_to_56_stops(job):
    job.with_suffix(".OUT").write_text(HEADER.format(proj="MGA2020 ZONE 58 PROJECTION"), encoding="latin-1")
    problems, _ = check(job)
    assert any("MGA zone 58" in p and "zones 49 to 56" in p for p in problems)


def test_zone_from_the_reference_position(job):
    gpd(job, -31.5, 146.0)                      # zone 55
    assert project_zone(job)[0] == 55 and check(job) == ([], [])
    assert to_network(read_project(job)).grid == "MGA zone 55"   # no listing, so no datum name


def test_no_geoid_stops(job):
    gpd(job, -32.2, 150.85)
    job.with_suffix(".N").unlink()
    problems, _ = check(job)
    assert len(problems) == 1 and "no geoid separation N" in problems[0]


def test_std_zero_holds_a_component_and_100_frees_it(job):
    """CompNet prints FIX for a .STD standard deviation of 0 and FREE for 100."""
    std = job.with_suffix(".STD")
    std.write_text('2 300400.000 6400060.000 112.000 0.000 0.000 100.000 "B"\n'
                   '3 300820.000 6400010.000 101.000 100.000 100.000 0.010 "C"\n', encoding="latin-1")
    proj = read_project(job)
    assert proj.control["B"].sd == (0.0, 0.0, None) and proj.control["C"].sd == (None, None, 0.01)
    _, weighted = read_control(std)
    assert weighted["B"][1] == (0.0, 0.0, None)


def test_listing_reads_fix_and_free_corrections():
    out = HEADER.format(proj="MGA2020 ZONE 55 PROJECTION") + """
                            WEIGHTED COORDINATE CORRECTIONS
    DESCRIPTION    EAST      vE      SR E       NORTH       vN      SR N      HT      vH     SR H

      B        300400.0000  0.0000    FIX   6400060.0000  0.0000    FIX   112.0000  -0.0909   FREE
      C        300820.0000  1.2605   FREE   6400010.0000  1.2370   FREE   101.0000   0.0203   2.03
"""
    path = ROOT / "output" / "scratch" / "pytest-compnet-checks" / "fixfree.OUT"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(out, encoding="latin-1")
    try:
        w = read_out(path).weighted
    finally:
        path.unlink()
    assert (w["B"]["sr_e"], w["B"]["sr_h"], w["B"]["vh"]) == ("FIX", "FREE", -0.0909)
    assert (w["C"]["sr_e"], w["C"]["sr_h"], w["C"]["vh"]) == ("FREE", 2.03, 0.0203)


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
