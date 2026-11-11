import math

from static_survey.adjust import adjust
from static_survey.io.infinity import read_asc
import pytest

from static_survey.adjust import SEC
from static_survey.rawjob import Instrument, RawJob, build, reduce_gsi
from static_survey.io.gsi import read_gsi

from pathlib import Path

import numpy as np

from static_survey import geodesy as G
from static_survey.io.infinity import Baseline
from synthetic import N0, TRUE, write_asc, write_gsi


def test_asc_reader(tmp_path):
    p = tmp_path / "v.asc"
    write_asc(p, (("A", "E"), ("B", "F")))
    bl = read_asc(p)
    assert [(b.frm, b.to) for b in bl] == [("A", "E"), ("B", "F")]
    assert abs(bl[0].cov[0] - 1.0e-7) < 1e-15 and bl[0].start == "01.04.2023 09:00:00"


def test_face_two_is_reduced_to_face_one(tmp_path):
    p = tmp_path / "f.gsi"
    write_gsi(p, stations=("A",), reps=2)
    red = reduce_gsi(read_gsi(p))
    n = len(TRUE) - 1
    for f1, f2 in zip(red[:n], red[n:]):
        assert f1.target == f2.target
        assert abs((f1.hz_deg - f2.hz_deg + 180) % 360 - 180) * 3600 < 0.2
        assert abs(f1.zen_deg - f2.zen_deg) * 3600 < 0.2


def test_rename_maps_field_names_to_project_names(tmp_path):
    """B is booked as BX in the GSI and E as EX in the .asc; --rename joins them up again."""
    g, a = tmp_path / "job.gsi", tmp_path / "job.asc"
    write_gsi(g, stations=("A", "B", "D"))
    g.write_text(g.read_text(encoding="latin-1").replace(f"+{'B':0>16}", f"+{'BX':0>16}"), encoding="latin-1")
    write_asc(a, (("A", "E"),))
    a.write_text(a.read_text(encoding="latin-1").replace(f"@-{'E':<16}", f"@-{'EX':<16}"), encoding="latin-1")
    common = dict(gsi=[g], asc=[a], fixed={"A": TRUE["A"], "C": TRUE["C"]},
                  geoid_table={n: N0 for n in TRUE}, refraction_sec_per_km=(1 - 0.13) * 16.188)

    net, _ = build(RawJob(**common))  # without the rename, BX and EX are extra marks
    names = {s.frm for s in net.sightings} | {s.to for s in net.sightings} | {v.to for v in net.vectors}
    assert {"BX", "EX"} <= names

    net, _ = build(RawJob(rename={"BX": "B", "EX": "E"}, **common))
    names = {s.frm for s in net.sightings} | {s.to for s in net.sightings} | {v.to for v in net.vectors}
    assert not {"BX", "EX"} & names
    res = adjust(net)
    for n, c in TRUE.items():
        assert math.dist(res.coords[n], c) < 0.002, n


def test_file_settings_weight_only_their_own_file(tmp_path):
    """Setup D comes from a second GSI file booked with looser settings; only its readings change."""
    g1, g2, a = tmp_path / "one.gsi", tmp_path / "two.gsi", tmp_path / "job.asc"
    write_gsi(g1, stations=("A", "B"))
    write_gsi(g2, stations=("D",))
    write_asc(a, (("A", "E"),))
    loose = Instrument(pointing_h=4, pointing_v=4, centring_inst=0.002, centring_target=0.002,
                       height_inst=0.005, height_target=0.005)
    job = RawJob(gsi=[g1, g2], asc=[a], fixed={"A": TRUE["A"], "C": TRUE["C"]}, file_instruments={"two.gsi": loose},
                 geoid_table={n: N0 for n in TRUE}, refraction_sec_per_km=(1 - 0.13) * 16.188)
    net, _ = build(job)
    seen = set()
    for s in net.sightings:
        inst = loose if s.ref.startswith("two.gsi") else Instrument()
        seen.add(s.ref.split()[0])
        assert (s.sd_dir, s.sd_zen, s.sd_dist) == pytest.approx(inst.sigmas(s.slope, s.zen)), s.ref
    assert seen == {"one.gsi", "two.gsi"}
    one = next(s for s in net.sightings if s.ref.startswith("one.gsi"))
    two = next(s for s in net.sightings if s.ref.startswith("two.gsi"))
    assert two.sd_dir / SEC > 4 > one.sd_dir / SEC
    res = adjust(net)
    for n, c in TRUE.items():
        assert math.dist(res.coords[n], c) < 0.002, n

    with pytest.raises(ValueError, match="not in the job"):
        build(RawJob(gsi=[g1], fixed=job.fixed, geoid_table=job.geoid_table, file_instruments={"two.gsi": loose}))


def test_raw_files_to_coordinates(tmp_path):
    g, a = tmp_path / "job.gsi", tmp_path / "job.asc"
    write_gsi(g, stations=("A", "B", "D"))
    write_asc(a, (("A", "E"),))
    job = RawJob(gsi=[g], asc=[a], fixed={"A": TRUE["A"], "C": TRUE["C"]},
                 geoid_table={n: N0 for n in TRUE}, refraction_sec_per_km=(1 - 0.13) * 16.188)
    net, info = build(job)
    assert len({s.setup for s in net.sightings}) == 3
    assert any("GNSS vector" in l for l in info["provisional_log"])
    res = adjust(net)
    for n, c in TRUE.items():
        assert math.dist(res.coords[n], c) < 0.002, n


ROOT = Path(__file__).resolve().parents[1]
GNSS_PAIRS = (("A", "B"), ("A", "E"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "F"), ("F", "B"), ("A", "C"))


def _scratch(name):
    d = ROOT / "output" / "scratch" / "pytest-rawjob" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def _baseline(frm, to, lon_from, lon_to, lat=-31.5):
    a, b = (G.geodetic_to_ecef(lat, math.radians(lon), 300.0) for lon in (lon_from, lon_to))
    a, b = (np.atleast_2d(x)[0] for x in (a, b))
    return Baseline(frm, to, tuple(a), tuple(b - a), 1.0, (1e-6, 0, 0, 1e-6, 0, 1e-6), 0.0, 0.0, "", "x.asc")


def test_asc_positions_set_the_zone():
    """A GNSS job in zone 55 computed as zone 56 is tens of metres out with no warning; the .asc says where it is."""
    a = _scratch("zone55") / "job.asc"
    write_asc(a, GNSS_PAIRS, epsg=7855)
    job = RawJob(asc=[a], fixed={"A": TRUE["A"], "C": TRUE["C"]}, geoid_table={n: N0 for n in TRUE})
    net, _ = build(job)
    assert (net.epsg, net.grid) == (7855, "MGA2020 zone 55")
    res = adjust(net)
    for n, c in TRUE.items():
        assert math.dist(res.coords[n], c) < 0.002, n


def test_asc_positions_in_two_zones_or_an_unhandled_zone_stop_the_job():
    from static_survey.rawjob import Unsupported, asc_zone

    assert asc_zone([_baseline("P", "Q", 146.0, 146.1)]) == 55
    assert asc_zone([_baseline("P", "Q", 151.0, 151.1)]) == 56
    with pytest.raises(Unsupported, match=r"more than one MGA zone: zone 55: P; zone 56: Q"):
        asc_zone([_baseline("P", "Q", 149.99, 150.01)])
    with pytest.raises(Unsupported, match=r"MGA zone 57\. static-survey handles"):
        asc_zone([_baseline("P", "Q", 160.0, 160.1)])


def test_the_command_line_stops_on_a_zone_problem_with_a_message(capsys):
    from static_survey import cli

    d = _scratch("two-zones")
    asc, ctl = d / "job.asc", d / "control.csv"
    b = _baseline("P", "Q", 149.99, 150.01)
    asc.write_text("\r\n".join([
        "@+P  {:.4f} {:.4f} {:.4f}".format(*b.from_xyz), "@-Q  {:.4f} {:.4f} {:.4f}".format(*b.dxyz),
        "@=    0.5000   0.00000010  -0.00000005   0.00000006   0.00000008  -0.00000003   0.00000012",
        "@:       1.5000        0.0000 ", "@;       1.6000        0.0000 ", "@*01.04.2023 09:00:00", ""]),
        encoding="latin-1")
    ctl.write_text("P,700000,6500000,300\n", encoding="utf-8")
    assert cli.main(["raw", "--asc", str(asc), "--control", str(ctl), "--geoid-grid", "x.tif", "--job", "T"]) == 2
    err = capsys.readouterr().err
    assert "static-survey stopped without adjusting T" in err and "more than one MGA zone" in err
