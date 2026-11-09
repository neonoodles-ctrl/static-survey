import math

from static_survey.adjust import adjust
from static_survey.io.infinity import read_asc
import pytest

from static_survey.adjust import SEC
from static_survey.rawjob import Instrument, RawJob, build, reduce_gsi
from static_survey.io.gsi import read_gsi

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
