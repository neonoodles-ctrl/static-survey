from static_survey.io.gsi import read_gsi

SAMPLE = (
    "*110001+000000000000STN1 84..10+0000000123456789 85..10+0000001234567891 86..10+0000000000100500 88..10+0000000000001500 \n"
    "*110002+000000000000TGT7 21.324+0000000012345670 22.324+0000000009103250 31..00+0000000000150123 87..10-0000000000001200 19....+0000000004091134 \n"
)


def test_reads_station_and_observation(tmp_path):
    p = tmp_path / "x.gsi"
    p.write_text(SAMPLE.replace("\n", "\r\n"), encoding="latin-1")
    st, ob = read_gsi(p)
    assert st.kind == "station" and st.point == "STN1"
    assert abs(st.e - 123456.789) < 1e-9 and abs(st.hi - 1.5) < 1e-9
    assert ob.point == "TGT7"
    assert abs(ob.hz - (123 + 45 / 60 + 67.0 / 3600)) < 1e-9
    assert abs(ob.zen - (91 + 3 / 60 + 25.0 / 3600)) < 1e-9
    assert abs(ob.slope - 150.123) < 1e-9 and abs(ob.ht + 1.2) < 1e-9
    assert ob.time == "04-09 11:34"
