import math

import pytest

from static_survey.adjust import adjust
from static_survey.adjust import SEC
from static_survey.diagnose import (bad_pointings, distance_constants, global_test, orientation_changes,
                                    swapped_names, unchecked_marks, vertical_check)

# The synthetic noise is all per reading (2" directions, 3" zeniths), so all of it is pointing.
SYN = lambda s: (2.0, 3.0)  # noqa: E731
from static_survey.io.compnet import dms_to_deg

from synthetic import TRUE, build


def test_exact_observations_recover_truth():
    res = adjust(build(noise=False))
    for n, c in TRUE.items():
        for i in range(3):
            assert abs(res.coords[n][i] - c[i]) < 1e-4, (n, i)
    assert res.vf < 1e-6


def test_noisy_network_is_unbiased():
    """With correct weights the variance factor averages 1 (single runs miss the 5% band 1 time in 20)."""
    vfs = []
    for seed in range(1, 13):
        res = adjust(build(seed=seed))
        for n, c in TRUE.items():
            assert math.dist(res.coords[n][:2], c[:2]) < 0.004
        vfs.append(res.vf)
    assert 0.9 < sum(vfs) / len(vfs) < 1.1
    assert global_test(adjust(build(seed=12))).evidence["two_sided_pass"]


def test_twist_is_found_as_a_step_at_the_right_reading():
    net = build(seed=4, twist=("B", 6, 12.0))
    res = adjust(net)
    found = [f for f in orientation_changes(net, res) if f.evidence["station"] == "B"]
    assert found, "twist at B not found"
    e = found[0].evidence
    assert e["model"] == "step"
    assert e["k"] == 6
    assert abs(abs(e["shift"]) - 12.0) < 2.5
    assert e["worst_after"] < e["worst_before"]


def test_no_false_twist_on_clean_data():
    for seed in (5, 6):
        net = build(seed=seed)
        assert orientation_changes(net, adjust(net)) == []


def test_refraction_is_told_apart_from_a_height_error():
    net = build(seed=7, refraction=(("D", "E"), 1.5))
    found = [f for f in vertical_check(net, SYN) if {f.evidence["a"], f.evidence["b"]} == {"D", "E"}]
    assert found, "refraction on D-E not found"
    e = found[0].evidence
    assert e["preferred"] == "refraction"
    assert abs(e["dk"] - 1.5) < 3 * e["dk_sd"] + 0.2


def _vertical(net):
    return vertical_check(net, SYN)


@pytest.mark.parametrize("seed", [20, 21, 22, 23, 24])
def test_wrong_instrument_height_is_named_at_its_setup(seed):
    """Seeds 20-49 all name it; seeds 9 and 16 come out can't-tell (a rival passes the split 1% level)."""
    net = build(seed=seed)
    for s in net.sightings:
        if s.setup == "B":
            s.hi += 0.030  # booked 30 mm too high
    found = [f for f in _vertical(net) if f.evidence["preferred"] != "either"]
    assert found, "instrument height at B not found"
    w = found[0].evidence["winner"]
    assert found[0].evidence["preferred"] == "instrument_height" and w["station"] == "B"
    assert abs(w["value"] - 0.030) < 3 * w["sd"] + 0.002
    assert "too high" in found[0].summary
    assert len(found) == 1, [f.title for f in found]


def test_wrong_target_height_is_named_at_the_right_end():
    net = build(seed=10)
    for s in net.sightings:
        if (s.frm, s.to) == ("D", "B"):
            s.ht += 0.050  # B's target height booked 50 mm too high at D (4 readings; 30 mm is not screened here)
    found = [f for f in _vertical(net) if f.evidence["preferred"] != "either"]
    assert found, "target height D->B not found"
    w = found[0].evidence["winner"]
    assert found[0].evidence["preferred"] == "target_height" and (w["frm"], w["to"]) == ("D", "B")
    assert abs(-w["value"] - 0.050) < 3 * w["sd"] + 0.002
    assert "too high" in found[0].summary


def test_no_vertical_finding_on_clean_data():
    for seed in (5, 6, 11):
        assert _vertical(build(seed=seed)) == []


def _swap_findings(net):
    return swapped_names(net, adjust(net))


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_swapped_point_numbers_are_named(seed):
    net = build(seed=seed)
    at_e = [i for i, s in enumerate(net.sightings) if s.setup == "E"]
    a, b = next((i, j) for i, j in zip(at_e, at_e[1:]) if {net.sightings[i].to, net.sightings[j].to} == {"A", "B"})
    net.sightings[a].to, net.sightings[b].to = net.sightings[b].to, net.sightings[a].to
    found = _swap_findings(net)
    assert len(found) == 1, [f.title for f in found]
    e = found[0].evidence
    assert e["pair"] and e["station"] == "E" and {e["booked"], e["proposed"]} == {"A", "B"}
    assert set(e["si"]) == {a, b}
    assert e["w_after"] < e["w_limit"] < e["w_before"]


def test_a_wrong_point_number_on_one_reading_is_named():
    net = build(seed=4)
    i = next(i for i, s in enumerate(net.sightings) if s.setup == "C" and s.to == "D")
    net.sightings[i].to = "F"
    found = _swap_findings(net)
    assert len(found) == 1 and not found[0].evidence["pair"]
    e = found[0].evidence
    assert (e["booked"], e["proposed"], e["si"]) == ("F", "D", [i])


def test_no_swap_finding_on_clean_data():
    for seed in (5, 6, 7, 8):
        assert _swap_findings(build(seed=seed)) == []


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_wrong_prism_constant_at_a_setup_is_named(seed):
    net = build(seed=seed)
    for s in net.sightings:
        if s.setup == "B" and s.slope:
            s.slope += 0.010  # instrument at B set 10 mm wrong
    found = distance_constants(net, SYN)
    assert len(found) == 1, [f.title for f in found]
    e = found[0].evidence
    assert e["preferred"] == "setup_constant" and e["winner"]["station"] == "B"
    assert abs(e["winner"]["value"] - 0.010) < 3 * e["winner"]["sd"] + 0.001
    assert "long" in found[0].title


def test_wrong_prism_on_a_mark_is_named():
    """F sits inside the network, so moving F cannot soak up a constant on the distances to it. (D, on the
    edge, often absorbs it: 27 of 30 random marks are named, the other 3 are not flagged.)"""
    net = build(seed=2)
    for s in net.sightings:
        if s.to == "F" and s.slope:
            s.slope -= 0.010  # the prism on F reads 10 mm short
    found = distance_constants(net, SYN)
    assert len(found) == 1, [f.title for f in found]
    e = found[0].evidence
    assert e["preferred"] == "target_constant" and e["winner"]["target"] == "F"
    assert abs(e["winner"]["value"] + 0.010) < 3 * e["winner"]["sd"] + 0.001
    assert "short" in found[0].title


def test_no_prism_finding_on_clean_data():
    for seed in (5, 6, 7, 8):
        assert distance_constants(build(seed=seed), SYN) == []


@pytest.mark.parametrize("comp,size", [("hz", 15.0), ("zen", 20.0)])
def test_a_single_bad_pointing_is_named(comp, size):
    net = build(seed=1)
    i = next(i for i, s in enumerate(net.sightings) if s.setup == "C" and s.to == "E")
    setattr(net.sightings[i], comp, getattr(net.sightings[i], comp) + size * SEC)
    found = bad_pointings(net, SYN)
    assert len(found) == 1 and found[0].evidence["si"] == [i], [f.title for f in found]
    r = found[0].evidence["readings"][0]
    assert r["component"] == comp and abs(r["diff_sec"] - size) < 6


@pytest.mark.parametrize("seed", [4, 5, 6])
def test_a_twisted_setup_is_not_read_as_bad_pointings(seed):
    """diagnose puts directions back on one orientation before comparing repeats."""
    from static_survey.diagnose import diagnose

    _, found = diagnose(build(seed=seed, twist=("B", 6, 12.0)), pointing=SYN)
    assert any(f.kind == "orientation_step" for f in found)
    assert not [f.title for f in found if f.kind == "pointing"]


def test_no_bad_pointing_on_clean_data():
    for seed in (5, 6, 7, 8):
        assert bad_pointings(build(seed=seed), SYN) == []


def test_every_mark_is_checked_in_a_braced_network():
    net = build(seed=8)
    assert unchecked_marks(net, adjust(net)) is None


def test_dms():
    assert abs(dms_to_deg("294.095900") - (294 + 9 / 60 + 59 / 3600)) < 1e-12
    assert abs(dms_to_deg("-7.0534") - -(7 + 5 / 60 + 34 / 3600)) < 1e-12


def test_weighted_control_corrections_average_zero():
    """With only weighted control, least squares corrections to the weighted marks average zero (weights 1/sd^2);
    a listing whose corrections do not is reported as not least squares, one that does is not."""
    from types import SimpleNamespace

    from static_survey.report import _weighted_paragraph, compare_weighted

    net = build(seed=3)
    off = {"A": (0.006, -0.004, 0.003), "C": (-0.002, 0.005, -0.004), "D": (0.001, 0.002, 0.005)}
    net.weighted = {n: (tuple(TRUE[n][i] + off[n][i] for i in range(3)), (0.02, 0.02, 0.03)) for n in off}
    net.fixed = {}
    net.points = {n: c for n, c in TRUE.items()}
    res = adjust(net)

    def listing(shift):
        adjusted = {n: dict(e=res.coords[n][0], n=res.coords[n][1], h=res.coords[n][2], sd_e=0.01, sd_n=0.01, sd_h=0.01)
                    for n in TRUE}
        weighted = {n: dict(ve=round(res.coords[n][0] - v[0] + shift, 4), vn=round(res.coords[n][1] - v[1], 4),
                            vh=round(res.coords[n][2] - v[2], 4)) for n, (v, _) in net.weighted.items()}
        return SimpleNamespace(adjusted=adjusted, weighted=weighted)

    same = compare_weighted(net, res, listing(0.0))
    assert max(abs(x) for x in same["mean_ours"]) < 1e-3                     # mm
    assert max(abs(x) for x in same["mean_compnet"]) <= 0.05
    assert "Within its 0.1 mm printing that is zero too" in _weighted_paragraph(same)[2]
    moved = compare_weighted(net, res, listing(0.001))
    assert abs(moved["mean_compnet"][0] - 1.0) < 0.06
    assert "not the least squares solution" in _weighted_paragraph(moved)[2]
