import math

import numpy as np

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


def test_held_and_free_control_components():
    """A weighted mark with standard deviations (0, 0, None) is held in E and N and left free in H (CompNet's FIX
    and FREE in .STD): E and N come out exactly as given, H from the observations, and the two held components
    add two degrees of freedom, as two removed unknowns would."""
    base = build(seed=3)
    r0 = adjust(base)
    given = (TRUE["D"][0] + 0.005, TRUE["D"][1] - 0.003, TRUE["D"][2] + 0.050)
    net = build(seed=3)
    net.weighted = {"D": (given, (0.0, 0.0, None))}
    res = adjust(net)
    assert abs(res.coords["D"][0] - given[0]) < 1e-5 and abs(res.coords["D"][1] - given[1]) < 1e-5
    assert abs(res.coords["D"][2] - TRUE["D"][2]) < 0.01            # free: not pulled 50 mm up
    assert res.dof == r0.dof + 2


def test_gnss_only_job_is_diagnosed():
    """A job with GNSS vectors and no total station readings adjusts and runs the cause-finding without the
    set-by-set checks, which are about total station repeats."""
    from synthetic import _geom
    from static_survey.adjust import Vector
    from static_survey.diagnose import diagnose

    geom = _geom(TRUE)
    pairs = [("A", "B"), ("A", "D"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A"), ("B", "F"), ("F", "D"),
             ("E", "F"), ("C", "F")]
    net = build(seed=5)
    net.sightings = []
    net.vectors = [Vector(a, b, tuple(geom[b][0] - geom[a][0]), 0.005, 0.010, ref=f"{a}-{b}") for a, b in pairs]
    res, findings = diagnose(net)
    assert res.converged and res.dof == 3 * len(pairs) - 3 * 4 - 2   # 4 free marks, GNSS scale and rotation
    for n in ("B", "D", "E", "F"):
        assert max(abs(res.coords[n][i] - TRUE[n][i]) for i in range(3)) < 1e-4
    assert not any(f.kind in ("repeats", "unchecked") for f in findings)


def _gnss_net(seed=5, plant=None):
    """GNSS-only copy of the synthetic network; plant=(pair index, metres) adds that much to a vector's height."""
    from synthetic import _geom
    from static_survey import geodesy as G
    from static_survey.adjust import Vector

    geom = _geom(TRUE)
    pairs = [("A", "B"), ("A", "D"), ("B", "C"), ("C", "D"), ("D", "E"), ("E", "A"), ("B", "F"), ("F", "D"),
             ("E", "F"), ("C", "F")]
    net = build(seed=seed)
    net.sightings = []
    net.vectors = []
    for i, (a, b) in enumerate(pairs):
        d = geom[b][0] - geom[a][0]
        if plant and plant[0] == i:
            d = d + plant[1] * geom[b][1][2]                     # along the local up at b
        net.vectors.append(Vector(a, b, tuple(d), 0.005, 0.010, ref=f"GPS line {i + 1}"))
    return net


def test_gnss_components_get_their_own_w_test():
    """Each GNSS component gets Baarda's w from its own residual covariance in E, N, H, so a height planted in one
    vector tops the table on that vector's dH, and the components' redundancies add up to the degrees of freedom."""
    from static_survey.diagnose import worst_residuals

    clean = adjust(_gnss_net())
    assert abs(sum(r["r"] for r in clean.rows if r["kind"] == "GNSS") - clean.dof) < 0.5
    res = adjust(_gnss_net(plant=(3, 0.06)))                       # 60 mm up on C -> D
    top = worst_residuals(res, 3, kinds=("DIR", "ZEN", "DIST", "GNSS"))[0]
    assert (top["frm"], top["to"], top["comp"]) == ("C", "D", "H") and top["w"] > 3.29
    assert worst_residuals(res, 3) == []                          # the default list is still total station only


def test_gnss_only_report_and_held_marks():
    """A GNSS-only report lists GNSS components among the largest residuals and does not mention refraction; a
    mark held in E and N shows as held, with no ellipse."""
    from static_survey.cad import coordinate_rows
    from static_survey.report import render

    net = _gnss_net()
    net.weighted = {"D": (TRUE["D"], (0.0, 0.0, None)), "F": (TRUE["F"], (None, None, 0.01))}
    res = adjust(net)
    rows = coordinate_rows(net, res)
    by = {r["name"]: r for r in rows}
    assert by["D"]["status"] == "held E/N" and by["D"]["a"] is None and by["F"]["status"] == "weighted H"
    md = render("SYN", net, res, [], [], coords=rows)
    assert "| GNSS dH |" in md or "| GNSS dE |" in md or "| GNSS dN |" in md
    assert "refraction" not in md
    assert "| D | held E/N |" in md and md.split("| D | held E/N |")[1].split("\n")[0].rstrip().endswith("| held |")


def test_compnet_implied_geoid_is_named_only_where_it_differs():
    """CompNet's OBS dH that imply a different N at one mark name that mark with the size; a listing that agrees
    names nothing."""
    from types import SimpleNamespace

    from static_survey import geodesy as G
    from static_survey.report import compare_geoid

    net = _gnss_net()
    pos = {n: TRUE[n] for n in TRUE}

    def listing(bump):
        vecs = []
        for v in net.vectors:
            e, n, h = pos[v.frm]
            lat, lon = G.grid_to_latlon(e, n, net.epsg)
            xa = np.array(G.geodetic_to_ecef(lat, lon, h + net.geoid[v.frm]), float).ravel()
            _, _, hb = G.ecef_to_geodetic(xa + np.array(v.dxyz))
            dh = float(hb[0]) - (h + net.geoid[v.frm]) - (net.geoid[v.to] - net.geoid[v.frm])
            dh -= bump.get(v.to, 0.0) - bump.get(v.frm, 0.0)        # CompNet's N at a mark is higher by bump
            vecs.append(dict(frm=v.frm, to=v.to, obs_enh=(0.0, 0.0, round(dh, 3))))
        adjusted = {n: dict(e=c[0], n=c[1], h=c[2]) for n, c in pos.items() if n not in net.fixed}
        return SimpleNamespace(vectors=vecs, adjusted=adjusted, fixed={n: pos[n] for n in net.fixed})

    assert compare_geoid(net, listing({})) == []
    found = compare_geoid(net, listing({"E": 0.013}))
    assert [g["mark"] for g in found] == ["E"] and abs((found[0]["n_compnet"] - found[0]["n_ours"]) - 0.013) < 0.0008
