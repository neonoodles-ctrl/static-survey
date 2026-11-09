"""Planted-mistake scorecard on the synthetic network. Runs anywhere."""

import pytest

from static_survey import blunders

from synthetic import build

SYN = lambda s: (2.0, 3.0)  # noqa: E731  synthetic noise is all per reading

ALL = dict(twist=1, drift=1, refraction=1, target_height=1, inst_height=1, pointing=1, prism=1, swap=1)


def _snapshot(net):
    return [(s.to, s.hz, s.zen, s.slope, s.hi, s.ht) for s in net.sightings]


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_planted_twist_and_refraction_are_named_on_synthetic_data(seed):
    net = build(seed=seed)
    before = _snapshot(net)
    baseline, scores, _ = blunders.run(net, ALL, seed=seed, pointing=SYN)
    assert _snapshot(net) == before, "the job itself must not change"
    assert not [f for f in baseline if blunders.place(f)], "clean synthetic job should raise no located finding"
    by_kind = {s.case.kind: s for s in scores}
    assert set(by_kind) == set(ALL)
    assert by_kind["twist"].outcome == "named"
    assert by_kind["refraction"].outcome == "named"
    assert by_kind["swap"].outcome == "named"
    assert by_kind["prism"].outcome == "named"
    assert all(not s.elsewhere for s in scores), [s.elsewhere for s in scores]


def test_each_copy_changes_only_its_own_readings():
    net = build(seed=1)
    before = _snapshot(net)
    cases = blunders.make_cases(net, [], ALL, seed=1)
    for c in cases:
        after = _snapshot(c.net)
        changed = {i for i, (a, b) in enumerate(zip(before, after)) if a != b}
        assert changed and changed <= set(c.members), (c.kind, changed - set(c.members))
    assert _snapshot(net) == before


def test_scoring_tells_named_from_wrong_cause():
    """A wrong target height must come out as that target height at that setup; the other end, refraction
    or a can't-tell at the right leg is the right place but not named."""
    from static_survey.diagnose import Finding

    case = blunders.Case("target_height", "B from A", "1 mm", [0], {"S"}, {frozenset(("A", "B"))},
                         target=dict(frm="A", to="B"))

    def vertical(preferred, **winner):
        return Finding("refraction", "A ↔ B", "", dict(a="A", b="B", preferred=preferred, tied=[],
                                                        winner=dict(label="x", **winner)))

    class Res:
        rows = []

    right = vertical("target_height", key="target_height", frm="A", to="B")
    assert blunders.score(case, [], [right], Res).outcome == "named"
    assert blunders.score(case, [], [vertical("target_height", key="target_height", frm="B", to="A")], Res).outcome == "reading"
    assert blunders.score(case, [], [vertical("refraction", key="refraction", a="A", b="B")], Res).outcome == "reading"
    assert blunders.score(case, [], [vertical("either", key="target_height", frm="A", to="B")], Res).outcome == "reading"
    assert blunders.score(case, [], [], Res).outcome == "missed"
    assert blunders.score(case, [right], [right], Res).outcome == "missed", "a finding already on the clean job is not new"
