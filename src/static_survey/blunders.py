"""Put known field mistakes into copies of a job and score what the diagnosis names.

Each copy gets exactly one mistake, sized at 3-5 times the standard error of
the readings it touches (a swapped point number is sized by the geometry).
Places the clean job already flags are left alone, so a real finding is not
mistaken for an injected one.

Scores per copy:

* ``named``   a new finding of the right kind at the right place
* ``reading`` the changed readings show in the report's largest-residuals
              table, or a new finding at the right place gives another cause
* ``missed``  neither

New findings anywhere else are counted as false alarms.

This works on copies only. The job itself is never changed.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from .adjust import SEC, Network
from .diagnose import Finding, _minutes, diagnose, worst_residuals

# What a correct diagnosis says for each mistake: (finding kind, preferred explanation).
EXPECT: dict[str, tuple[str, str] | None] = {
    "twist": ("orientation_step", ""),
    "drift": ("orientation_drift", ""),
    "refraction": ("refraction", "refraction"),
    "target_height": ("refraction", "target_height"),
    "inst_height": ("refraction", "instrument_height"),
    "prism": ("prism", "setup_constant"),
    "swap": ("swap", ""),
    "pointing": ("pointing", ""),
}
TITLES = {
    "twist": "Setup twist (orientation step)", "drift": "Orientation drift", "refraction": "Refraction on one leg",
    "target_height": "Wrong target height", "inst_height": "Wrong instrument height",
    "prism": "Wrong prism constant", "swap": "Swapped point numbers", "pointing": "One bad pointing",
}
TABLE_ROWS = 12   # the report's largest-residuals table


@dataclass
class Case:
    kind: str
    where: str
    size: str
    members: list[int]
    setups: set[str] = field(default_factory=set)
    legs: set[frozenset] = field(default_factory=set)
    net: Network | None = None
    target: dict = field(default_factory=dict)   # the exact end a height or refraction finding must name


@dataclass
class Score:
    case: Case
    outcome: str          # named | reading | missed
    said: str             # what the diagnosis said at the place
    elsewhere: list[str]  # new findings at places nothing was changed


def place(f: Finding):
    """Where a finding points: ('setup', name) or ('leg', {a, b}); None for job-wide findings."""
    e = f.evidence
    if f.kind.startswith("orientation"):
        return ("setup", e["setup"])
    if f.kind in ("swap", "pointing"):
        return ("setup", e["setup"])
    if f.kind == "prism":
        return ("setup", e["setup"]) if e["place"] == "setup" else ("target", e["target"])
    if f.kind == "refraction":
        if e.get("b") is None:
            return ("setup", e["setup"])
        return ("leg", frozenset((e["a"], e["b"])))
    return None


def _copy(net: Network, members: list[int]) -> Network:
    n2 = copy.copy(net)
    n2.sightings = list(net.sightings)
    for si in members:
        n2.sightings[si] = copy.copy(net.sightings[si])
    return n2


def _index(net: Network):
    by_setup: dict[str, list[int]] = defaultdict(list)
    by_leg: dict[tuple[str, str], list[int]] = defaultdict(list)
    for si, s in enumerate(net.sightings):
        by_setup[s.setup].append(si)
        if s.zen is not None:
            by_leg[(s.frm, s.to)].append(si)
    legs = sorted({tuple(sorted(k)) for k in by_leg if (k[1], k[0]) in by_leg
                   and len(by_leg[k]) >= 3 and len(by_leg[(k[1], k[0])]) >= 3})
    return by_setup, by_leg, legs


def _zen_height(net: Network, idx: list[int]) -> float:
    """Height change (m) that moves these zenith readings by one standard error, median over the set."""
    vals = [s.sd_zen * s.slope / max(math.sin(s.zen), 1e-6)
            for s in (net.sightings[i] for i in idx) if s.zen is not None and s.slope]
    return float(np.median(vals))


def make_cases(net: Network, findings: list[Finding], plan: dict[str, int], seed: int = 1,
               sizes: tuple[float, ...] = (3.0, 4.0, 5.0)) -> list[Case]:
    """One copy of ``net`` per mistake. ``plan`` maps a mistake kind to how many copies."""
    rng = np.random.default_rng(seed)
    by_setup, by_leg, legs = _index(net)
    flagged = {place(f) for f in findings} - {None}
    busy_setups = {p[1] for p in flagged if p[0] == "setup"}
    busy_legs = {p[1] for p in flagged if p[0] == "leg"}
    station = {st: net.sightings[idx[0]].frm for st, idx in by_setup.items()}
    free_legs = [l for l in legs if frozenset(l) not in busy_legs]
    used: set = set()
    cases: list[Case] = []

    def pick(options):
        options = [o for o in options if o not in used]
        if not options:
            raise ValueError("not enough places left in this job for the requested mistakes")
        o = options[rng.integers(len(options))]
        used.add(o)
        return o

    def dirs(st):
        return [i for i in by_setup[st] if net.sightings[i].hz is not None]

    def legs_of(st):
        """Reciprocal legs observed from this setup."""
        out = set()
        for i in by_setup[st]:
            s = net.sightings[i]
            if s.zen is not None and tuple(sorted((s.frm, s.to))) in free_legs:
                out.add(frozenset((s.frm, s.to)))
        return out

    free_setups = sorted(st for st in by_setup if st not in busy_setups)
    for kind, count in plan.items():
        for c in range(count):
            k = sizes[c % len(sizes)]
            if kind in ("twist", "drift", "pointing", "prism", "swap", "inst_height"):
                if kind == "drift":
                    opts = []
                    for st in free_setups:
                        t = [_minutes(net.sightings[i].time) for i in dirs(st)]
                        if len(t) >= 8 and None not in t and max(t) - min(t) >= 10:
                            opts.append(st)
                elif kind == "prism":
                    opts = [st for st in free_setups
                            if sum(1 for i in by_setup[st] if net.sightings[i].slope) >= 4]
                elif kind == "inst_height":
                    opts = [st for st in free_setups if legs_of(st)]
                elif kind == "swap":
                    opts = [st for st in free_setups if len({net.sightings[i].to for i in dirs(st)}) >= 2]
                else:
                    opts = [st for st in free_setups if len(dirs(st)) >= 8]
                st = pick(opts)
                stn = station[st]
                if kind == "twist":
                    m = dirs(st)
                    cut = len(m) // 2
                    shift = k * float(np.median([net.sightings[i].sd_dir for i in m]))
                    moved = m[cut:]
                    n2 = _copy(net, moved)
                    for i in moved:
                        n2.sightings[i].hz += shift
                    cases.append(Case(kind, f"{stn}, from reading {cut + 1} of {len(m)}",
                                      f"{shift / SEC:.1f}\" ({k:g}σ)", moved, {st}, set(), n2))
                elif kind == "drift":
                    m = dirs(st)
                    t = np.array([_minutes(net.sightings[i].time) for i in m], float)
                    total = k * float(np.median([net.sightings[i].sd_dir for i in m]))
                    n2 = _copy(net, m)
                    for i, ti in zip(m, t):
                        n2.sightings[i].hz += total * (ti - t.min()) / (t.max() - t.min())
                    cases.append(Case(kind, f"{stn}, {len(m)} readings over {t.max() - t.min():.0f} min",
                                      f"{total / SEC:.1f}\" in total ({k:g}σ)", m, {st}, set(), n2))
                elif kind == "pointing":
                    i = dirs(st)[rng.integers(len(dirs(st)))]
                    s = net.sightings[i]
                    n2 = _copy(net, [i])
                    n2.sightings[i].hz += k * s.sd_dir
                    cases.append(Case(kind, f"{stn} → {s.to}, {s.ref}", f"{k * s.sd_dir / SEC:.1f}\" ({k:g}σ)",
                                      [i], {st}, set(), n2, target=dict(si=i)))
                elif kind == "prism":
                    m = [i for i in by_setup[st] if net.sightings[i].slope]
                    d = k * float(np.median([net.sightings[i].sd_dist for i in m]))
                    n2 = _copy(net, m)
                    for i in m:
                        n2.sightings[i].slope += d
                    cases.append(Case(kind, f"{stn}, all {len(m)} distances", f"{d * 1000:+.1f} mm ({k:g}σ)",
                                      m, {st}, {frozenset((stn, net.sightings[i].to)) for i in m}, n2,
                                      target=dict(station=stn)))
                elif kind == "swap":
                    m = dirs(st)
                    pairs = [(a, b) for a, b in zip(m, m[1:]) if net.sightings[a].to != net.sightings[b].to]
                    a, b = pairs[rng.integers(len(pairs))]
                    sa, sb = net.sightings[a], net.sightings[b]
                    ang = abs((sa.hz - sb.hz + math.pi) % (2 * math.pi) - math.pi)
                    n2 = _copy(net, [a, b])
                    n2.sightings[a].to, n2.sightings[b].to = sb.to, sa.to
                    cases.append(Case(kind, f"{stn}: {sa.to} and {sb.to} swapped at {sa.ref}",
                                      f"{math.degrees(ang):.1f}° apart", [a, b], {st},
                                      {frozenset((stn, sa.to)), frozenset((stn, sb.to))}, n2,
                                      target=dict(names={sa.to, sb.to})))
                else:  # inst_height
                    m = by_setup[st]
                    d = k * _zen_height(net, m)
                    n2 = _copy(net, m)
                    for i in m:
                        n2.sightings[i].hi += d
                    cases.append(Case(kind, f"{stn}, whole setup", f"{d * 1000:+.1f} mm ({k:g}σ on zeniths)",
                                      m, {st}, legs_of(st), n2, target=dict(station=stn)))
            else:  # target_height, refraction: one reciprocal leg
                a, b = pick(free_legs)
                if kind == "target_height":
                    if rng.integers(2):
                        a, b = b, a
                    m = by_leg[(a, b)]
                    d = k * _zen_height(net, m)
                    n2 = _copy(net, m)
                    for i in m:
                        n2.sightings[i].ht += d
                    # A wrong target height also moves the target mark, so its own setup is part of the place.
                    at_b = {st for st, idx in by_setup.items() if station[st] == b}
                    cases.append(Case(kind, f"{b} as seen from {a}, {len(m)} readings",
                                      f"{d * 1000:+.1f} mm ({k:g}σ on zeniths)", m,
                                      {net.sightings[i].setup for i in m} | at_b, {frozenset((a, b))}, n2,
                                      target=dict(frm=a, to=b)))
                else:
                    m = by_leg[(a, b)] + by_leg[(b, a)]
                    dz = k * float(np.median([net.sightings[i].sd_zen for i in m]))
                    n2 = _copy(net, m)
                    for i in m:
                        n2.sightings[i].zen -= dz
                    cases.append(Case(kind, f"{a} ↔ {b}, {len(m)} zeniths", f"{dz / SEC:.1f}\" both ends ({k:g}σ)",
                                      m, {net.sightings[i].setup for i in m}, {frozenset((a, b))}, n2,
                                      target=dict(a=a, b=b)))
    return cases


def _at_case(f: Finding, case: Case) -> bool:
    p = place(f)
    return p is not None and ((p[0] == "setup" and p[1] in case.setups) or (p[0] == "leg" and p[1] in case.legs)
                              or (p[0] == "target" and any(p[1] in leg for leg in case.legs)))


def _says(f: Finding) -> str:
    if f.kind == "refraction":
        e, w = f.evidence, f.evidence["winner"]
        if e["preferred"] == "either":
            return "can't tell: " + " / ".join([w["label"]] + e["tied"] + e.get("also_present", []))
        return w["label"]
    if f.kind == "swap":
        return f.title.split(": ", 1)[1]
    if f.kind == "pointing":
        r = max(f.evidence["readings"], key=lambda x: x["z"])
        return f"bad pointing at {r['ref']} ({r['z']:.1f}× pointing error)"
    if f.kind == "prism":
        e, w = f.evidence, f.evidence["winner"]
        if e["preferred"] == "either":
            return "can't tell: " + " / ".join([w["label"]] + e["tied"] + e.get("also_present", []))
        return f"{w['label']}, {w['value'] * 1000:+.1f} mm"
    return {"orientation_step": "orientation step", "orientation_drift": "orientation drift"}.get(f.kind, f.kind)


def _right_end(f: Finding, case: Case) -> bool:
    """A height or refraction finding must name the end that was changed, not just the leg."""
    if not case.target:
        return True
    if f.kind == "swap":
        return "names" not in case.target or {f.evidence["booked"], f.evidence["proposed"]} == case.target["names"]
    if f.kind == "prism":
        return f.evidence["winner"].get("station") == case.target.get("station")
    if f.kind == "pointing":
        return "si" not in case.target or case.target["si"] in f.evidence["si"]
    if f.kind != "refraction":
        return True
    w, t = f.evidence["winner"], case.target
    if "station" in t:
        return w.get("station") == t["station"]
    if "frm" in t:
        return (w.get("frm"), w.get("to")) == (t["frm"], t["to"])
    return {w.get("a"), w.get("b")} == {t["a"], t["b"]}


def score(case: Case, baseline: list[Finding], findings: list[Finding], res) -> Score:
    old = {place(f) for f in baseline} - {None}
    new = [f for f in findings if place(f) is not None and place(f) not in old]
    here = [f for f in new if _at_case(f, case)]
    elsewhere = [f.title for f in new if not _at_case(f, case)]
    want = EXPECT[case.kind]
    if want and any(f.kind == want[0] and (not want[1] or f.evidence.get("preferred") == want[1])
                    and _right_end(f, case) for f in here):
        return Score(case, "named", "; ".join(_says(f) for f in here), elsewhere)
    in_table = [r for r in worst_residuals(res, TABLE_ROWS) if r.get("si") in set(case.members)]
    if here or in_table:
        said = "; ".join(_says(f) for f in here)
        if in_table:
            best = min(range(len(in_table)), key=lambda j: -in_table[j]["w"])
            r = in_table[best]
            said = (said + "; " if said else "") + f"{len(in_table)} changed reading(s) in the residuals table, top w {r['w']:.1f}"
        return Score(case, "reading", said, elsewhere)
    return Score(case, "missed", "nothing at this place", elsewhere)


def run(net: Network, plan: dict[str, int], seed: int = 1, pointing=None) -> tuple[list[Finding], list[Score], float]:
    """Clean findings, one score per copy, and the smallest w in the clean job's residuals table.
    ``pointing`` is passed to ``diagnose``."""
    res0, baseline = diagnose(net, pointing=pointing)
    table = worst_residuals(res0, TABLE_ROWS)
    floor = table[-1]["w"] if table else float("nan")
    scores = []
    for case in make_cases(net, baseline, plan, seed):
        res, found = diagnose(case.net, pointing=pointing)
        scores.append(score(case, baseline, found, res))
    return baseline, scores, floor


def render(job: str, baseline: list[Finding], scores: list[Score], floor: float | None = None,
           notes: list[str] | None = None) -> str:
    L = [f"# {job}: does the cause-finding name planted mistakes?", ""]
    n = len(scores)
    tally = {o: sum(1 for s in scores if s.outcome == o) for o in ("named", "reading", "missed")}
    alarms = sum(len(s.elsewhere) for s in scores)
    L += [f"{n} copies of the job, one planted mistake each. "
          f"**{tally['named']} named correctly, {tally['reading']} found at the right reading but not named, "
          f"{tally['missed']} missed.** {alarms} new finding(s) appeared at places nothing was changed.", ""]
    L += ["| Mistake | Copies | Named | Right reading, not named | Missed |", "|---|---|---|---|---|"]
    for kind in dict.fromkeys(s.case.kind for s in scores):
        ss = [s for s in scores if s.case.kind == kind]
        L.append(f"| {TITLES[kind]} | {len(ss)} | " + " | ".join(
            str(sum(1 for s in ss if s.outcome == o)) for o in ("named", "reading", "missed")) + " |")
    L += ["", "## Each copy", "", "| # | Mistake | Where | Size | Result | What the diagnosis said | Elsewhere |",
          "|---|---|---|---|---|---|---|"]
    for i, s in enumerate(scores, 1):
        L.append(f"| {i} | {TITLES[s.case.kind]} | {s.case.where} | {s.case.size} | {s.outcome} | {s.said} | "
                 f"{'; '.join(s.elsewhere) or '—'} |")
    never = [k for k in dict.fromkeys(s.case.kind for s in scores)
             if not any(s.outcome == "named" for s in scores if s.case.kind == k)]
    if never:
        L += ["", "## Not named in any copy", ""]
        for k in never:
            ss = [s for s in scores if s.case.kind == k]
            how = ("no cause check exists for it yet" if EXPECT[k] is None else
                   f"the {EXPECT[k][0].replace('_', ' ')} check exists but did not name it")
            L.append(f"- **{TITLES[k]}** ({len(ss)} copies, {sum(1 for s in ss if s.outcome == 'missed')} missed): {how}.")
    L += ["", "## Clean job", "", "Findings on the unchanged job. Their places were left out of the planting.", ""]
    for f in baseline:
        if place(f) is not None:
            L.append(f"- {f.title}")
    if floor is not None and math.isfinite(floor):
        L += ["", f"The clean job's largest-residuals table ends at w {floor:.2f}. A planted reading must beat that "
              "to show in the table."]
    L += ["", "## How it was scored", "",
          "- **Named:** a new finding of the right kind at the place of the mistake. A setup twist must come out as "
          "an orientation step, a drift as a drift, refraction as refraction on that leg, a wrong target height as that "
          "target height as booked at that setup, a wrong instrument height as that setup's instrument height.",
          f"- **Right reading, not named:** a changed reading is in the report's largest-residuals table "
          f"(top {TABLE_ROWS} by w), or a new finding at that place gives another cause.",
          "- **Missed:** neither.",
          "- **Size:** 3, 4 or 5 times the standard error of the readings touched, in turn. A height error is sized "
          "by the zenith change it causes. A swapped point number is sized by the angle between the two marks.",
          "- A swapped point number must come out as those two names at that setup. A wrong prism constant must "
          "come out as a constant on every distance from that setup.",
          "- A single bad pointing must come out as that reading."]
    if notes:
        L += ["", "## Notes", ""] + [f"- {x}" for x in notes]
    return "\n".join(L) + "\n"
