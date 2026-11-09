"""Cause-finding on top of a least squares adjustment.

Data snooping says *which* observation does not fit. These checks ask *why*,
by testing a specific cause as an extra model parameter and keeping it only
when the improvement is statistically significant:

* ``global_test``        two-sided chi-square test of the variance factor
* ``orientation_breaks`` a setup whose circle orientation jumped part-way
                         through (tripod twist, bumped instrument)
* ``vertical_patterns``  zenith misfits on a reciprocal leg or a whole setup:
                         refraction, a wrong target height or a wrong
                         instrument height, each fitted and tested against
                         the others
* ``distance_constants`` distances from one setup or to one mark all off by
                         the same amount (a prism constant set wrong, or
                         the wrong prism)
* ``bad_pointings``      a reading that disagrees with the other repeats of
                         its own set (a single bad pointing)
* ``swapped_names``      a reading that fits another mark seen from the same
                         setup much better than the one it was booked to
                         (a wrong or swapped point number)
* ``repeat_effect``      what the statistics look like if each set of
                         repeated readings counts as one measurement
* ``unchecked_marks``    marks no other observation can check

Every finding carries the numbers that support it. Nothing here changes the
surveyor's data; each finding is a proposal.
"""

from __future__ import annotations

import copy
import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from . import geodesy as G
from .adjust import SEC, Extra, Network, Result, Sighting, adjust


@dataclass
class Finding:
    kind: str
    title: str
    summary: str
    evidence: dict = field(default_factory=dict)
    severity: str = "info"   # info | check | likely-cause


def cost(res: Result) -> float:
    return float(sum(r["vw"] ** 2 for r in res.rows))


# ------------------------------------------------------------------ global

def global_test(res: Result, alpha: float = 0.05) -> Finding:
    lo = stats.chi2.ppf(alpha / 2, res.dof) / res.dof
    hi = stats.chi2.ppf(1 - alpha / 2, res.dof) / res.dof
    upper_only = res.vf <= stats.chi2.ppf(1 - alpha, res.dof) / res.dof
    two_sided = lo <= res.vf <= hi
    if two_sided:
        verdict = "passes both sides"
    elif res.vf < lo:
        verdict = "passes the upper side only; it is below the lower limit"
    else:
        verdict = "fails the upper limit"
    return Finding(
        kind="global_test",
        title="Global test (variance factor)",
        summary=(f"Variance factor {res.vf:.3f} with {res.dof} degrees of freedom; "
                 f"95% two-sided band {lo:.3f}–{hi:.3f}. The adjustment {verdict}."),
        evidence=dict(vf=res.vf, dof=res.dof, lower=lo, upper=hi, upper_only_pass=bool(upper_only),
                      two_sided_pass=bool(two_sided), groups=res.group_vf),
        severity="info" if two_sided else "check",
    )


# --------------------------------------------------------- orientation

def _minutes(t: str) -> float | None:
    """'MM-DD hh:mm' -> minutes since the start of the month (enough within a setup)."""
    if not t:
        return None
    try:
        md, hm = t.split()
        _, dd = md.split("-")
        hh, mm = hm.split(":")
        return int(dd) * 1440 + int(hh) * 60 + int(mm)
    except ValueError:
        return None


def _split_net(net: Network, setup: str, members: list[int]) -> Network:
    n2 = copy.copy(net)
    n2.sightings = list(net.sightings)
    for si in members:
        s = copy.copy(n2.sightings[si])
        s.setup = setup + "/b"
        n2.sightings[si] = s
    return n2


def orientation_changes(net: Network, res: Result, alpha: float = 0.01, min_side: int = 2) -> list[Finding]:
    """Setups whose circle orientation moved during the setup: a step (bumped or
    settled instrument) or a steady drift (tripod twisting as it warms)."""
    by_setup: dict[str, list[dict]] = defaultdict(list)
    for r in res.rows:
        if r["kind"] == "DIR":
            by_setup[r["setup"]].append(r)
    cands = []
    n_tests = 0
    for setup, rows in by_setup.items():
        rows = sorted(rows, key=lambda r: r["si"])
        v = np.array([r["vw"] for r in rows])
        n = len(v)
        if n < 2 * min_side:
            continue
        best_step = (0.0, None)
        for k in range(min_side, n - min_side + 1):
            n_tests += 1
            d2 = (v[:k].mean() - v[k:].mean()) ** 2 / (1 / k + 1 / (n - k))
            if d2 > best_step[0]:
                best_step = (d2, k)
        drift = (0.0, None)
        t = [_minutes(net.sightings[r["si"]].time) for r in rows]
        if all(x is not None for x in t) and max(t) - min(t) >= 5:
            n_tests += 1
            tc = np.array(t, float) - np.mean(t)
            drift = (float((v @ tc) ** 2 / (tc @ tc)), np.array(t, float) - min(t))
        cands.append((max(best_step[0], drift[0]), setup, rows, best_step, drift))
    crit = stats.chi2.isf(alpha / max(n_tests, 1), 1)
    base = cost(res)
    out = []
    for score, setup, rows, step, drift in sorted(cands, key=lambda c: -c[0]):
        if score < crit:
            break
        station = rows[0]["frm"]
        worst_before = max(r["v_over_sigma"] for r in rows)
        trials = []
        if step[1] is not None:
            k = step[1]
            first, later = [r["si"] for r in rows[:k]], [r["si"] for r in rows[k:]]
            moved = first if len(first) <= len(later) else later
            r2 = adjust(_split_net(net, setup, moved))
            sh = (r2.orientations[setup + "/b"] - r2.orientations[setup]) / SEC
            trials.append(("step", base - cost(r2), r2, dict(k=k, shift=sh if moved is later else -sh)))
        if drift[1] is not None:
            mins = list(drift[1])
            r2 = adjust(net, [Extra("orient_drift", [r["si"] for r in rows], f"drift {station}", values=mins)])
            rate = r2.extras[0]["value"] / SEC
            trials.append(("drift", base - cost(r2), r2, dict(rate=rate, span=max(mins),
                                                               rate_sd=r2.extras[0]["sd"] / SEC)))
        kind, gain, r2, info = max(trials, key=lambda x: x[1])
        alt = [t for t in trials if t[0] != kind]
        alt_txt = ""
        if alt:
            other = "A steady drift" if alt[0][0] == "drift" else "A single step"
            alt_txt = f" {other} fits less well (χ² gain {alt[0][1]:.1f} against {gain:.1f})."
        f_stat = gain / (cost(r2) / r2.dof)
        p = stats.f.sf(f_stat, 1, r2.dof) * max(n_tests, 1)
        if p > alpha:
            continue
        after = {r["si"]: r for r in r2.rows if r["kind"] == "DIR"}
        worst_after = max(after[r["si"]]["v_over_sigma"] for r in rows)
        t0, t1 = net.sightings[rows[0]["si"]].time, net.sightings[rows[-1]["si"]].time
        if kind == "step":
            k = info["k"]
            tb, ta = net.sightings[rows[k - 1]["si"]].time, net.sightings[rows[k]["si"]].time
            gap = ""
            if tb and ta:
                mb, ma = _minutes(tb), _minutes(ta)
                gap = (f" The instrument log jumps from {tb} to {ta} ({ma - mb} minutes) between the two groups."
                       if ma is not None and mb is not None and ma - mb >= 5 else
                       f" There is no pause in the instrument log at that point ({tb}), so the instrument "
                       f"probably settled or was nudged while observing.")
            title = f"Setup at {station}: orientation jumped part-way through"
            summary = (f"The first {k} of {len(rows)} direction readings at {station} sit about "
                       f"{abs(info['shift']):.1f}\" away from the rest, all in the same direction.{gap} "
                       f"Giving the setup a second orientation from reading {k + 1} on takes the largest |v/σ| "
                       f"on this setup from {worst_before:.2f} to {worst_after:.2f} without deleting any reading.{alt_txt}")
        else:
            title = f"Setup at {station}: orientation drifted during the setup"
            summary = (f"The {len(rows)} direction readings at {station} ({t0} to {t1}) turn steadily by about "
                       f"{info['rate']:+.2f}\" ± {info['rate_sd']:.2f} per minute, "
                       f"{abs(info['rate'] * info['span']):.1f}\" over {info['span']:.0f} minutes, "
                       f"the pattern of a tripod twisting as it warms or settles. Modelling the drift takes the "
                       f"largest |v/σ| on this setup from {worst_before:.2f} to {worst_after:.2f}.{alt_txt}")
        out.append(Finding(
            kind="orientation_" + kind, title=title, summary=summary,
            evidence=dict(setup=setup, station=station, n=len(rows), model=kind, chi2_gain=gain,
                          chi2_gain_alternative=[t[1] for t in trials if t[0] != kind], f=f_stat,
                          p_bonferroni=p, worst_before=worst_before, worst_after=worst_after, **info,
                          readings=[dict(to=r["to"], time=net.sightings[r["si"]].time, res_sec=round(r["res_sec"], 2),
                                         ref=r["ref"], si=r["si"]) for r in rows]),
            severity="likely-cause",
        ))
    return out


orientation_breaks = orientation_changes


# ------------------------------------------------------- vertical angles

@dataclass
class _Expl:
    key: str      # refraction | target_height | instrument_height
    label: str
    extra: Extra
    info: dict


def _vertical_explanations_leg(net: Network, a: str, b: str, ab: list[int], ba: list[int], members_of) -> list[_Expl]:
    setups_a = sorted({net.sightings[i].setup for i in ab})
    setups_b = sorted({net.sightings[i].setup for i in ba})
    return [
        _Expl("refraction", f"refraction on {a} ↔ {b}", Extra("refraction", ab + ba, f"{a}<->{b}"), dict(a=a, b=b)),
        _Expl("target_height", f"target height of {b} as booked at {a}", Extra("ht_offset", ab, f"HT {a}->{b}"),
              dict(frm=a, to=b)),
        _Expl("target_height", f"target height of {a} as booked at {b}", Extra("ht_offset", ba, f"HT {b}->{a}"),
              dict(frm=b, to=a)),
        _Expl("instrument_height", f"instrument height at {a}", Extra("ht_offset", members_of(setups_a), f"HI {a}"),
              dict(station=a, setups=setups_a)),
        _Expl("instrument_height", f"instrument height at {b}", Extra("ht_offset", members_of(setups_b), f"HI {b}"),
              dict(station=b, setups=setups_b)),
    ]


def _effective(net: Network, members: list[int]) -> frozenset:
    """The vertical readings an explanation can move (an instrument height moves every zenith of its setup)."""
    return frozenset(i for i in members if net.sightings[i].zen is not None)


def _moves(net: Network, extra: Extra) -> tuple[str, frozenset]:
    """The readings an extra parameter can move: distances for a distance constant, zeniths otherwise."""
    if extra.kind in ("dist_const", "dist_scale"):
        return ("dist", frozenset(i for i in extra.members if net.sightings[i].slope is not None))
    return ("zen", _effective(net, extra.members))


def _compare(net: Network, accepted: list[Extra], base_cost: float, expls: list[_Expl], alpha: float):
    """Fit each explanation alone (on top of those already accepted), then test the best against each
    rival with both in the model.

    The best wins over a rival when it still improves the fit significantly with the rival present (F-test
    at ``alpha``; the best must do this against every rival, which needs no correction) and the rival does
    not improve it significantly with the best present (F-test at ``alpha`` split over the rivals, since any
    one of them doing so counts). Explanations that move exactly the same readings cannot be told apart and
    are tied without a fit."""
    fits = []
    taken = {_moves(net, x) for x in accepted}
    for e in expls:
        if _moves(net, e.extra) in taken:
            continue  # the same readings are already explained by an accepted cause
        try:
            r = adjust(net, accepted + [e.extra])
        except np.linalg.LinAlgError:
            continue  # cannot be told apart from the causes already accepted
        fits.append((e, r, base_cost - cost(r)))
    if not fits:
        return None
    fits.sort(key=lambda t: -t[2])
    best, best_res, best_gain = fits[0]
    beaten, tied, both = [], [], []
    alpha_rival = alpha / max(len(fits) - 1, 1)
    for e, r, g in fits[1:]:
        if _moves(net, e.extra) == _moves(net, best.extra):
            tied.append((e, g))
            continue
        try:
            joint = adjust(net, accepted + [best.extra, e.extra])
        except np.linalg.LinAlgError:
            tied.append((e, g))
            continue
        cj = cost(joint)
        s2 = cj / joint.dof
        p_best = stats.f.sf(max(cost(r) - cj, 0.0) / s2, 1, joint.dof)         # best on top of the rival
        p_rival = stats.f.sf(max(cost(best_res) - cj, 0.0) / s2, 1, joint.dof)  # rival on top of the best
        if p_best <= alpha and p_rival > alpha_rival:
            beaten.append((e, g))
        elif p_best <= alpha:
            both.append((e, g))
        else:
            tied.append((e, g))
    return (best, best_res, best_gain), beaten, tied, both, fits


def _cap(text: str) -> str:
    return text[:1].upper() + text[1:]


def _join(items) -> str:
    items = list(items)
    return items[0] if len(items) == 1 else ", ".join(items[:-1]) + " and " + items[-1]


def _height_text(e: _Expl, value: float, sd: float) -> str:
    # The fitted offset raises the target in the model. A booked target height that is too high
    # needs a negative offset; a booked instrument height that is too high needs a positive one.
    err = -value if e.key == "target_height" else value
    word = "high" if err > 0 else "low"
    return f"{abs(err) * 1000:.1f} ± {sd * 1000:.1f} mm too {word}"


def vertical_patterns(net: Network, res: Result, alpha: float = 0.01, min_n: int = 3,
                      counts: list[int] | None = None) -> list[Finding]:
    """Vertical misfits on reciprocal legs and whole setups: refraction, a wrong target height or a wrong
    instrument height. Each explanation is fitted as an extra parameter and the winner must beat every
    rival with both in the model; when none does, the finding says it can't tell and what would.

    Places are taken worst first. Each place is tested with every other flagged place's best explanation
    in the model, so one mistake is not counted as evidence next door or reported twice."""
    setup_members: dict[str, list[int]] = defaultdict(list)
    for si, s in enumerate(net.sightings):
        setup_members[s.setup].append(si)

    def members_of(setups):
        return [i for st in setups for i in setup_members[st]]

    def n_read(rows) -> int:
        """Readings behind these rows (a set mean stands for all the readings of its set)."""
        return sum(counts[x["si"]] for x in rows) if counts else len(rows)

    def group(rows):
        by_leg: dict[tuple[str, str], list[dict]] = defaultdict(list)
        by_setup: dict[str, list[dict]] = defaultdict(list)
        for r in rows:
            if r["kind"] == "ZEN":
                by_leg[(r["frm"], r["to"])].append(r)
                by_setup[r["setup"]].append(r)
        return by_leg, by_setup

    by_leg, by_setup_rows = group(res.rows)
    pairs, done = [], set()
    for (a, b), rows in by_leg.items():
        if (b, a) in by_leg and frozenset((a, b)) not in done and len(rows) >= min_n and len(by_leg[(b, a)]) >= min_n:
            done.add(frozenset((a, b)))
            pairs.append((a, b))
    setups = [st for st, rows in by_setup_rows.items() if len(rows) >= min_n]
    n_tests = max(len(pairs) + len(setups), 1)
    crit = stats.chi2.isf(alpha / n_tests, 1)

    def screen(what, where, by_leg, by_setup_rows) -> float:
        """A leg: the part of its zenith misfit shared by both ends. A setup: its mean zenith misfit."""
        if what == "leg":
            f, r_ = by_leg[where], by_leg[(where[1], where[0])]
            common = (np.mean([x["vw"] for x in f]) + np.mean([x["vw"] for x in r_])) / 2
            return float(common ** 2 / (1 / len(f) + 1 / len(r_)))
        v = np.array([x["vw"] for x in by_setup_rows[where]])
        return float(v.sum() ** 2 / len(v))

    cands = [(screen("leg", ab, by_leg, by_setup_rows), "leg", ab) for ab in pairs]
    cands += [(screen("setup", st, by_leg, by_setup_rows), "setup", st) for st in setups]
    cands = sorted([c for c in cands if c[0] >= crit], key=lambda c: -c[0])

    def explanations(what, where):
        if what == "leg":
            a, b = where
            ab, ba = [x["si"] for x in by_leg[(a, b)]], [x["si"] for x in by_leg[(b, a)]]
            return _vertical_explanations_leg(net, a, b, ab, ba, members_of), None
        rows = by_setup_rows[where]
        station = net.sightings[rows[0]["si"]].frm
        expls = [_Expl("instrument_height", f"instrument height at {station}",
                       Extra("ht_offset", setup_members[where], f"HI {station}"), dict(station=station, setups=[where]))]
        by_target: dict[str, list[int]] = defaultdict(list)
        for x in rows:
            by_target[x["to"]].append(x["si"])
        for t, idx in by_target.items():
            if len(idx) < min_n:
                continue
            expls.append(_Expl("target_height", f"target height of {t} as booked at {station}",
                               Extra("ht_offset", idx, f"HT {station}->{t}"), dict(frm=station, to=t)))
            back = by_leg.get((t, station), [])
            if len(back) >= min_n:
                expls.append(_Expl("refraction", f"refraction on {station} ↔ {t}",
                                   Extra("refraction", idx + [x["si"] for x in back], f"{station}<->{t}"),
                                   dict(a=station, b=t)))
        return expls, station

    # Pass 1: the best single explanation at each flagged place.
    flagged = []
    for _, what, where in cands:
        expls, station = explanations(what, where)
        best_single, best_cost = None, math.inf
        for e in expls:
            try:
                c = cost(adjust(net, [e.extra]))
            except np.linalg.LinAlgError:
                continue
            if c < best_cost:
                best_single, best_cost = e, c
        if best_single is not None:
            flagged.append((what, where, expls, station, best_single))

    # Pass 2: test each place with every other place's best explanation in the model, so a misfit next
    # door is not taken as evidence here, and a knock-on of another place's mistake drops out.
    out: list[Finding] = []
    named: set[frozenset] = set()   # readings already behind a reported finding's best explanation
    for what, where, expls, station, own_best in flagged:
        own = {_effective(net, e.extra.members) for e in expls}
        nuisance, seen = [], set()
        for w2, wh2, _, _, other in flagged:
            eff = _effective(net, other.extra.members)
            if (w2, wh2) != (what, where) and eff not in own and eff not in seen:
                nuisance.append(other.extra)
                seen.add(eff)
        cur, cur_cost = res, cost(res)
        if nuisance:
            try:
                cur = adjust(net, nuisance)
                cur_cost = cost(cur)
            except np.linalg.LinAlgError:
                nuisance, cur, cur_cost = [], res, cost(res)
            bl, bs = group(cur.rows)
            if screen(what, where, bl, bs) < crit:
                continue
        compared = _compare(net, nuisance, cur_cost, expls, alpha)
        if compared is None:
            continue
        (best, best_res, best_gain), beaten, tied, both, fits = compared
        if _effective(net, best.extra.members) in named:
            continue  # the same cause is already reported from another place
        if what == "leg":
            a, b = where
        k = len(nuisance)   # index of the tested explanation among the fitted extras
        f_stat = best_gain / (cost(best_res) / best_res.dof)
        p = stats.f.sf(f_stat, 1, best_res.dof) * n_tests
        if p > alpha:
            continue
        gains = {e.label: g for e, _, g in fits}
        value, value_sd = best_res.extras[k]["value"], best_res.extras[k]["sd"]
        if tied or both:
            # No single cause: either rivals explain the misfit as well (tied), or they still improve the fit
            # with the best one in the model (both), so more than one effect may be at work.
            preferred = "either"
            group_ = [best] + [e for e, _ in tied + both]
            marks = sorted({m for e in group_ for m in (e.info.get("station"), e.info.get("to"), e.info.get("a"),
                                                        e.info.get("b")) if m})
            parts = [f"{_cap(best.label)} fits best (χ² {best_gain:.1f})."]
            if tied:
                parts.append(f"{_cap(_join(f'{e.label} (χ² {g:.1f})' for e, g in tied))} fit(s) about as well, and neither "
                             f"improves the fit significantly once the other is in the model.")
            if both:
                parts.append(f"{_cap(_join(f'{e.label} (χ² {g:.1f})' for e, g in both))} still improve(s) the fit "
                             f"significantly with it in the model, so more than one effect may be at work.")
            verdict = (" ".join(parts) + f" This network can't single out one cause. An independent height for "
                       f"{', '.join(marks)} (levelling or GNSS), re-measured instrument and target heights, or "
                       f"simultaneous reciprocal readings would separate them.")
        else:
            preferred = best.key
            verdict = (f"Best explanation: {best.label} (χ² {best_gain:.1f}). With each rival in the model it still "
                       f"improves the fit significantly, and no rival does with it: "
                       f"{_join(f'{e.label} (χ² {g:.1f})' for e, g in beaten)}.")
        if what == "leg":
            mean_f = np.mean([x["res_sec"] for x in by_leg[(a, b)]])
            mean_r = np.mean([x["res_sec"] for x in by_leg[(b, a)]])
            dist = np.mean([net.sightings[x["si"]].slope or 0 for x in by_leg[(a, b)]])
            ref = next((r for e, r, _ in fits if e.key == "refraction"), None)
            lead = (f"Zenith residuals average {mean_f:+.1f}\" from {a} and {mean_r:+.1f}\" from {b} "
                    f"({n_read(by_leg[(a, b)])} and {n_read(by_leg[(b, a)])} readings, {dist:.0f} m). ")
            place = f"{a} ↔ {b}"
            ev = dict(a=a, b=b, mean_res_sec_ab=mean_f, mean_res_sec_ba=mean_r, n_ab=n_read(by_leg[(a, b)]),
                      n_ba=n_read(by_leg[(b, a)]), length_m=dist, dk=ref.extras[k]["value"] if ref else None,
                      dk_sd=ref.extras[k]["sd"] if ref else None)
        else:
            rows = by_setup_rows[where]
            lead = (f"Zenith residuals at the setup on {station} average "
                    f"{np.mean([x['res_sec'] for x in rows]):+.1f}\" over {n_read(rows)} readings. ")
            place = f"Setup at {station}"
            ev = dict(a=station, b=None, setup=where)
        if preferred == "refraction":
            title = f"{best.info['a']} ↔ {best.info['b']}: refraction, not a height error"
            est = f"Estimated Δk = {value:+.2f} ± {value_sd:.2f} (job value {net.refraction_k:.2f}). "
        elif preferred == "target_height":
            title = f"{best.info['frm']} → {best.info['to']}: target height looks wrong"
            est = (f"The target height booked at {best.info['frm']} for {best.info['to']} looks "
                   f"{_height_text(best, value, value_sd)}. ")
        elif preferred == "instrument_height":
            title = f"Setup at {best.info['station']}: instrument height looks wrong"
            est = (f"The instrument height booked at {best.info['station']} looks "
                   f"{_height_text(best, value, value_sd)}. ")
        else:
            title = f"{place}: refraction or a height error, can't tell"
            est = ""
        named.add(_effective(net, best.extra.members))
        winner = dict(key=best.key, label=best.label, value=value, sd=value_sd, **best.info)
        out.append(Finding(
            kind="refraction", title=title, summary=lead + est + verdict,
            evidence=dict(**ev, preferred=preferred, winner=winner, chi2_gains=gains,
                          tied=[e.label for e, _ in tied], beaten=[e.label for e, _ in beaten],
                          also_present=[e.label for e, _ in both], p_bonferroni=p, screen=what),
            severity="likely-cause" if preferred != "either" else "check",
        ))
    return out


def set_means(net: Network, pointing=None) -> tuple[Network, list[int]]:
    """One sighting per (setup, target) set, weighted for what repeating actually buys.

    Pointing error is new on every reading, so it averages down over the n readings of a set. Centring and
    the instrument and target heights are set up and measured once, so their errors are shared by every
    reading and do not. Each reading's standard error is split that way: the pointing part comes from
    ``pointing(sighting) -> (horizontal ", vertical ")`` and the rest is taken as shared. Distances keep
    their single-reading figure (EDM constant, scale and centring are all shared). The default pointing,
    2"/4", is the one recovered from a real CompNet job.

    Returns the network of set means and, per set mean, how many zenith readings it stands for."""
    pointing = pointing or (lambda s: (2.0, 4.0))
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for si, s in enumerate(net.sightings):
        groups[(s.setup, s.to)].append(si)

    def combine(totals: list[float], point: float) -> float:
        total2 = float(np.mean(np.square(totals)))
        return math.sqrt(point ** 2 / len(totals) + max(total2 - point ** 2, 0.0))

    merged: list[Sighting] = []
    counts: list[int] = []
    for idxs in groups.values():
        ss = [net.sightings[i] for i in idxs]
        ph, pv = pointing(ss[0])
        m = copy.copy(ss[0])
        hz = [s for s in ss if s.hz is not None]
        zen = [s for s in ss if s.zen is not None]
        sl = [s.slope for s in ss if s.slope is not None]
        if hz:
            m.hz = math.atan2(np.mean([math.sin(s.hz) for s in hz]), np.mean([math.cos(s.hz) for s in hz])) % (2 * math.pi)
            m.sd_dir = combine([s.sd_dir for s in hz], ph * SEC)
        if zen:
            m.zen = float(np.mean([s.zen for s in zen]))
            m.sd_zen = combine([s.sd_zen for s in zen], pv * SEC)
        m.slope = float(np.mean(sl)) if sl else None
        merged.append(m)
        counts.append(len(zen))
    n2 = copy.copy(net)
    n2.sightings = merged
    return n2, counts


def vertical_check(net: Network, pointing=None, alpha: float = 0.01) -> list[Finding]:
    """``vertical_patterns`` on set means, so repeated readings count for what they are worth."""
    n2, counts = set_means(net, pointing)
    return vertical_patterns(n2, adjust(n2), alpha=alpha, min_n=1, counts=counts)


# ------------------------------------------------------- distance constants

def distance_constants(net: Network, pointing=None, alpha: float = 0.01) -> list[Finding]:
    """A constant error on distances: the instrument at one setup set to the wrong prism constant, or the
    wrong prism on one mark. Runs on set means (a constant is shared by every repeat). Setups and marks are
    screened on their mean distance misfit; at each flagged place a constant for that setup and for each mark
    involved are fitted, and one is named only if it beats every rival with both in the model (the rule the
    vertical check uses). Each place is tested with the other flagged places' best explanations in the model."""
    n2, _ = set_means(net, pointing)
    res = adjust(n2)
    groups: dict[tuple[str, str], int] = {}
    for s in net.sightings:
        if s.slope is not None:
            groups[(s.setup, s.to)] = groups.get((s.setup, s.to), 0) + 1
    rows = [r for r in res.rows if r["kind"] == "DIST"]
    by_setup: dict[str, list[dict]] = defaultdict(list)
    by_target: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_setup[r["setup"]].append(r)
        by_target[r["to"]].append(r)
    places = [("setup", st) for st in by_setup] + [("target", t) for t in by_target]
    crit = stats.chi2.isf(alpha / max(len(places), 1), 1)

    def screen(rows_now, what, where) -> float:
        sel = [r for r in rows_now if r["kind"] == "DIST" and (r["setup"] if what == "setup" else r["to"]) == where]
        if not sel:
            return 0.0
        v = np.array([r["vw"] for r in sel])
        return float(v.sum() ** 2 / len(v))

    def readings(rows_sel) -> int:
        return sum(groups.get((r["setup"], r["to"]), 1) for r in rows_sel)

    station_of = {r["setup"]: r["frm"] for r in rows}
    setup_expl = {st: _Expl("setup_constant", f"a constant on every distance from {station_of[st]}",
                            Extra("dist_const", [r["si"] for r in rs], f"EDM {station_of[st]}"),
                            dict(station=station_of[st], setup=st))
                  for st, rs in by_setup.items()}
    target_expl = {t: _Expl("target_constant", f"a constant on every distance to {t}",
                            Extra("dist_const", [r["si"] for r in rs], f"prism {t}"), dict(target=t))
                   for t, rs in by_target.items()}

    def explanations(what, where):
        """A setup's own constant competes with the prism on each mark it measured and with the prism on its
        own station mark (distances to it move with it); a mark's prism with each setup that measured it and
        with that mark's own setups."""
        if what == "setup":
            stn = station_of[where]
            marks = sorted({r["to"] for r in by_setup[where]} | ({stn} if stn in target_expl else set()))
            return [setup_expl[where]] + [target_expl[t] for t in marks]
        own = {st for st in setup_expl if station_of[st] == where}
        return [target_expl[where]] + [setup_expl[st] for st in sorted({r["setup"] for r in by_target[where]} | own)]

    flagged = []
    for what, where in sorted(places, key=lambda p: -screen(res.rows, *p)):
        if screen(res.rows, what, where) < crit:
            continue
        expls = explanations(what, where)
        best, best_cost = None, math.inf
        for e in expls:
            try:
                c = cost(adjust(n2, [e.extra]))
            except np.linalg.LinAlgError:
                continue
            if c < best_cost:
                best, best_cost = e, c
        if best is not None:
            flagged.append((what, where, expls, best))

    out: list[Finding] = []
    named: set = set()
    for what, where, expls, _ in flagged:
        own = {_moves(n2, e.extra) for e in expls}
        nuisance, seen = [], set()
        for w2, wh2, _, other in flagged:
            mv = _moves(n2, other.extra)
            if (w2, wh2) != (what, where) and mv not in own and mv not in seen:
                nuisance.append(other.extra)
                seen.add(mv)
        cur = res
        if nuisance:
            try:
                cur = adjust(n2, nuisance)
            except np.linalg.LinAlgError:
                nuisance, cur = [], res
            if screen(cur.rows, what, where) < crit:
                continue
        compared = _compare(n2, nuisance, cost(cur), expls, alpha)
        if compared is None:
            continue
        (best, best_res, best_gain), beaten, tied, both, fits = compared
        k = len(nuisance)
        p = stats.f.sf(best_gain / (cost(best_res) / best_res.dof), 1, best_res.dof) * len(places)
        if p > alpha or _moves(n2, best.extra) in named:
            continue
        named.add(_moves(n2, best.extra))
        value, value_sd = best_res.extras[k]["value"], best_res.extras[k]["sd"]
        sel = by_setup[where] if what == "setup" else by_target[where]
        mean_mm = float(np.mean([r["res"] for r in sel])) * 1000
        place = (f"the setup on {station_of[where]}" if what == "setup" else f"distances to {where}")
        lead = (f"Distance residuals at {place} average {mean_mm:+.1f} mm (computed minus observed) over "
                f"{readings(sel)} readings in {len(sel)} sets. ")
        gains = {e.label: g for e, _, g in fits}
        if tied or both:
            preferred = "either"
            group_ = [best] + [e for e, _ in tied + both]
            parts = [f"{_cap(best.label)} fits best (χ² {best_gain:.1f})."]
            if tied:
                parts.append(f"{_cap(_join(f'{e.label} (χ² {g:.1f})' for e, g in tied))} fit(s) about as well, and "
                             f"neither improves the fit significantly once the other is in the model.")
            if both:
                parts.append(f"{_cap(_join(f'{e.label} (χ² {g:.1f})' for e, g in both))} still improve(s) the fit "
                             f"significantly with it in the model, so more than one effect may be at work.")
            title = (f"Setup at {station_of[where]}: distances off, setup or prism can't tell" if what == "setup"
                     else f"Distances to {where}: off, prism or setup can't tell")
            verdict = (" ".join(parts) + " A check of the prism constant set in the instrument and of the prism "
                       "used on " + ", ".join(sorted({e.info.get('target') or e.info.get('station') for e in group_}))
                       + " would separate them.")
            est = ""
        else:
            preferred = best.key
            word = "long" if value > 0 else "short"
            if best.key == "setup_constant":
                title = f"Setup at {best.info['station']}: distances read {abs(value) * 1000:.1f} mm {word}"
                est = (f"Every distance from {best.info['station']} looks {abs(value) * 1000:.1f} ± "
                       f"{value_sd * 1000:.1f} mm too {word}: the pattern of a prism constant set wrong in the "
                       f"instrument for this setup. ")
            else:
                title = f"Distances to {best.info['target']} read {abs(value) * 1000:.1f} mm {word}"
                est = (f"Every distance to {best.info['target']} looks {abs(value) * 1000:.1f} ± "
                       f"{value_sd * 1000:.1f} mm too {word}: the pattern of the wrong prism, or a wrong prism "
                       f"constant, on that mark. ")
            verdict = (f"Best explanation: {best.label} (χ² {best_gain:.1f}). With each rival in the model it still "
                       f"improves the fit significantly, and no rival does with it: "
                       f"{_join(f'{e.label} (χ² {g:.1f})' for e, g in beaten)}.")
        winner = dict(key=best.key, label=best.label, value=value, sd=value_sd, **best.info)
        ev = dict(setup=where, station=station_of[where]) if what == "setup" else dict(target=where)
        out.append(Finding(
            kind="prism", title=title, summary=lead + est + verdict + " This is a proposal: the data is unchanged.",
            evidence=dict(**ev, place=what, preferred=preferred, winner=winner, chi2_gains=gains,
                          tied=[e.label for e, _ in tied], beaten=[e.label for e, _ in beaten],
                          also_present=[e.label for e, _ in both], p_bonferroni=p, mean_res_mm=mean_mm),
            severity="likely-cause" if preferred != "either" else "check",
        ))
    return out

# ------------------------------------------------------- single readings

def orientation_corrections(findings: list[Finding], net: Network) -> dict[int, float]:
    """Arc-seconds to add to each direction reading so it is on the setup's first orientation, from the
    orientation step and drift findings on ``net``. Keyed by the reading itself (``id`` of the sighting),
    so it still applies in a subset of the network."""
    corr: dict[int, float] = {}
    for f in findings:
        e = f.evidence
        if f.kind == "orientation_step":
            for r in e["readings"][e["k"]:]:
                corr[id(net.sightings[r["si"]])] = e["shift"]
        elif f.kind == "orientation_drift":
            t = [_minutes(r["time"]) for r in e["readings"]]
            if None in t:
                continue
            t0 = min(t)
            for r, ti in zip(e["readings"], t):
                corr[id(net.sightings[r["si"]])] = e["rate"] * (ti - t0)
    return corr


def bad_pointings(net: Network, pointing=None, alpha: float = 0.01,
                  corrections: dict[int, float] | None = None) -> list[Finding]:
    """A single reading that disagrees with the other repeats of its own set (same setup, same mark).

    Within a set only the pointing changes from reading to reading: centring and the measured heights are
    the same for every repeat. So each direction and zenith is compared with the mean of the other repeats,
    in units of the pointing error x sqrt(1 + 1/(n-1)), and flagged above the normal limit at ``alpha``
    split over every reading tested (sets of 3 or more). Pointing comes from ``pointing(sighting) ->
    (horizontal ", vertical ")``, default 2"/4" (recovered from a real CompNet job). ``corrections`` (from ``orientation_corrections``) put
    directions on one orientation first, so a setup that twisted part-way is not read as bad pointings.
    One flagged reading in a set is called a bad pointing; several are reported as a split set. Proposal
    only."""
    corrections = corrections or {}
    pointing = pointing or (lambda s: (2.0, 4.0))
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for si, s in enumerate(net.sightings):
        groups[(s.setup, s.to)].append(si)
    # Per set and component: the readings and their offsets from the first, in arc-seconds.
    blocks = []
    for key, idx in groups.items():
        for comp in ("hz", "zen"):
            sel = [i for i in idx if getattr(net.sightings[i], comp) is not None]
            if len(sel) < 3:
                continue
            ref = getattr(net.sightings[sel[0]], comp)
            vals = {i: math.remainder(getattr(net.sightings[i], comp) - ref, 2 * math.pi) / SEC
                    + (corrections.get(id(net.sightings[i]), 0.0) if comp == "hz" else 0.0) for i in sel}
            blocks.append((key, comp, vals, pointing(net.sightings[sel[0]])[0 if comp == "hz" else 1]))
    n_tested = sum(len(b[2]) for b in blocks)
    if not n_tested:
        return []
    crit = float(stats.norm.isf(alpha / (2 * n_tested)))
    # Within each set, take out the worst reading while it fails, so one bad reading cannot drag the
    # comparison for its neighbours.
    flagged: dict[tuple[str, str], list[tuple]] = defaultdict(list)
    for key, comp, vals, p in blocks:
        vals = dict(vals)
        while len(vals) >= 3:
            n = len(vals)
            total = sum(vals.values())
            scale = p * math.sqrt(1 + 1 / (n - 1))
            i, d = max(((i, v - (total - v) / (n - 1)) for i, v in vals.items()), key=lambda t: abs(t[1]))
            z = abs(d) / scale
            if z <= crit:
                break
            flagged[key].append((key, i, comp, float(d), float(z), n, p))
            del vals[i]
    tests = n_tested
    out = []
    for (setup, to), hits in sorted(flagged.items(), key=lambda kv: -max(h[4] for h in kv[1])):
        station = net.sightings[hits[0][1]].frm
        idx = groups[(setup, to)]
        dists = [net.sightings[i].slope for i in idx if net.sightings[i].slope]
        sight = float(np.median(dists)) if dists else None
        parts = []
        split = False
        for comp in ("hz", "zen"):
            hc = sorted([h for h in hits if h[2] == comp], key=lambda h: -h[4])
            if not hc:
                continue
            name = "direction" if comp == "hz" else "zenith"
            n_all = hc[0][5] if len(hc) == 1 else len([i for i in idx if getattr(net.sightings[i], comp) is not None])
            if len(hc) == 1:
                _, i, _, d, z, n, p = hc[0]
                parts.append(f"{net.sightings[i].ref}: {name} {d:+.1f}\" from the mean of the other {n - 1} repeats "
                             f"({z:.1f} × the {p:g}\" pointing error, allowing for that mean)")
            else:
                split = True
                ds = [h[3] for h in hc]
                parts.append(f"{len(hc)} of {n_all} {name}s sit together {np.mean(ds):+.1f}\" (from {min(ds):+.1f} to "
                             f"{max(ds):+.1f}) from the rest: {', '.join(net.sightings[h[1]].ref for h in hc)}")
        length = (f" The sight is {sight:.1f} m long, where 1 mm across the line of sight is "
                  f"{0.001 / sight / SEC:.0f}\"." if sight else "")
        if split:
            title = f"{station} → {to}: the set splits into two groups"
            meaning = (" More than one reading moves together, so this is not a single bad pointing: the set holds "
                       "two groups of readings that disagree.")
        else:
            title = f"{station} → {to}: one bad pointing"
            meaning = (" Centring and heights are the same for every repeat, so only the pointing can differ: this "
                       "looks like a single bad pointing.")
        out.append(Finding(
            kind="pointing", title=title,
            summary=("; ".join(parts) + f". The limit is {crit:.2f} (1% split over the {tests} readings tested)."
                     + meaning + length + " This is a proposal: the data is unchanged."),
            evidence=dict(setup=setup, station=station, to=to, si=sorted({h[1] for h in hits}), split=split,
                          sight_m=sight,
                          readings=[dict(ref=net.sightings[i].ref, component=comp, diff_sec=d, z=z, n=n)
                                    for _, i, comp, d, z, n, _p in hits], limit=crit),
            severity="likely-cause" if not split else "check",
        ))
    return out

# ------------------------------------------------------- point numbers

def _model_obs(net: Network, res: Result, s: Sighting, to: str) -> tuple[float, float, float]:
    """Direction, zenith and slope distance the adjusted network predicts for sighting ``s`` if it had been
    to mark ``to`` (same formulas as ``adjust``)."""
    names = [s.frm, to]
    enh = np.array([res.coords[n] for n in names], float)
    known = [v for v in net.geoid.values() if v is not None and not math.isnan(v)]
    n_mean = float(np.mean(known)) if known else 0.0
    nv = np.array([net.geoid.get(n, n_mean) for n in names], float)
    xyz_g, _, lat, lon = G.grid_ecef(enh[:, 0], enh[:, 1], enh[:, 2] + nv, net.epsg)
    if net.level_frame:
        xyz, _, _, _ = G.grid_ecef(enh[:, 0], enh[:, 1], enh[:, 2] + n_mean, net.epsg)
    else:
        xyz = xyz_g
    R = G.enu_matrix(lat, lon)
    up = R[:, 2]
    e = R[0] @ ((xyz[1] + s.ht * up[1]) - (xyz[0] + s.hi * up[0]))
    rho = math.hypot(e[0], e[1])
    hz = math.atan2(e[0], e[1]) - res.orientations.get(s.setup, 0.0)
    zen = math.atan2(rho, e[2]) - net.refraction_k * rho / (2 * G.R_MEAN)
    return hz, zen, float(np.linalg.norm(e))


def _misfit(net: Network, res: Result, s: Sighting, to: str) -> list[float]:
    """|obs - model| / sd for each component of ``s`` if it were a sighting to ``to``."""
    hz, zen, dist = _model_obs(net, res, s, to)
    out = []
    if s.hz is not None:
        out.append(abs(math.remainder(s.hz - hz, 2 * math.pi)) / s.sd_dir)
    if s.zen is not None:
        out.append(abs(s.zen - zen) / s.sd_zen)
    if s.slope is not None:
        out.append(abs(s.slope - dist) / s.sd_dist)
    return out


def _subset(net: Network, keep: list[int], relabel: dict[int, str] | None = None) -> Network:
    n2 = copy.copy(net)
    n2.sightings = []
    for si in keep:
        s = net.sightings[si]
        if relabel and si in relabel:
            s = copy.copy(s)
            s.to = relabel[si]
        n2.sightings.append(s)
    return n2


def _w_limit(res: Result, alpha: float) -> float:
    n = sum(1 for r in res.rows if r["kind"] in ("DIR", "ZEN", "DIST") and not math.isnan(r["w"]))
    return float(stats.norm.isf(alpha / (2 * max(n, 1))))


def swapped_names(net: Network, res: Result, alpha: float = 0.01, max_out: int = 25) -> list[Finding]:
    """Readings that fail the w-test and fit another mark seen from the same setup.

    A wrong point number can pull the whole adjustment off, so the worst failing reading is set aside and
    the network re-adjusted, until nothing fails (iterative data snooping; at most ``max_out`` readings).
    Each reading set aside is then compared with every other mark its setup observed, using that clean
    adjustment. A relabelling is proposed only when exactly one mark fits every component within the
    w-test limit, and it is confirmed by re-adjusting with the reading back in under the new name.
    Repeats with the same proposal are reported together; two names booked the wrong way round at one
    setup are reported as a swap. Nothing is taken out of the job itself."""
    crit = _w_limit(res, alpha)
    keep = list(range(len(net.sightings)))
    out_order: list[tuple[int, float]] = []
    cur = res
    while len(out_order) < max_out:
        rows = [r for r in cur.rows if r["kind"] in ("DIR", "ZEN", "DIST") and not math.isnan(r["w"])]
        worst = max(rows, key=lambda r: r["w"], default=None)
        if worst is None or worst["w"] <= crit:
            break
        si = keep[worst["si"]]
        out_order.append((si, worst["w"]))
        keep = [i for i in keep if i != si]
        try:
            cur = adjust(_subset(net, keep))
        except (np.linalg.LinAlgError, ValueError, KeyError):
            return []
    if not out_order:
        return []
    clean_net, clean = _subset(net, keep), cur
    seen_from: dict[str, set[str]] = defaultdict(set)
    for s in net.sightings:
        seen_from[s.setup].add(s.to)

    proposals: dict[tuple[str, str, str], list[int]] = defaultdict(list)   # (setup, booked, proposed) -> readings
    w_first = dict(out_order)
    note: dict[int, float] = {}
    for si, _ in out_order:
        s = net.sightings[si]
        if s.setup not in clean.orientations and s.hz is not None:
            continue  # the whole setup went: nothing left to orient the reading against
        fits = []
        for x in sorted(seen_from[s.setup] - {s.to, s.frm}):
            if x in clean.coords and s.frm in clean.coords:
                m = _misfit(clean_net, clean, s, x)
                if m and max(m) <= crit:
                    fits.append(x)
        if len(fits) == 1 and s.to in clean.coords:
            proposals[(s.setup, s.to, fits[0])].append(si)
            note[si] = max(_misfit(clean_net, clean, s, s.to))

    findings: list[Finding] = []
    done: set[tuple[str, str, str]] = set()
    for key in sorted(proposals, key=lambda k: -len(proposals[k])):
        if key in done:
            continue
        setup, booked, proposed = key
        partner = (setup, proposed, booked)
        pair = partner in proposals and partner not in done
        mine = proposals[key]
        theirs = proposals[partner] if pair else []
        done.add(key)
        if pair:
            done.add(partner)
        members = mine + theirs
        relabel = {si: proposed for si in mine} | {si: booked for si in theirs}
        back = sorted(keep + members)
        try:
            as_booked = adjust(_subset(net, back))
            renamed = adjust(_subset(net, back, relabel))
        except (np.linalg.LinAlgError, ValueError, KeyError):
            continue
        pos = {si: k for k, si in enumerate(back)}
        after = max((r["w"] for r in renamed.rows if r.get("si") in {pos[m] for m in members}
                     and not math.isnan(r["w"])), default=0.0)
        gain = cost(as_booked) - cost(renamed)
        if after > crit or gain <= 0:
            continue
        station = net.sightings[members[0]].frm
        refs = [net.sightings[si].ref for si in members]
        if pair:
            title = f"{station}: {booked} and {proposed} look swapped"
            what = (f"At the setup on {station}, {len(mine)} reading(s) booked to {booked} fit {proposed}, "
                    f"and {len(theirs)} booked to {proposed} fit {booked}")
        else:
            title = f"{station}: reading(s) booked to {booked} look like sightings to {proposed}"
            what = f"At the setup on {station}, {len(members)} reading(s) booked to {booked} fit {proposed}"
        findings.append(Finding(
            kind="swap", title=title,
            summary=(f"{what} ({'; '.join(refs[:4])}{' …' if len(refs) > 4 else ''}). Against the network "
                     f"adjusted without them, the worst part of these readings is {max(note[m] for m in members):.0f} "
                     f"standard errors off as booked, and every part is within the w-test limit ({crit:.2f}) "
                     f"under the proposed name(s); no other mark seen from this setup fits. Put back with the "
                     f"names changed, the largest w on them is {after:.2f} (as booked: "
                     f"{max(w_first[m] for m in members):.1f}) and the fit improves by χ² {gain:.1f}. "
                     f"This is a proposal: the data is unchanged."),
            evidence=dict(setup=setup, station=station, booked=booked, proposed=proposed, pair=pair, si=members,
                          refs=refs, w_before=max(w_first[m] for m in members), w_after=after, chi2_gain=gain,
                          w_limit=crit),
            severity="likely-cause",
        ))
    return findings

# --------------------------------------------------------- repeats

def set_averaged(net: Network) -> tuple[Network, dict]:
    """A copy of the network with each repeated (setup, target) set averaged into one sighting."""
    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for si, s in enumerate(net.sightings):
        groups[(s.setup, s.to)].append(si)
    merged: list[Sighting] = []
    for (setup, to), idxs in groups.items():
        ss = [net.sightings[i] for i in idxs]
        hz = [s.hz for s in ss if s.hz is not None]
        zen = [s.zen for s in ss if s.zen is not None]
        sl = [s.slope for s in ss if s.slope is not None]
        m = copy.copy(ss[0])
        m.hz = math.atan2(np.mean(np.sin(hz)), np.mean(np.cos(hz))) % (2 * math.pi) if hz else None
        m.zen = float(np.mean(zen)) if zen else None
        m.slope = float(np.mean(sl)) if sl else None
        merged.append(m)
    n2 = copy.copy(net)
    n2.sightings = merged
    return n2, groups


def repeat_effect(net: Network, res: Result, averaged: tuple[Network, Result, dict] | None = None) -> Finding:
    """Re-adjust with each repeated (setup, target) set averaged into one measurement."""
    if averaged is None:
        n2, groups = set_averaged(net)
        r2 = adjust(n2)
    else:
        n2, r2, groups = averaged
    reps = [len(v) for v in groups.values()]
    lo = stats.chi2.ppf(0.025, r2.dof) / r2.dof
    hi = stats.chi2.ppf(0.975, r2.dof) / r2.dof
    sd_ratio = []
    for n in res.sd:
        if n in r2.sd and res.sd[n][0] > 0:
            sd_ratio.append(np.mean(np.array(r2.sd[n]) / np.array(res.sd[n])))
    return Finding(
        kind="repeats",
        title="Repeated readings counted as independent",
        summary=(f"{len(net.sightings)} sightings come from {len(groups)} setup-to-target sets "
                 f"(median {int(np.median(reps))} readings per set). Repeats from one setup share the same "
                 f"centring, so they are not independent. If each set counts as one measurement, the variance "
                 f"factor is {r2.vf:.2f} with {r2.dof} degrees of freedom (95% band {lo:.2f}–{hi:.2f}), "
                 f"and the a-priori coordinate standard deviations grow by about {np.median(sd_ratio):.1f}×."
                 + (" So counting repeats is not why the variance factor is low: even set by set, the data fits "
                    "better than the standard errors assume." if r2.vf < lo else "")),
        evidence=dict(n_sightings=len(net.sightings), n_sets=len(groups), median_repeats=float(np.median(reps)),
                      vf_sets=r2.vf, dof_sets=r2.dof, band=(lo, hi), sd_growth_median=float(np.median(sd_ratio)),
                      group_vf_sets=r2.group_vf),
        severity="check",
    )


# ------------------------------------------------------- unchecked marks

def unchecked_marks(net: Network, res: Result, r_min: float = 0.02) -> Finding | None:
    """Marks no independent observation can check. Pass the *set-averaged*
    network and result: repeated readings from one setup check each other but
    not the mark, so they must not count as redundancy here."""
    by_pt: dict[str, list[float]] = defaultdict(list)
    for r in res.rows:
        if r["kind"] in ("DIR", "ZEN", "DIST"):
            by_pt[r["to"]].append(r["r"])
            by_pt[r["frm"]].append(r["r"])
    weak = sorted(n for n, rs in by_pt.items() if n not in net.fixed and max(rs) < r_min)
    if not weak:
        return None
    return Finding(
        kind="unchecked",
        title="Marks with no independent check",
        summary=(f"{len(weak)} marks are fixed by a single set of observations from one setup (repeats of the "
                 f"same sighting don't count), so a wrong target height, prism or point number on them could not "
                 f"be detected: {', '.join(weak)}."),
        evidence=dict(marks=weak),
        severity="check",
    )


def worst_residuals(res: Result, n: int = 10) -> list[dict]:
    rows = [r for r in res.rows if r["kind"] in ("DIR", "ZEN", "DIST") and not math.isnan(r["w"])]
    return sorted(rows, key=lambda r: -r["w"])[:n]


def diagnose(net: Network, res: Result | None = None, pointing=None) -> tuple[Result, list[Finding]]:
    """``pointing(sighting) -> (horizontal, vertical)`` pointing errors in arc-seconds; see ``set_means``."""
    res = res or adjust(net)
    findings = [global_test(res)]
    findings += orientation_changes(net, res)
    swaps = swapped_names(net, res)
    findings += swaps
    # Readings already explained as wrong point numbers would drag their set means off, so the set-mean
    # checks run without them.
    drop = {si for f in swaps for si in f.evidence["si"]}
    rest = _subset(net, [i for i in range(len(net.sightings)) if i not in drop]) if drop else net
    findings += vertical_check(rest, pointing)
    findings += distance_constants(rest, pointing)
    findings += bad_pointings(rest, pointing, corrections=orientation_corrections(findings, net))
    n2, groups = set_averaged(net)
    r2 = adjust(n2)
    findings.append(repeat_effect(net, res, (n2, r2, groups)))
    u = unchecked_marks(n2, r2)
    if u:
        findings.append(u)
    return res, findings
