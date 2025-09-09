from typing import List, Dict, Tuple
from ..utils.time_utils import hhmm_to_min, duration_min, find_overlaps
from ..core.config import PenaltiesMinutes, RealismConfig

def events_index(events: List[List[str]]) -> dict[str, tuple[str,str]]:
    return {name: (start, end) for name, start, end in events}

def weighted_minutes(name: str, start: str, end: str, priority_names: set[str], *, allow_cross_midnight: bool) -> float:
    w = 2.0 if name in priority_names else 1.0
    return w * duration_min(start, end, allow_cross_midnight=allow_cross_midnight)

def wis_optimum(events: List[List[str]], priority_events: List[str], realism: RealismConfig) -> float:
    """Weighted Interval Scheduling optimum in minutes, with optional day-bounds filtering."""
    ds = hhmm_to_min(realism.day_start)
    de = hhmm_to_min(realism.day_end)
    pset = set(priority_events)
    items: List[Tuple[int,int,float]] = []
    for name, s, e in events:
        sm, em = hhmm_to_min(s), hhmm_to_min(e)
        if realism.enforce_day_bounds and not realism.allow_cross_midnight:
            if sm < ds or em > de:
                continue
        dur = duration_min(s, e, allow_cross_midnight=realism.allow_cross_midnight)
        if dur <= 0:
            continue
        # Normalize wrapped intervals by extending end past midnight when needed
        end_norm = em + 1440 if (realism.allow_cross_midnight and em < sm) else em
        w = (2.0 if name in pset else 1.0) * dur
        items.append((sm, end_norm, w))
    items.sort(key=lambda x: x[1])
    n = len(items)
    if n == 0: return 1.0
    ends = [it[1] for it in items]
    # p(j): rightmost compatible
    p = [ -1 ] * n
    for j in range(n):
        lo, hi = 0, j-1
        pj = -1
        while lo <= hi:
            mid = (lo+hi)//2
            if ends[mid] <= items[j][0]:
                pj = mid; lo = mid+1
            else:
                hi = mid-1
        p[j] = pj
    M = [0.0]*(n+1)
    for j in range(1, n+1):
        take = items[j-1][2] + (M[p[j-1]+1] if p[j-1] != -1 else 0.0)
        skip = M[j-1]
        M[j] = max(take, skip)
    return max(1.0, M[n])  # guard to avoid /0

def score_with_penalties(
    proposal: List[Dict[str,str]],
    events: List[List[str]],
    priority_events: List[str],
    *,
    strict_times: bool,
    penalties: PenaltiesMinutes,
    realism: RealismConfig
) -> tuple[float, dict]:
    """
    Returns (minutes_after_penalties, diagnostics).
    Base minutes = sum weighted minutes for valid chosen events.
    Penalties expressed in minutes; subtracted then clamped at 0.
    """
    idx = events_index(events)
    pset = set(priority_events)
    seen = set()
    base_minutes = 0.0
    penalty_minutes = 0.0
    intervals: List[tuple[int,int]] = []

    ds = hhmm_to_min(realism.day_start)
    de = hhmm_to_min(realism.day_end)

    for e in proposal:
        nm, st, en = e["name"], e["start"], e["end"]
        if nm not in idx:
            penalty_minutes += penalties.hallucinated_event;  continue
        vst, ven = idx[nm]
        if strict_times and (st != vst or en != ven):
            penalty_minutes += penalties.time_mismatch;  continue
        # Normalize to catalog times under strict mode
        st, en = (vst, ven) if strict_times else (st, en)

        if nm in seen:
            penalty_minutes += penalties.duplicate_event;  continue
        seen.add(nm)

        dur = duration_min(st, en, allow_cross_midnight=realism.allow_cross_midnight)
        if dur <= 0:
            penalty_minutes += penalties.nonpositive_duration;  continue

        smin, emin = hhmm_to_min(st), hhmm_to_min(en)
        if realism.enforce_day_bounds and not realism.allow_cross_midnight:
            if smin < ds or emin > de:
                penalty_minutes += penalties.out_of_bounds;  continue

        base_minutes += weighted_minutes(nm, st, en, pset, allow_cross_midnight=realism.allow_cross_midnight)
        end_norm = emin + 1440 if (realism.allow_cross_midnight and emin < smin) else emin
        intervals.append((smin, end_norm))

    # overlaps
    overlaps = find_overlaps(intervals)
    penalty_minutes += penalties.overlap * len(overlaps)

    # min-gap
    if realism.min_gap_minutes > 0:
        ints = sorted(intervals, key=lambda x: x[0])
        for i in range(len(ints) - 1):
            gap = ints[i+1][0] - ints[i][1]
            if gap < realism.min_gap_minutes:
                penalty_minutes += penalties.min_gap_violation

    final_minutes = max(0.0, base_minutes - penalty_minutes)
    diag = {
        "base_minutes": base_minutes,
        "penalty_minutes": penalty_minutes,
        "overlaps": len(overlaps),
        "intervals": intervals,
    }
    return final_minutes, diag
