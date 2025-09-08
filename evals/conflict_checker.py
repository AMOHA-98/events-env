from typing import List, Dict, Any, Tuple
from ..utils.time_utils import hhmm_to_min, duration_min, find_overlaps

def events_index(events: List[List[str]]) -> dict[str, tuple[str,str]]:
    return {name: (start, end) for name, start, end in events}

def check_conflicts(
    proposal: List[Dict[str,str]],
    catalog: List[List[str]],
    *,
    strict_times: bool = True,
    day_start: str = "00:00",
    day_end: str = "24:00",
    allow_cross_midnight: bool = False,
    min_gap_minutes: int = 0,
) -> Dict[str, Any]:
    idx = events_index(catalog)
    report: Dict[str, Any] = {
        "not_in_catalog": [],
        "time_mismatches": [],
        "duplicates": [],
        "nonpositive": [],
        "out_of_bounds": [],
        "overlaps": [],
        "min_gap_violations": 0,
        "summary": "",
    }
    seen = set()
    intervals: List[Tuple[int,int]] = []
    chosen_norm: List[Tuple[str,str,str]] = []

    ds = hhmm_to_min(day_start)
    de = hhmm_to_min(day_end)

    for e in proposal:
        nm, st, en = e["name"], e["start"], e["end"]
        if nm not in idx:
            report["not_in_catalog"].append(nm);  continue
        vst, ven = idx[nm]
        if strict_times and (st != vst or en != ven):
            report["time_mismatches"].append(nm);  continue
        # normalize to catalog times when strict
        st, en = (vst, ven) if strict_times else (st, en)

        if nm in seen:
            report["duplicates"].append(nm);  continue
        seen.add(nm)

        if duration_min(st, en) <= 0:
            report["nonpositive"].append(nm);  continue

        smin, emin = hhmm_to_min(st), hhmm_to_min(en)
        if not allow_cross_midnight and (smin < ds or emin > de):
            report["out_of_bounds"].append(nm);  continue

        intervals.append((smin, emin))
        chosen_norm.append((nm, st, en))

    # overlaps
    overlaps = find_overlaps(intervals)
    report["overlaps"] = overlaps

    # min-gap
    if min_gap_minutes > 0:
        ints = sorted(intervals, key=lambda x: x[0])
        violations = 0
        for i in range(len(ints) - 1):
            gap = ints[i+1][0] - ints[i][1]
            if gap < min_gap_minutes:
                violations += 1
        report["min_gap_violations"] = violations

    # Summary text (human feedback)
    bullets = []
    if report["not_in_catalog"]:
        bullets.append(f"- {len(report['not_in_catalog'])} event(s) not in catalog: {sorted(set(report['not_in_catalog']))}")
    if report["time_mismatches"]:
        bullets.append(f"- {len(report['time_mismatches'])} time mismatch(es): {sorted(set(report['time_mismatches']))}")
    if report["duplicates"]:
        bullets.append(f"- {len(report['duplicates'])} duplicate(s): {sorted(set(report['duplicates']))}")
    if report["nonpositive"]:
        bullets.append(f"- {len(report['nonpositive'])} nonpositive duration: {sorted(set(report['nonpositive']))}")
    if report["out_of_bounds"]:
        bullets.append(f"- {len(report['out_of_bounds'])} outside day bounds: {sorted(set(report['out_of_bounds']))}")
    if overlaps:
        bullets.append(f"- {len(overlaps)} overlap(s) detected")
    if report["min_gap_violations"]:
        bullets.append(f"- {report['min_gap_violations']} min-gap violation(s)")

    report["summary"] = "No issues found." if not bullets else "Issues:\n" + "\n".join(bullets)
    report["normalized"] = [{"name": nm, "start": st, "end": en} for (nm, st, en) in chosen_norm]
    return report
