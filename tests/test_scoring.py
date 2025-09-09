from events_env.evals.scoring import score_with_penalties, wis_optimum
from events_env.core.config import PenaltiesMinutes, RealismConfig


def test_score_monotonic_and_clipped():
    events = [["A","01:00","03:00"], ["B","03:00","04:00"]]
    prop  = [{"name":"A","start":"01:00","end":"03:00"}]
    pen = PenaltiesMinutes(overlap=1000.0)
    realism = RealismConfig(enforce_day_bounds=True)

    base_only, _ = score_with_penalties(prop, events, [], strict_times=True, penalties=pen, realism=realism)
    assert base_only > 0

    prop2 = prop + [{"name":"B","start":"03:00","end":"04:00"}]
    s2, _ = score_with_penalties(prop2, events, [], strict_times=True, penalties=pen, realism=realism)
    assert s2 >= 0


def test_wis_no_overlap_equals_sum():
    events = [["A","01:00","02:00"], ["B","02:00","04:00"]]
    opt = wis_optimum(events, [], RealismConfig())
    assert opt == 60 + 120


def test_wis_picks_heavier_when_overlap():
    events = [["A","01:00","03:00"], ["B","02:00","04:00"]]
    opt = wis_optimum(events, ["A"], RealismConfig())
    assert opt == 2*(120)


