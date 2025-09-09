from events_env.evals.conflict_checker import check_conflicts


def test_conflicts_basic_no_issues():
    events = [["A","01:00","02:00"], ["B","02:00","03:00"]]
    prop = [{"name":"A","start":"01:00","end":"02:00"},
            {"name":"B","start":"02:00","end":"03:00"}]
    rep = check_conflicts(prop, events, strict_times=True)
    assert rep["summary"] == "No issues found."
    assert not rep["overlaps"] and rep["min_gap_violations"] == 0


def test_conflicts_overlap_and_duplicate():
    events = [["A","01:00","02:00"], ["B","01:30","02:30"]]
    prop = [{"name":"A","start":"01:00","end":"02:00"},
            {"name":"A","start":"01:00","end":"02:00"},
            {"name":"B","start":"01:30","end":"02:30"}]
    rep = check_conflicts(prop, events, strict_times=True)
    assert "duplicate" in rep["summary"]
    assert rep["overlaps"]


