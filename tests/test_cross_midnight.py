from events_env.utils.time_utils import duration_min


def test_duration_cross_midnight_enabled():
    assert duration_min("23:30", "00:30", allow_cross_midnight=True) == 60


def test_duration_cross_midnight_disabled():
    assert duration_min("23:30", "00:30", allow_cross_midnight=False) == 0


