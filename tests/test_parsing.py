import pytest
from events_env.io.parsing import parse_schedule_any


CASES = [
    ('{"schedule":[{"name":"A","start":"01:00","end":"02:00"}]}', True),
    ('```json\n{"schedule":[{"name":"A","start":"01:00","end":"02:00"}]}\n```', True),
    ('<schedule><event><name>A</name><start>01:00</start><end>02:00</end></event></schedule>', True),
    ('<think>blah</think>{"schedule":[{"name":"A","start":"01:00","end":"02:00"}]}', True),
    ('not json or xml', False),
]


@pytest.mark.parametrize("text,ok", CASES)
def test_parse_schedule_any(text, ok):
    got = parse_schedule_any(text, allow_reasoning_tag=True)
    assert (got is not None) == ok


