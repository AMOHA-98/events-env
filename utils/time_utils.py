def hhmm_to_min(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)

def min_to_hhmm(x: int) -> str:
    h = x // 60
    m = x % 60
    return f"{h:02d}:{m:02d}"

def duration_min(start: str, end: str, *, allow_cross_midnight: bool = False) -> int:
    s = hhmm_to_min(start)
    e = hhmm_to_min(end)
    if allow_cross_midnight and e < s:
        e += 1440
    return max(0, e - s)

def no_overlap(intervals: list[tuple[int,int]]) -> bool:
    ints = sorted(intervals, key=lambda x: x[0])
    for i in range(len(ints) - 1):
        if ints[i][1] > ints[i+1][0]:
            return False
    return True

def find_overlaps(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]:
    ints = sorted(intervals, key=lambda x: x[0])
    overlaps = []
    for i in range(len(ints) - 1):
        if ints[i][1] > ints[i+1][0]:
            overlaps.append((i, i+1))
    return overlaps
 