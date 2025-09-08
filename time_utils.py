def hhmm_to_min(t: str) -> int:
    h, m = t.split(":")
    return int(h) * 60 + int(m)

def min_to_hhmm(x: int) -> str:
    h = x // 60
    m = x % 60
    return f"{h:02d}:{m:02d}"

def duration_min(start: str, end: str) -> int:
    return max(0, hhmm_to_min(end) - hhmm_to_min(start))

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
 