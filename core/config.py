from dataclasses import dataclass, field
from typing import Literal, Optional

@dataclass(slots=True)
class PenaltiesMinutes:
    hallucinated_event: float = 10.0
    time_mismatch: float = 10.0
    duplicate_event: float = 10.0
    nonpositive_duration: float = 10.0
    overlap: float = 20.0                # applied per overlap
    out_of_bounds: float = 10.0          # per offending event
    min_gap_violation: float = 10.0      # per violation

@dataclass(slots=True)
class RealismConfig:
    enforce_day_bounds: bool = True
    day_start: str = "00:00"
    day_end: str = "24:00"
    allow_cross_midnight: bool = False
    min_gap_minutes: int = 0             # 0 = no check
    # Hooks for future realism
    enforce_venue_exclusive: bool = False  # requires dataset locations/venues

@dataclass(slots=True)
class EventRubricConfig:
    normalize_with_optimal: Literal["dataset", "dp", "none"] = "dataset"
    strict_times: bool = True
    clip_to_unit: bool = True
    allow_reasoning_tag: bool = True
    penalties: PenaltiesMinutes = field(default_factory=PenaltiesMinutes)
    realism: RealismConfig = field(default_factory=RealismConfig)

@dataclass(slots=True)
class MultiTurnConfig:
    max_turns: int = 3
    feedback_role: Literal["system", "user"] = "user"
    stop_early_on_clean: bool = True
