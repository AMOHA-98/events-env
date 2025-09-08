import json
from verifiers import Rubric
from typing import Any, Dict, List, Union
from ..io.parsing import parse_schedule_any
from .scoring import score_with_penalties, wis_optimum
from ..core.config import EventRubricConfig

class EventSchedulingRubric(Rubric):
    """
    Emits a single scalar reward. If normalized, returns in [0,1].
    `answer` (from dataset mapping) must be a JSON string containing:
      { "events": [[name,start,end],...],
        "priority_events": [name,...],
        "optimal_score": <int or null> }
    """
    def __init__(self, cfg: EventRubricConfig):
        super().__init__(funcs=[], weights=[])
        self.cfg = cfg
        self.add_reward_func(self._reward, weight=1.0)

    def _extract_text(self, completion: Union[str, List[Dict[str,Any]]]) -> str:
        if isinstance(completion, list) and completion and "content" in completion[-1]:
            return completion[-1]["content"]
        return str(completion)

    def _reward(self, completion, answer, **kwargs) -> float:
        text = self._extract_text(completion)
        schedule = parse_schedule_any(text, allow_reasoning_tag=self.cfg.allow_reasoning_tag)
        if schedule is None:
            return 0.0

        info = json.loads(answer)
        events: List[List[str]] = info["events"]
        priority_events: List[str] = info["priority_events"]
        ds_opt = info.get("optimal_score", None)

        minutes, diag = score_with_penalties(
            proposal=schedule,
            events=events,
            priority_events=priority_events,
            strict_times=self.cfg.strict_times,
            penalties=self.cfg.penalties,
            realism=self.cfg.realism,
        )

        # normalization
        if self.cfg.normalize_with_optimal == "none":
            reward = minutes  # raw minutes
        else:
            if self.cfg.normalize_with_optimal == "dataset" and isinstance(ds_opt, (int, float)) and ds_opt > 0:
                denom = float(ds_opt)
            else:
                denom = wis_optimum(events, priority_events, self.cfg.realism)
            reward = minutes / max(1.0, denom)

        if self.cfg.clip_to_unit:
            reward = max(0.0, min(1.0, reward))
        return float(reward)
