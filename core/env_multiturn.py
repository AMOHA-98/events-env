from typing import List, Dict, Any, Union, Literal, Tuple
from verifiers.envs.multiturn import MultiTurnEnv
from ..io.parsing import parse_schedule_any
from ..evals.conflict_checker import check_conflicts

SYSTEM = (
    "You are a scheduling assistant. Given an events list and priority names, "
    "return ONLY a schedule in **JSON or XML**:\n"
    'JSON: {"schedule":[{"name":...,"start":"HH:MM","end":"HH:MM"}, ...]}\n'
    "XML:  <schedule><event><name>..</name><start>HH:MM</start><end>HH:MM</end></event>...</schedule>\n"
    "Use only listed events, exact times, avoid overlaps, honor day bounds. /no_think"
)

class EventSchedulingMultiTurnEnv(MultiTurnEnv):
    """Propose -> validator feedback -> revise, built on verifiers.MultiTurnEnv."""
    def __init__(
        self,
        *,
        max_turns: int = 3,
        feedback_role: Literal["system","user"] = "user",
        stop_early_on_clean: bool = True,
        **kwargs: Any,
    ):
        super().__init__(max_turns=max_turns, **kwargs)
        self.feedback_role = feedback_role
        self.stop_early_on_clean = stop_early_on_clean

    async def env_response(self, messages: List[Dict[str,str]], state: Dict[str,Any], **kwargs: Any):
        # Extract last assistant output
        last_text = ""
        if messages and messages[-1].get("role") == "assistant":
            last_text = messages[-1].get("content", "")

        # Track turn
        state = dict(state or {})
        state["turn"] = int(state.get("turn", 0)) + 1

        # Attempt to parse schedule
        schedule = parse_schedule_any(last_text, allow_reasoning_tag=True)
        if schedule is None:
            feedback = (
                "Your output was not a valid JSON or XML schedule. "
                "Please output only one of the required formats. "
                "Do not include explanations."
            )
            return [{"role": self.feedback_role, "content": feedback}], state

        # Build validator feedback with rubric settings
        import json
        answer = kwargs.get("answer", None)
        try:
            info_obj = json.loads(answer) if isinstance(answer, str) else (answer or {})
        except Exception:
            info_obj = {}

        rubric_cfg = getattr(getattr(self, "rubric", None), "cfg", None)
        strict_times = getattr(rubric_cfg, "strict_times", True)
        realism = getattr(rubric_cfg, "realism", None)
        day_start = getattr(realism, "day_start", "00:00") if realism else "00:00"
        day_end = getattr(realism, "day_end", "24:00") if realism else "24:00"
        allow_cross_midnight = getattr(realism, "allow_cross_midnight", False) if realism else False
        min_gap_minutes = getattr(realism, "min_gap_minutes", 0) if realism else 0

        rep = check_conflicts(
            proposal=schedule,
            catalog=info_obj.get("events", []),
            strict_times=strict_times,
            day_start=day_start,
            day_end=day_end,
            allow_cross_midnight=allow_cross_midnight,
            min_gap_minutes=min_gap_minutes,
        )
        state["validator_report"] = rep
        state["normalized_schedule"] = rep.get("normalized")

        if rep["summary"] == "No issues found.":
            if self.stop_early_on_clean:
                # Empty env message signals completion for many trainers, but we still return a gentle ack
                return [{"role": self.feedback_role, "content": "Looks good."}], state

        feedback = (
            f"Validator feedback:\n{rep['summary']}\n"
            "Revise and return ONLY the schedule in JSON or XML. No commentary."
        )
        return [{"role": self.feedback_role, "content": feedback}], state

    async def is_completed(self, messages: List[Dict[str,str]], state: Dict[str,Any], **kwargs: Any) -> bool:
        # Early stop on clean report
        rep = (state or {}).get("validator_report", {})
        if self.stop_early_on_clean and rep.get("summary") == "No issues found.":
            return True
        # Or stop when reaching max turns
        return int((state or {}).get("turn", 0)) >= int(getattr(self, "max_turns", 3))
