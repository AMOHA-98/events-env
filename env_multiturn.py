from typing import List, Dict, Any, Union, Literal, Tuple
from verifiers.envs.environment import Environment
from openai import OpenAI
from .parsing import parse_schedule_any
from .conflict_checker import check_conflicts
from .config import MultiTurnConfig

SYSTEM = (
    "You are a scheduling assistant. Given an events list and priority names, "
    "return ONLY a schedule in **JSON or XML**:\n"
    'JSON: {"schedule":[{"name":...,"start":"HH:MM","end":"HH:MM"}, ...]}\n'
    "XML:  <schedule><event><name>..</name><start>HH:MM</start><end>HH:MM</end></event>...</schedule>\n"
    "Use only listed events, exact times, avoid overlaps, honor day bounds. /no_think"
)

class EventSchedulingMultiTurnEnv(Environment):
    """
    Multi-turn: propose -> internal feedback -> revise (no external tool dependency).
    Feedback is injected as a message describing issues.
    """
    def __init__(self, mt_cfg: MultiTurnConfig, message_type: Literal["chat","completion"]="chat", **kwargs):
        super().__init__(message_type=message_type, **kwargs)
        self.message_type = message_type
        self.mt_cfg = mt_cfg

    async def _ask(self, client: OpenAI, model: str, messages, sampling_args):
        return await self.get_model_response(
            client=client,
            model=model,
            prompt=messages,
            sampling_args=sampling_args,
            message_type="chat",
        )

    async def rollout(
        self,
        client: OpenAI,
        model: str,
        prompt: Union[str, List[Dict[str, Any]]],
        answer: str,
        task: str = "default",
        info: Dict[str, Any] = {},
        sampling_args: Dict[str, Any] = {},
        **kwargs: Any,
    ):
        # prompt is messages [system, user]
        messages: List[Dict[str,str]] = list(prompt)  # copy
        all_responses = []

        for turn in range(self.mt_cfg.max_turns):
            resp = await self._ask(client, model, messages, sampling_args)
            all_responses.append(resp)

            text = resp.choices[0].message.content if hasattr(resp, "choices") else str(resp)
            # Try to parse; if parseable, run internal checks and possibly ask for revision
            schedule = parse_schedule_any(text, allow_reasoning_tag=True)
            if schedule is None:
                feedback = (
                    "Your output was not a valid JSON or XML schedule. "
                    "Please output only one of the required formats. "
                    "Do not include explanations."
                )
            else:
                # Build quick feedback from answer->constraints
                import json
                info_obj = json.loads(answer)
                # Pull realism/strict settings from the attached rubric if available
                rubric_cfg = getattr(getattr(self, "rubric", None), "cfg", None)
                strict_times = getattr(rubric_cfg, "strict_times", True)
                realism = getattr(rubric_cfg, "realism", None)
                day_start = getattr(realism, "day_start", "00:00") if realism else "00:00"
                day_end = getattr(realism, "day_end", "24:00") if realism else "24:00"
                allow_cross_midnight = getattr(realism, "allow_cross_midnight", False) if realism else False
                min_gap_minutes = getattr(realism, "min_gap_minutes", 0) if realism else 0

                rep = check_conflicts(
                    proposal=schedule,
                    catalog=info_obj["events"],
                    strict_times=strict_times,
                    day_start=day_start,
                    day_end=day_end,
                    allow_cross_midnight=allow_cross_midnight,
                    min_gap_minutes=min_gap_minutes,
                )
                if rep["summary"] == "No issues found." and self.mt_cfg.stop_early_on_clean:
                    break
                feedback = (
                    f"Validator feedback:\n{rep['summary']}\n"
                    "Revise and return ONLY the schedule in JSON or XML. No commentary."
                )

            messages.append({"role": "assistant", "content": text})
            messages.append({"role": self.mt_cfg.feedback_role, "content": feedback})

        # Return the last assistant message only
        final_text = all_responses[-1].choices[0].message.content if all_responses else ""
        return [{"role": "assistant", "content": final_text}], {"responses": all_responses}
