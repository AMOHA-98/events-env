from typing import List, Dict, Any, Union, Literal, Tuple
from verifiers.envs.environment import Environment
from openai import OpenAI

SYSTEM = (
    "You are a scheduling assistant. Given an events list and priority names, "
    "return ONLY a schedule in **JSON or XML**:\n"
    'JSON: {"schedule":[{"name":...,"start":"HH:MM","end":"HH:MM"}, ...]}\n'
    "XML:  <schedule><event><name>..</name><start>HH:MM</start><end>HH:MM</end></event>...</schedule>\n"
    "Use only listed events, exact times (unless instructed otherwise), avoid overlaps, honor day bounds. /no_think"
)

class EventSchedulingEnv(Environment):
    """Single-turn chat environment (baseline)."""
    def __init__(self, message_type: Literal["chat","completion"]="chat", **kwargs):
        super().__init__(message_type=message_type, **kwargs)
        self.message_type = message_type

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
    ) -> Tuple[Union[str, List[Dict[str, str]]], Dict[str, Any]]:
        completion = await self.get_model_response(
            client=client,
            model=model,
            prompt=prompt,
            sampling_args=sampling_args,
            message_type=self.message_type,
        )

        if hasattr(completion, "choices") and len(completion.choices) > 0:
            completion_text = completion.choices[0].message.content
            state = {"responses": [completion]}
        elif isinstance(completion, str):
            completion_text = completion
            state = {"responses": [completion]}
        else:
            completion_text = str(completion)
            state = {"responses": [completion]}

        if self.message_type == "chat":
            messages = [{"role": "assistant", "content": completion_text}]
            return messages, state
        else:
            return completion_text, state
