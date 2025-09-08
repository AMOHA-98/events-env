## events-env

Reinforcement-learning style evaluation environments for event scheduling with LLMs. It includes:

- **Single-turn env** `EventSchedulingEnv`: prompts the model once and scores the returned schedule.
- **Multi-turn env** `EventSchedulingMultiTurnEnv`: iterative propose-and-revise with automatic validator feedback.
- **Rubric** `EventSchedulingRubric`: converts a model completion into a scalar reward with realism-aware penalties.

### Quickstart (Windows PowerShell)

```powershell
# Create and sync a uv virtualenv
uv venv
uv sync
.\.venv\Scripts\Activate.ps1
```

This project depends on `datasets` and `openai`. It also uses `verifiers` framework providing `Environment` and `Rubric` interfaces. Ensure it is installed and on `PYTHONPATH`.

### Dataset

By default, loaders use the Hugging Face dataset `anakin87/events-scheduling`. Each example contains:

- `events`: `[[name, start, end], ...]`
- `priority_events`: `[name, ...]`
- `prompt`: A human-readable listing used for the user message
- `optimal_score` (optional): best known weighted minutes for normalization

### Loading environments

Use `loader.load_environment` for single-turn or `loader.load_environment_multiturn` for multi-turn. Both accept convenience args for dataset sizing and realism.

```python
from events_env.io.loader import load_environment, load_environment_multiturn

# Single-turn
env = load_environment(
    num_train_examples=500,     # -1 for all
    num_eval_examples=100,
    normalize_with_optimal="dataset",  # "dataset" | "dp" | "none"
    strict=True,
    allow_reasoning_tag=True,
    realism_min_gap=0,
    realism_enforce_bounds=True,
)

# Multi-turn
mt_env = load_environment_multiturn(
    num_train_examples=500,
    num_eval_examples=100,
    normalize_with_optimal="dataset",
    strict=True,
    allow_reasoning_tag=True,
    realism_min_gap=0,
    realism_enforce_bounds=True,
    max_turns=3,
)
```

Both envs produce prompts of the form `[{role: system}, {role: user}]` and expect the model to answer with ONLY JSON or XML schedule formats.

### Schedule formats

- JSON: `{ "schedule": [{"name": "...", "start": "HH:MM", "end": "HH:MM"}, ...] }`
- XML: `<schedule><event><name>..</name><start>HH:MM</start><end>HH:MM</end></event>...</schedule>`

The parser strips code fences and an optional leading `<think>...</think>` block.

### Scoring (Rubric)

`EventSchedulingRubric` computes a reward from 0 to 1 (clipped) when `normalize_with_optimal != "none"`.

Base utility = sum of minutes, double-weighted for events in `priority_events`.

Penalties (in minutes):

- Hallucinated event
- Time mismatch (under `strict=True`)
- Duplicate event
- Nonpositive duration
- Overlaps (per overlap)
- Out of bounds (if day-bounds enforced)
- Min-gap violations (if `min_gap_minutes > 0`)

Normalization denominator:

- `dataset`: use dataset-provided `optimal_score` when available; otherwise fallback to DP optimum
- `dp`: always compute a dynamic-programming weighted-interval optimum
- `none`: use raw minutes (no clipping unless `clip_to_unit=True` is disabled in code)

### Realism controls

Configured via `EventRubricConfig.realism`:

- `enforce_day_bounds` with `day_start`, `day_end`
- `allow_cross_midnight`
- `min_gap_minutes`

These also drive the multi-turn validator feedback inside `EventSchedulingMultiTurnEnv`.

### Example rollout loop (pseudo)

```python
from openai import OpenAI

client = OpenAI()
model = "gpt-4o-mini"
env = load_environment(num_train_examples=100, num_eval_examples=20)

for ex in env.dataset:
    messages = ex["prompt"]
    # env.get_model_response abstracts chat/completion API differences
    completion, state = await env.rollout(client, model, messages, ex["answer"])  # returns assistant messages
    reward = env.rubric(completion, ex["answer"])  # scalar
```

For multi-turn, the environment injects validator feedback each turn until a clean schedule or `max_turns` is reached.

### Troubleshooting

- Ensure `verifiers` base package is installed and importable (providing `Environment` and `Rubric`).
- If completions include commentary, enable `allow_reasoning_tag` so `<think>...</think>` blocks are stripped, and remind the model to output ONLY JSON or XML.
- If rewards are always 0, check that outputs parse as valid JSON/XML and that event names exactly match the dataset.
- If normalization seems off, switch `normalize_with_optimal` to `dp` to compute a consistent denominator.


