import json
from typing import Literal
from datasets import load_dataset
from ..evals.rubric import EventSchedulingRubric
from ..core.config import EventRubricConfig, MultiTurnConfig
from ..core.env_singleturn import EventSchedulingEnv, SYSTEM as SYSTEM_SINGLE
from ..core.env_multiturn import EventSchedulingMultiTurnEnv, SYSTEM as SYSTEM_MULTI

def _map_example(ex, system_prompt: str):
    # Dataset fields expected:
    # ex["events"]: [[name,start,end], ...]
    # ex["priority_events"]: [name,...]
    # ex["optimal_score"]: int (weighted minutes), may be None
    # ex["prompt"]: human-readable listing
    user_prompt = ex["prompt"]
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    ex["prompt"] = messages
    ex["answer"] = json.dumps({
        "events": ex["events"],
        "priority_events": ex["priority_events"],
        "optimal_score": ex.get("optimal_score", None),
    })
    return ex

def _holdout(train_split, desired_test: int = 100, seed: int = 42):
    sz = len(train_split)
    test_size = max(1, min(desired_test, max(1, sz // 10)))
    split = train_split.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    return split["train"], split["test"]

def _maybe_select(ds, n: int):
    return ds if n == -1 else ds.select(range(min(n, len(ds))))

def load_environment(
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    normalize_with_optimal: Literal["dataset","dp","none"]="dataset",
    strict: bool = True,
    allow_reasoning_tag: bool = True,
    realism_min_gap: int = 0,
    realism_enforce_bounds: bool = True,
):
    ds = load_dataset("anakin87/events-scheduling")
    train_split = ds.get("train") or ds[list(ds.keys())[0]]
    eval_split = ds.get("test")
    if eval_split is None:
        train_split, eval_split = _holdout(train_split)

    train_split = train_split.map(lambda ex: _map_example(ex, SYSTEM_SINGLE))
    eval_split  = eval_split.map(lambda ex: _map_example(ex, SYSTEM_SINGLE))

    train_split = _maybe_select(train_split, num_train_examples)
    eval_split  = _maybe_select(eval_split,  num_eval_examples)

    from .config import EventRubricConfig, RealismConfig, PenaltiesMinutes
    cfg = EventRubricConfig(
        normalize_with_optimal=normalize_with_optimal,
        strict_times=strict,
        allow_reasoning_tag=allow_reasoning_tag,
    )
    # realism toggles from args
    cfg.realism.min_gap_minutes = int(realism_min_gap)
    cfg.realism.enforce_day_bounds = bool(realism_enforce_bounds)

    rubric = EventSchedulingRubric(cfg)

    env = EventSchedulingEnv(
        dataset=train_split,
        eval_dataset=eval_split,
        parser=None,
        system_prompt=SYSTEM_SINGLE,
        rubric=rubric,
        message_type="chat",
    )
    return env

def load_environment_multiturn(
    num_train_examples: int = -1,
    num_eval_examples: int = -1,
    normalize_with_optimal: Literal["dataset","dp","none"]="dataset",
    strict: bool = True,
    allow_reasoning_tag: bool = True,
    realism_min_gap: int = 0,
    realism_enforce_bounds: bool = True,
    max_turns: int = 3,
):
    ds = load_dataset("anakin87/events-scheduling")
    train_split = ds.get("train") or ds[list(ds.keys())[0]]
    eval_split = ds.get("test")
    if eval_split is None:
        train_split, eval_split = _holdout(train_split)

    train_split = train_split.map(lambda ex: _map_example(ex, SYSTEM_MULTI))
    eval_split  = eval_split.map(lambda ex: _map_example(ex, SYSTEM_MULTI))

    train_split = _maybe_select(train_split, num_train_examples)
    eval_split  = _maybe_select(eval_split,  num_eval_examples)

    from .config import EventRubricConfig
    cfg = EventRubricConfig(
        normalize_with_optimal=normalize_with_optimal,
        strict_times=strict,
        allow_reasoning_tag=allow_reasoning_tag,
    )
    rubric = EventSchedulingRubric(cfg)

    env = EventSchedulingMultiTurnEnv(
        dataset=train_split,
        eval_dataset=eval_split,
        parser=None,
        system_prompt=SYSTEM_MULTI,
        rubric=rubric,
        max_turns=max_turns,
        message_type="chat",
        feedback_role="user",
        stop_early_on_clean=True,
    )
    return env
