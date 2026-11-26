"""Build a single-phase batch within a per-step token budget."""

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum


class Phase(Enum):
    PREFILL = "prefill"
    DECODE = "decode"


@dataclass
class Request:
    request_id: int
    prompt: tuple[int, ...]
    max_new_tokens: int
    computed: int = 0
    generated: int = 0
    last_token: int | None = None


@dataclass(frozen=True)
class BatchItem:
    request_id: int
    phase: Phase
    position: int
    token_ids: tuple[int, ...]
    sample: bool


def plan_batch(requests: Iterable[Request], token_budget: int) -> tuple[BatchItem, ...]:
    """Follow admission order and use the oldest request to choose the phase."""
    batch: list[BatchItem] = []
    for request in requests:
        phase = Phase.PREFILL if request.computed < len(request.prompt) else Phase.DECODE
        if batch and phase != batch[0].phase:
            continue
        start = request.computed
        if phase == Phase.PREFILL:
            tokens = request.prompt[start : start + token_budget]
            sample = start + len(tokens) == len(request.prompt)
        else:
            assert request.last_token is not None
            tokens = (request.last_token,)
            sample = True
        batch.append(BatchItem(request.request_id, phase, start, tokens, sample))
        token_budget -= len(tokens)
        if token_budget == 0:
            break
    return tuple(batch)
