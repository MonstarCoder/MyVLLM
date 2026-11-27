"""Synchronous request lifecycle. The caller supplies the execution backend."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import count
from typing import Protocol

from .scheduler import BatchItem, Request, plan_batch


class ModelRunner(Protocol):
    def execute(self, batch: tuple[BatchItem, ...]) -> Mapping[int, int]:
        """Return one next-token ID for each item marked for sampling."""
        ...


@dataclass(frozen=True)
class TokenEvent:
    request_id: int
    token_id: int
    finished: bool


class Engine:
    def __init__(self, runner: ModelRunner, *, token_budget: int, eos_ids: Sequence[int] = ()):
        if token_budget < 1:
            raise ValueError("token_budget must be positive")
        self._release = getattr(runner, "release", None)
        self._runner = runner
        self._token_budget = token_budget
        self._eos_ids = frozenset(eos_ids)
        self._requests: dict[int, Request] = {}
        self._ids = count()

    def add_request(self, prompt: Sequence[int], *, max_new_tokens: int) -> int:
        """Admit a tokenized prompt and return its engine-local ID."""
        tokens = tuple(prompt)
        if not tokens or max_new_tokens < 1:
            raise ValueError("prompt must contain tokens and max_new_tokens must be positive")
        request_id = next(self._ids)
        self._requests[request_id] = Request(request_id, tokens, max_new_tokens)
        return request_id

    @property
    def has_requests(self) -> bool:
        return bool(self._requests)

    def step(self) -> list[TokenEvent]:
        batch = plan_batch(self._requests.values(), self._token_budget)
        if not batch:
            return []
        sampled = dict(self._runner.execute(batch))
        expected = {item.request_id for item in batch if item.sample}
        if sampled.keys() != expected:
            raise ValueError("runner returned token IDs for an unexpected set of requests")
        events = []
        for item in batch:
            request = self._requests[item.request_id]
            request.computed += len(item.token_ids)
            if item.sample:
                token = sampled[item.request_id]
                request.last_token = token
                request.generated += 1
                finished = request.generated == request.max_new_tokens or token in self._eos_ids
                events.append(TokenEvent(item.request_id, token, finished))
                if finished:
                    if self._release is not None:
                        self._release(item.request_id)
                    del self._requests[item.request_id]
        return events
