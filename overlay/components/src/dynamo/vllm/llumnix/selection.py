from __future__ import annotations

from collections.abc import Iterable

from .types import MigrationLimit, MigrationPolicy, RequestPhase, RequestSnapshot


def _cost(request: RequestSnapshot, limit: MigrationLimit) -> int:
    if limit in {MigrationLimit.TOKEN, MigrationLimit.RATIO}:
        return request.token_cost
    if limit == MigrationLimit.NUM_REQ:
        return 1
    raise NotImplementedError(limit)


def _budget(limit: MigrationLimit, value: float, total_gpu_tokens: int) -> int:
    if limit == MigrationLimit.RATIO:
        return int(total_gpu_tokens * value)
    return int(value)


def _ordered_running(
    requests: list[RequestSnapshot], policy: MigrationPolicy
) -> list[RequestSnapshot]:
    if policy in {MigrationPolicy.SR, MigrationPolicy.FCWSR}:
        return sorted(requests, key=lambda request: request.token_cost)
    if policy == MigrationPolicy.LR:
        return sorted(requests, key=lambda request: request.token_cost, reverse=True)
    if policy == MigrationPolicy.LCR:
        return list(reversed(requests))
    return requests


def _take(
    candidates: Iterable[RequestSnapshot],
    budget: int,
    limit: MigrationLimit,
    *,
    waiting: bool,
) -> tuple[list[RequestSnapshot], int]:
    selected: list[RequestSnapshot] = []
    accumulated = 0
    for request in candidates:
        if accumulated >= budget:
            break
        if not request.migratable:
            continue
        if not waiting and request.output_tokens < 1:
            continue
        request_cost = _cost(request, limit)


        if request_cost > budget:
            continue
        selected.append(request)
        accumulated += request_cost
    return selected, max(0, budget - accumulated)


def select_requests(
    requests: Iterable[RequestSnapshot],
    policy: MigrationPolicy,
    limit: MigrationLimit,
    value: float,
    total_gpu_tokens: int,
    *,
    excluded_request_ids: Iterable[str] = (),
) -> list[RequestSnapshot]:


    excluded = {str(request_id) for request_id in excluded_request_ids}
    requests = [request for request in requests if request.request_id not in excluded]
    running = [r for r in requests if r.phase == RequestPhase.RUNNING]
    waiting = [r for r in requests if r.phase == RequestPhase.WAITING]


    if limit == MigrationLimit.NUM_REQ and int(value) == -1:
        return [
            r
            for r in requests
            if r.migratable and r.phase in {RequestPhase.RUNNING, RequestPhase.WAITING}
        ]

    budget = _budget(limit, value, total_gpu_tokens)
    if budget <= 0:
        return []

    if policy in {MigrationPolicy.FCW, MigrationPolicy.FCWSR}:
        selected, remaining = _take(waiting, budget, limit, waiting=True)
        if policy == MigrationPolicy.FCWSR and remaining > 0:
            more, _ = _take(
                _ordered_running(running, policy), remaining, limit, waiting=False
            )
            selected.extend(more)
        return selected

    selected, _ = _take(_ordered_running(running, policy), budget, limit, waiting=False)
    return selected
