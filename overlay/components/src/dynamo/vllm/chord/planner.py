from __future__ import annotations

from collections.abc import Collection, Mapping
from dataclasses import dataclass

from .config import TBF_PRECISION
from .types import ChordAssignment, ChordWorkerSnapshot, WaitingRequestSnapshot


@dataclass(frozen=True, slots=True)
class ChordPlan:
    assignments: tuple[ChordAssignment, ...]
    resource_limit: float
    all_placed: bool
    stopped_request_id: str | None = None


@dataclass(slots=True)
class _CandidateCapacity:
    free_blocks: float
    used_blocks: float


def _ordered_requests(
    requests: list[WaitingRequestSnapshot], now: float, coefficient: float, output_priority: bool
) -> list[WaitingRequestSnapshot]:
    return sorted(
        requests,
        key=lambda request: (
            -request.urgency(now, coefficient, output_priority=output_priority),
            request.arrival_time,
            request.request_id,
        ),
    )


def _limited_candidates(
    workers: list[ChordWorkerSnapshot], resource_limit: float
) -> dict[int, _CandidateCapacity]:
    candidates: dict[int, _CandidateCapacity] = {}
    for worker in sorted(workers, key=lambda item: item.worker_id):
        used = float(worker.effective_projected_blocks)
        virtual_free = resource_limit * float(worker.total_blocks) - used
        candidates[worker.worker_id] = _CandidateCapacity(
            free_blocks=max(0.0, virtual_free),
            used_blocks=used,
        )
    return candidates


def _best_fit_plan(
    requests: list[WaitingRequestSnapshot],
    candidates: dict[int, _CandidateCapacity],
    *,
    max_plan_requests: int,
    skip_unfit: bool,
    now: float,
    coefficient: float,
    dispatch_threshold_s: float,
    output_priority: bool,
    forbidden_targets: Mapping[str, Collection[int]] | None = None,
) -> ChordPlan:
    assignments: list[ChordAssignment] = []
    all_placed = True
    stopped_request_id: str | None = None

    for request in requests:
        if len(assignments) >= max_plan_requests:
            all_placed = False
            stopped_request_id = request.request_id
            break
        available = [
            worker_id
            for worker_id, capacity in candidates.items()
            if capacity.free_blocks > 0
        ]
        if not available:
            all_placed = False
            stopped_request_id = request.request_id
            break

        max_used = max(candidates[worker_id].used_blocks for worker_id in available)
        forbidden = (
            forbidden_targets.get(request.request_id, ())
            if forbidden_targets is not None
            else ()
        )
        fitting = [
            worker_id
            for worker_id in available
            if candidates[worker_id].free_blocks - request.block_demand > 0
            and worker_id not in forbidden
        ]
        if not fitting:
            all_placed = False
            stopped_request_id = request.request_id
            if skip_unfit or request.urgency(now, coefficient, output_priority=output_priority) < dispatch_threshold_s:
                continue
            break

        below_max = [
            worker_id
            for worker_id in fitting
            if max_used - candidates[worker_id].used_blocks - request.block_demand > 0
        ]
        if below_max:
            target = min(
                below_max,
                key=lambda worker_id: (
                    max_used
                    - (candidates[worker_id].used_blocks + request.block_demand),
                    worker_id,
                ),
            )
        else:
            target = min(
                fitting,
                key=lambda worker_id: (
                    candidates[worker_id].used_blocks + request.block_demand - max_used,
                    worker_id,
                ),
            )

        assignments.append(ChordAssignment(request=request, target_worker_id=target))
        candidates[target].free_blocks -= request.block_demand
        candidates[target].used_blocks += request.block_demand

    return ChordPlan(
        assignments=tuple(assignments),
        resource_limit=1.0,
        all_placed=all_placed,
        stopped_request_id=stopped_request_id,
    )


def derive_redispatching_plan(
    requests: list[WaitingRequestSnapshot],
    workers: list[ChordWorkerSnapshot],
    *,
    now: float,
    urgency_coefficient: float = 3.0,
    max_t: float = 1.0,
    max_plan_requests: int = 256,
    skip_unfit: bool = False,
    dispatch_threshold_s: float = 0.0,
    output_priority: bool = True,
    forbidden_targets: Mapping[str, Collection[int]] | None = None,
) -> ChordPlan:


    skip_unfit = skip_unfit or dispatch_threshold_s == -1
    if not requests or not workers:
        return ChordPlan((), max_t, not requests)
    ordered = _ordered_requests(requests, now, urgency_coefficient, output_priority)[:max_plan_requests]
    right_plan = _best_fit_plan(
        ordered,
        _limited_candidates(workers, max_t),
        max_plan_requests=max_plan_requests,
        skip_unfit=skip_unfit,
        now=now,
        coefficient=urgency_coefficient,
        dispatch_threshold_s=dispatch_threshold_s,
        output_priority=output_priority,
        forbidden_targets=forbidden_targets,
    )
    resource_limit = max_t
    if right_plan.all_placed:
        left, right = 0.0, max_t
        while right - left > TBF_PRECISION:
            middle = (left + right) / 2
            candidate = _best_fit_plan(
                ordered,
                _limited_candidates(workers, middle),
                max_plan_requests=max_plan_requests,
                skip_unfit=skip_unfit,
                now=now,
                coefficient=urgency_coefficient,
                dispatch_threshold_s=dispatch_threshold_s,
                output_priority=output_priority,
                forbidden_targets=forbidden_targets,
            )
            if candidate.all_placed:
                right = middle
                right_plan = candidate
            else:
                left = middle
        resource_limit = right

    return ChordPlan(
        assignments=right_plan.assignments,
        resource_limit=resource_limit,
        all_placed=right_plan.all_placed,
        stopped_request_id=right_plan.stopped_request_id,
    )
