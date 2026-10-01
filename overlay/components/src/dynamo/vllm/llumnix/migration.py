from __future__ import annotations
import time
from collections.abc import Mapping
from dataclasses import dataclass
from .config import LlumnixConfig
from .types import InstanceStatus


@dataclass(frozen=True, slots=True)
class MigrationPair:
    source: InstanceStatus
    target: InstanceStatus
    source_load: float
    target_load: float


def _effective_projected_load(
    status: InstanceStatus,
    reserved_tokens_by_worker: Mapping[int, int],
) -> float:
    if status.num_total_gpu_tokens <= 0:
        return float("inf")
    reserved_tokens = max(0, int(reserved_tokens_by_worker.get(status.worker_id, 0)))
    return (status.projected_tokens + reserved_tokens) / status.num_total_gpu_tokens


def choose_migration_pairs(
    statuses: list[InstanceStatus],
    config: LlumnixConfig,
    *,
    now_ms: int | None = None,
    filter_stale: bool = True,
    reserved_tokens_by_worker: Mapping[int, int] | None = None,
    reserved_migrate_in_by_worker: Mapping[int, int] | None = None,
) -> list[MigrationPair]:


    now_ms = int(time.time() * 1000) if now_ms is None else now_ms
    reserved_tokens_by_worker = reserved_tokens_by_worker or {}
    reserved_migrate_in_by_worker = reserved_migrate_in_by_worker or {}
    fresh = [
        status
        for status in statuses
        if status.schedulable
        and (not filter_stale or not status.is_stale(config.status_stale_ms, now_ms))
    ]
    loads = {
        status.worker_id: _effective_projected_load(status, reserved_tokens_by_worker)
        for status in fresh
    }
    sources = [
        status
        for status in fresh
        if loads[status.worker_id] >= config.overload_threshold
        and status.num_migrate_out_reqs == 0
        and any(request.migratable for request in status.requests)
    ]
    targets = [
        status
        for status in fresh
        if loads[status.worker_id] < config.underload_threshold
        and (
            status.num_migrate_in_reqs
            + max(
                0,
                int(reserved_migrate_in_by_worker.get(status.worker_id, 0)),
            )
            < config.max_migrate_in_requests
        )
    ]
    if not sources or not targets:
        return []

    sources.sort(key=lambda status: (-loads[status.worker_id], status.worker_id))
    targets.sort(key=lambda status: (loads[status.worker_id], status.worker_id))
    pairs: list[MigrationPair] = []
    for source, target in zip(sources, targets):
        source_load = loads[source.worker_id]
        target_load = loads[target.worker_id]
        if source_load - target_load < config.load_balance_threshold:
            break
        pairs.append(
            MigrationPair(
                source=source,
                target=target,
                source_load=source_load,
                target_load=target_load,
            )
        )
    return pairs
