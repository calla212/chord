from __future__ import annotations


import os as _ttft_diag_os
if _ttft_diag_os.environ.get("DYN_TTFT_DIAG_ENABLE") == "1":
    import ttft_diag as _ttft_diag
else:
    _ttft_diag = None


import asyncio
import heapq
import json
import math
import queue
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..llumnix.types import InstanceStatus
from ..llumnix.metrics import EffectiveMetrics, calculate_effective_metrics
from .config import TRACE_FLUSH_INTERVAL_S
from .types import (
    ChordRequestPhase,
    ChordRequestState,
    ChordWorkerSnapshot,
    WaitingRequestSnapshot,
)

TRACE_END = object()
TRACE_CLOSE_TIMEOUT_S = 5.0


@dataclass(frozen=True, slots=True)
class DispatchLease:


    request_id: str
    backend_request_id: str
    epoch: int
    worker_id: int
    metadata: dict[str, Any]


@dataclass(slots=True)
class RequestRecord:


    request_id: str
    arrival_time: float
    prompt_tokens: int
    state: ChordRequestState = ChordRequestState.GLOBAL_WAITING
    epoch: int = 0
    next_epoch: int = 1
    compute_worker_id: int | None = None
    stream_owner_worker_id: int | None = None
    resume_output_token_ids: list[int] = field(default_factory=list)
    last_progress_time: float = 0.0
    num_preemptions: int = 0
    was_preempted: bool = False
    last_return_worker_id: int | None = None
    pool_version: int = 0
    forbidden_epochs: dict[int, set[int]] = field(default_factory=dict)
    assignment_waiter: asyncio.Future[DispatchLease] | None = field(
        default=None, repr=False
    )

    @property
    def context_tokens(self) -> int:
        return self.prompt_tokens + len(self.resume_output_token_ids)

    def block_demand(
        self, block_size: int, *, output_tokens_hint: int | None = None
    ) -> int:
        output_tokens = len(self.resume_output_token_ids)
        if output_tokens_hint is not None:
            output_tokens = max(output_tokens, int(output_tokens_hint))
        return max(
            1,
            math.ceil((self.prompt_tokens + output_tokens) / block_size),
        )

    def snapshot(self, block_size: int) -> WaitingRequestSnapshot:
        return WaitingRequestSnapshot(
            request_id=self.request_id,
            owner_worker_id=-1,
            version=self.epoch,
            phase=(
                ChordRequestPhase.PREEMPTED
                if self.was_preempted
                else ChordRequestPhase.NEVER_RUN
            ),
            prompt_tokens=self.prompt_tokens,
            output_tokens=len(self.resume_output_token_ids),
            num_preemptions=self.num_preemptions,
            block_demand=self.block_demand(block_size),
            arrival_time=self.arrival_time,
            last_progress_time=self.last_progress_time,
        )


@dataclass(frozen=True, slots=True)
class Reservation:
    request_id: str
    epoch: int
    blocks: int
    created_at: float
    context_tokens: int | None = None
    baseline_step_id: int = -1


@dataclass(slots=True)
class InstanceState:
    worker_id: int
    control_url: str
    connected: bool = False
    latest_round: int = -1
    compact: ChordWorkerSnapshot | None = None
    detailed: InstanceStatus | None = None
    detail_round: int = -1
    unobserved_dispatches: dict[tuple[str, int], Reservation] = field(
        default_factory=dict
    )
    migration_reservations: dict[tuple[str, int], Reservation] = field(
        default_factory=dict
    )
    forbidden_dispatches: set[tuple[str, int]] = field(default_factory=set)

    def active_migration_reservations(
        self,
        *,
        now: float,
        timeout_s: float,
    ) -> tuple[Reservation, ...]:
        return tuple(
            reservation
            for reservation in self.migration_reservations.values()
            if now - reservation.created_at < timeout_s
        )

    def accounting_overlay(
        self, *, now: float, migration_timeout_s: float, block_size: int
    ) -> dict[str, int]:
        migrations = self.active_migration_reservations(
            now=now, timeout_s=migration_timeout_s
        )
        return {
            "dispatch_requests": len(self.unobserved_dispatches),
            "dispatch_tokens": sum(
                item.context_tokens
                if item.context_tokens is not None
                else item.blocks * block_size
                for item in self.unobserved_dispatches.values()
            ),
            "migration_requests": len(migrations),
            "migration_tokens": sum(item.blocks * block_size for item in migrations),
        }

    def effective_metrics(
        self, *, now: float, migration_timeout_s: float, sample: Any | None = None
    ) -> EffectiveMetrics:
        sample = self.compact if sample is None else sample
        if sample is None:
            raise RuntimeError("worker has no load sample")
        return calculate_effective_metrics(
            sample,
            **self.accounting_overlay(
                now=now,
                migration_timeout_s=migration_timeout_s,
                block_size=sample.block_size,
            ),
        )

    def planning_snapshot(
        self,
        *,
        now: float | None = None,
        migration_reservation_timeout_s: float | None = None,
    ) -> ChordWorkerSnapshot | None:
        if not self.connected or self.compact is None or not self.compact.schedulable:
            return None
        if migration_reservation_timeout_s is None:
            migration_reservations = tuple(self.migration_reservations.values())
        else:
            migration_reservations = self.active_migration_reservations(
                now=time.monotonic() if now is None else now,
                timeout_s=migration_reservation_timeout_s,
            )
        return self.compact.with_reservations(
            dispatch_count=len(self.unobserved_dispatches),
            dispatch_blocks=sum(
                reservation.blocks
                for reservation in self.unobserved_dispatches.values()
            ),
            migration_blocks=sum(
                reservation.blocks for reservation in migration_reservations
            ),
        )


@dataclass(slots=True)
class CoordinatorEvent:
    kind: str
    data: dict[str, Any] = field(default_factory=dict)
    future: asyncio.Future[Any] | None = None


class GlobalPool:


    def __init__(self, *, output_priority: bool = True) -> None:
        self._output_priority = output_priority
        self._members: dict[str, int] = {}
        self._never_run: list[tuple[float, int, str, int]] = []
        self._preempted: list[tuple[float, int, str, int]] = []
        self._sequence = 0

    def __bool__(self) -> bool:
        return bool(self._members)

    def __len__(self) -> int:
        return len(self._members)

    def add(self, record: RequestRecord) -> None:
        decode = bool(record.resume_output_token_ids) if self._output_priority else record.was_preempted
        record.pool_version += 1
        version = record.pool_version
        self._members[record.request_id] = version
        self._sequence += 1
        entry = (
            record.last_progress_time if decode else record.arrival_time,
            self._sequence,
            record.request_id,
            version,
        )
        heapq.heappush(
            self._preempted if decode else self._never_run, entry
        )

    def discard(self, request_id: str) -> None:
        self._members.pop(request_id, None)

    def _valid(
        self,
        entry: tuple[float, int, str, int],
        records: dict[str, RequestRecord],
    ) -> bool:
        _, _, request_id, version = entry
        record = records.get(request_id)
        return (
            self._members.get(request_id) == version
            and record is not None
            and record.state == ChordRequestState.GLOBAL_WAITING
        )

    def _clean(
        self,
        heap: list[tuple[float, int, str, int]],
        records: dict[str, RequestRecord],
    ) -> None:
        while heap and not self._valid(heap[0], records):
            heapq.heappop(heap)

    def candidates(
        self,
        records: dict[str, RequestRecord],
        *,
        now: float,
        coefficient: float,
        limit: int,
    ) -> list[RequestRecord]:
        chosen: list[RequestRecord] = []
        removed_never: list[tuple[float, int, str, int]] = []
        removed_preempted: list[tuple[float, int, str, int]] = []
        while len(chosen) < limit:
            self._clean(self._never_run, records)
            self._clean(self._preempted, records)
            never = self._never_run[0] if self._never_run else None
            preempted = self._preempted[0] if self._preempted else None
            if never is None and preempted is None:
                break

            take_preempted = False
            if preempted is not None:
                if never is None:
                    take_preempted = True
                else:
                    never_record = records[never[2]]
                    preempted_record = records[preempted[2]]
                    never_urgency = max(0.0, now - never_record.arrival_time)
                    preempted_urgency = coefficient * max(
                        0.0, now - preempted_record.last_progress_time
                    )
                    take_preempted = preempted_urgency > never_urgency or (
                        preempted_urgency == never_urgency
                        and preempted_record.arrival_time < never_record.arrival_time
                    )

            if take_preempted:
                entry = heapq.heappop(self._preempted)
                removed_preempted.append(entry)
            else:
                entry = heapq.heappop(self._never_run)
                removed_never.append(entry)
            record = records.get(entry[2])
            if record is not None and self._valid(entry, records):
                chosen.append(record)

        for entry in removed_never:
            heapq.heappush(self._never_run, entry)
        for entry in removed_preempted:
            heapq.heappush(self._preempted, entry)
        return chosen


class AsyncTrace:


    def __init__(self, path: str, *, maxsize: int = 8192) -> None:
        self._path = path
        self._queue: queue.Queue[Any] | None = None
        self._thread: threading.Thread | None = None
        self.dropped = 0
        if path:
            self._queue = queue.Queue(maxsize=maxsize)
            self._thread = threading.Thread(
                target=self._run, name="chord-trace", daemon=True
            )
            self._thread.start()

    def emit(self, event: str, **fields: Any) -> None:
        if self._queue is None:
            return
        value = {"time": time.time(), "event": event, **fields}
        if _ttft_diag is not None:
            value["monotonic_ns"] = _ttft_diag.now_ns()
        try:
            self._queue.put_nowait(value)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        assert self._queue is not None
        target = Path(self._path)
        target.parent.mkdir(parents=True, exist_ok=True)
        batch: list[dict[str, Any]] = []
        stopping = False
        with target.open("a", encoding="utf-8") as handle:
            while not stopping:
                deadline = time.monotonic() + TRACE_FLUSH_INTERVAL_S
                while time.monotonic() < deadline:
                    try:
                        item = self._queue.get(
                            timeout=max(0.001, deadline - time.monotonic())
                        )
                    except queue.Empty:
                        break
                    if item is TRACE_END:
                        stopping = True
                        break
                    batch.append(item)
                if batch:
                    handle.writelines(
                        json.dumps(item, sort_keys=True) + "\n" for item in batch
                    )
                    handle.flush()
                    batch.clear()

    def close(self) -> None:
        if self._queue is None or self._thread is None:
            return
        try:
            self._queue.put(TRACE_END, timeout=TRACE_CLOSE_TIMEOUT_S)
        except queue.Full as error:
            raise RuntimeError(
                "Chord trace queue did not drain during close"
            ) from error
        self._thread.join(timeout=TRACE_CLOSE_TIMEOUT_S)
        if self._thread.is_alive():
            raise RuntimeError("Chord trace writer did not stop during close")
        self._queue = None
        self._thread = None
