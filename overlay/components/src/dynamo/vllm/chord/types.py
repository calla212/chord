from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, replace
from enum import Enum
from typing import Any


class ChordRequestPhase(str, Enum):
    NEVER_RUN = "never_run"
    PREEMPTED = "preempted"


class ChordRequestState(str, Enum):
    GLOBAL_WAITING = "global_waiting"
    DISPATCHING = "dispatching"
    LOCAL_WAITING = "local_waiting"
    RUNNING = "running"
    MIGRATING = "migrating"
    WAITING_RETURNING = "waiting_returning"
    FINISHED = "finished"
    CANCELLED = "cancelled"


@dataclass(frozen=True, slots=True)
class WaitingRequestSnapshot:
    request_id: str
    owner_worker_id: int
    version: int
    phase: ChordRequestPhase
    prompt_tokens: int
    output_tokens: int
    num_preemptions: int
    block_demand: int
    arrival_time: float
    last_progress_time: float

    def urgency(self, now: float, coefficient: float, *, output_priority: bool = True) -> float:
        decode = self.output_tokens > 0 if output_priority else self.phase == ChordRequestPhase.PREEMPTED
        baseline = self.last_progress_time if decode else self.arrival_time
        scale = coefficient if decode else 1.0
        return scale * max(0.0, now - baseline)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WaitingRequestSnapshot:
        return cls(
            request_id=str(value["request_id"]),
            owner_worker_id=int(value.get("owner_worker_id", -1)),
            version=int(value.get("version", value.get("epoch", 0))),
            phase=ChordRequestPhase(value.get("phase", "never_run")),
            prompt_tokens=int(value.get("prompt_tokens", 0)),
            output_tokens=int(value.get("output_tokens", 0)),
            num_preemptions=int(value.get("num_preemptions", 0)),
            block_demand=max(1, int(value.get("block_demand", 1))),
            arrival_time=float(value.get("arrival_time", 0.0)),
            last_progress_time=float(value.get("last_progress_time", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class WorkerWaitingSnapshot:
    request_id: str
    epoch: int
    phase: ChordRequestPhase
    arrival_time: float
    last_progress_time: float
    waiting_since: float

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> WorkerWaitingSnapshot:
        return cls(
            request_id=str(value["request_id"]),
            epoch=int(value["epoch"]),
            phase=ChordRequestPhase(value.get("phase", "never_run")),
            arrival_time=float(value.get("arrival_time", 0.0)),
            last_progress_time=float(value.get("last_progress_time", 0.0)),
            waiting_since=float(value.get("waiting_since", 0.0)),
        )


@dataclass(frozen=True, slots=True)
class RequestProbe:
    request_id: str
    epoch: int
    state: str

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RequestProbe:
        return cls(
            request_id=str(value["request_id"]),
            epoch=int(value["epoch"]),
            state=str(value.get("state", "absent")),
        )


@dataclass(frozen=True, slots=True)
class ChordWorkerSnapshot:


    worker_id: int
    round_id: int
    schedulable: bool
    block_size: int
    total_blocks: int
    actual_used_blocks: int
    projected_blocks: int
    num_running_requests: int
    num_waiting_requests: int
    num_loading_requests: int
    num_migrate_in_reqs: int
    num_migrate_out_reqs: int
    waiting_return_ack_pending: int = 0
    handoff_pending: int = 0
    num_connector_waiting_requests: int = 0
    step_id: int = 0
    timestamp_ms: int = field(default_factory=lambda: int(time.time() * 1000))
    projected_tokens: int = 0
    all_prefills_tokens_num: int = 0
    decode_batch_size: int = 0
    all_decode_tokens: int = 0
    num_ordinary_waiting_requests: int = 0
    num_waiting_return_pending: int = 0
    waiting: tuple[WorkerWaitingSnapshot, ...] = ()
    probes: tuple[RequestProbe, ...] = ()
    unobserved_dispatch_count: int = 0
    unobserved_dispatch_blocks: int = 0
    migration_reservation_blocks: int = 0

    @property
    def num_total_gpu_tokens(self) -> int:
        return self.total_blocks * self.block_size

    def is_stale(self, stale_ms: int, now_ms: int | None = None) -> bool:
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        return now_ms - self.timestamp_ms > stale_ms

    @property
    def effective_projected_blocks(self) -> int:
        return (
            self.projected_blocks
            + self.unobserved_dispatch_blocks
            + self.migration_reservation_blocks
        )

    def with_reservations(
        self,
        *,
        dispatch_count: int,
        dispatch_blocks: int,
        migration_blocks: int,
    ) -> ChordWorkerSnapshot:
        return replace(
            self,
            unobserved_dispatch_count=dispatch_count,
            unobserved_dispatch_blocks=dispatch_blocks,
            migration_reservation_blocks=migration_blocks,
        )

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ChordWorkerSnapshot:
        block_size = int(value["block_size"])
        projected_tokens = int(value.get("projected_tokens", 0))
        projected_blocks = int(
            value.get(
                "projected_blocks",
                (projected_tokens + block_size - 1) // block_size,
            )
        )
        return cls(
            worker_id=int(value["worker_id"]),
            round_id=int(value.get("round_id", 0)),
            schedulable=bool(value.get("schedulable", True)),
            block_size=block_size,
            total_blocks=int(value["total_blocks"]),
            actual_used_blocks=int(value.get("actual_used_blocks", 0)),
            projected_blocks=projected_blocks,
            projected_tokens=projected_tokens,
            num_connector_waiting_requests=int(
                value.get("num_connector_waiting_requests", 0)
            ),
            waiting_return_ack_pending=int(value.get("waiting_return_ack_pending", 0)),
            handoff_pending=int(value.get("handoff_pending", 0)),
            step_id=int(value["step_id"]),
            timestamp_ms=int(value["timestamp_ms"]),
            all_prefills_tokens_num=int(value["all_prefills_tokens_num"]),
            decode_batch_size=int(value["decode_batch_size"]),
            all_decode_tokens=int(value["all_decode_tokens"]),
            num_running_requests=int(value.get("num_running_requests", 0)),
            num_waiting_requests=int(value.get("num_waiting_requests", 0)),
            num_loading_requests=int(value.get("num_loading_requests", 0)),
            num_migrate_in_reqs=int(value.get("num_migrate_in_reqs", 0)),
            num_migrate_out_reqs=int(value.get("num_migrate_out_reqs", 0)),
            num_ordinary_waiting_requests=int(
                value.get("num_ordinary_waiting_requests", 0)
            ),
            num_waiting_return_pending=int(value.get("num_waiting_return_pending", 0)),
            waiting=tuple(
                WorkerWaitingSnapshot.from_dict(item)
                for item in value.get("waiting", ())
            ),
            probes=tuple(
                RequestProbe.from_dict(item) for item in value.get("probes", ())
            ),
        )


@dataclass(frozen=True, slots=True)
class ChordAssignment:
    request: WaitingRequestSnapshot
    target_worker_id: int

    @property
    def is_remote(self) -> bool:
        return self.request.owner_worker_id != self.target_worker_id

    def to_dict(self) -> dict[str, Any]:
        return {
            "request": self.request.to_dict(),
            "target_worker_id": self.target_worker_id,
            "is_remote": self.is_remote,
        }
