from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class MigrationPolicy(str, Enum):


    SR = "SR"
    LR = "LR"
    FCR = "FCR"
    LCR = "LCR"
    FCW = "FCW"
    FCWSR = "FCWSR"


class MigrationLimit(str, Enum):
    TOKEN = "TOKEN"
    NUM_REQ = "NUM_REQ"
    RATIO = "RATIO"


class RequestPhase(str, Enum):
    WAITING = "waiting"
    RUNNING = "running"
    LOADING = "loading"


@dataclass(frozen=True, slots=True)
class RequestSnapshot:
    request_id: str
    phase: RequestPhase
    prompt_tokens: int
    output_tokens: int
    computed_tokens: int
    arrival_time: float
    migratable: bool = True

    @property
    def token_cost(self) -> int:
        return self.prompt_tokens + self.output_tokens

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["phase"] = self.phase.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> RequestSnapshot:
        return cls(
            request_id=str(value["request_id"]),
            phase=RequestPhase(value["phase"]),
            prompt_tokens=int(value["prompt_tokens"]),
            output_tokens=int(value["output_tokens"]),
            computed_tokens=int(value["computed_tokens"]),
            arrival_time=float(value["arrival_time"]),
            migratable=bool(value.get("migratable", True)),
        )


@dataclass(slots=True)
class InstanceStatus:
    worker_id: int
    timestamp_ms: int
    step_id: int
    schedulable: bool
    block_size: int
    num_total_gpu_tokens: int
    num_used_gpu_tokens: int
    num_uncomputed_tokens_all_waiting_prefills: int
    num_unallocated_tokens_scheduler_running_prefills: int
    num_unallocated_tokens_hybrid_scheduler_waiting_decodes: int
    num_running_requests: int
    num_waiting_requests: int
    num_loading_requests: int
    num_migrate_in_reqs: int
    num_migrate_out_reqs: int
    num_uncomputed_tokens_scheduler_running_prefills: int = 0
    hybrid_scheduler_waiting_to_decode_requests_num: int = 0
    hybrid_scheduler_waiting_to_decode_tokens_num: int = 0
    scheduler_waiting_to_decode_requests_num: int = 0
    scheduler_waiting_to_decode_tokens_num: int = 0
    scheduler_running_to_decode_requests_num: int = 0
    scheduler_running_to_decode_tokens_num: int = 0
    num_tokens_loading_requests: int = 0
    requests: list[RequestSnapshot] = field(default_factory=list)
    rpc_host: str = "127.0.0.1"
    rpc_port: int = 0

    @property
    def all_prefills_tokens_num(self) -> int:
        return (
            self.num_uncomputed_tokens_all_waiting_prefills
            + self.num_uncomputed_tokens_scheduler_running_prefills
        )

    @property
    def projected_tokens(self) -> int:
        return (
            self.num_used_gpu_tokens
            + self.num_uncomputed_tokens_all_waiting_prefills
            + self.num_unallocated_tokens_scheduler_running_prefills
            + self.num_unallocated_tokens_hybrid_scheduler_waiting_decodes
        )

    @property
    def projected_kv_usage(self) -> float:
        if self.num_total_gpu_tokens <= 0:
            return float("inf")
        return self.projected_tokens / self.num_total_gpu_tokens

    @property
    def decode_batch_size(self) -> int:
        return (
            self.hybrid_scheduler_waiting_to_decode_requests_num
            + self.num_loading_requests
            + self.scheduler_waiting_to_decode_requests_num
            + self.scheduler_running_to_decode_requests_num
        )

    @property
    def all_decode_tokens(self) -> int:
        return (
            self.hybrid_scheduler_waiting_to_decode_tokens_num
            + self.num_tokens_loading_requests
            + self.scheduler_waiting_to_decode_tokens_num
            + self.scheduler_running_to_decode_tokens_num
        )

    def is_stale(self, stale_ms: int, now_ms: int | None = None) -> bool:
        now_ms = int(time.time() * 1000) if now_ms is None else now_ms
        return now_ms - self.timestamp_ms > stale_ms

    def has_request(self, request_id: str) -> bool:
        return any(request.request_id == request_id for request in self.requests)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["requests"] = [request.to_dict() for request in self.requests]
        value.update(
            all_prefills_tokens_num=self.all_prefills_tokens_num,
            projected_tokens=self.projected_tokens,
            projected_kv_usage=self.projected_kv_usage,
            decode_batch_size=self.decode_batch_size,
            all_decode_tokens=self.all_decode_tokens,
        )
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InstanceStatus:
        names = cls.__dataclass_fields__.keys()
        data = {name: value[name] for name in names if name in value}
        data["requests"] = [
            RequestSnapshot.from_dict(item) for item in value.get("requests", [])
        ]
        return cls(**data)


@dataclass(frozen=True, slots=True)
class MigrationDecision:
    request_id: str
    source_worker_id: int
    target_worker_id: int
    source_rpc_host: str
    source_rpc_port: int
    output_tokens: int
    output_tokens_hint: int
    policy: MigrationPolicy
    trigger_policy: str = "llumnix-load-balance"
    source_phase: str = "unknown"
    source_computed_tokens: int = 0

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["policy"] = self.policy.value
        return value
