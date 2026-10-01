from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Protocol


class LoadSample(Protocol):
    all_prefills_tokens_num: int
    projected_tokens: int
    num_total_gpu_tokens: int
    decode_batch_size: int
    all_decode_tokens: int


@dataclass(frozen=True, slots=True)
class EffectiveMetrics:
    all_prefills_tokens_num: int
    projected_tokens: int
    projected_kv_usage: float
    decode_batch_size: int
    all_decode_tokens: int
    inflight_dispatch_requests: int
    inflight_dispatch_prefill_tokens: int
    pending_migration_requests: int
    pending_migration_tokens: int

    def to_dict(self) -> dict:
        value = asdict(self)
        if not math.isfinite(self.projected_kv_usage):
            value["projected_kv_usage"] = None
        return value


def calculate_effective_metrics(
    status: LoadSample,
    *,
    dispatch_requests: int = 0,
    dispatch_tokens: int = 0,
    migration_requests: int = 0,
    migration_tokens: int = 0,
) -> EffectiveMetrics:


    dispatch_tokens = max(0, dispatch_tokens)
    migration_tokens = max(0, migration_tokens)
    projected = status.projected_tokens + dispatch_tokens + migration_tokens
    return EffectiveMetrics(
        all_prefills_tokens_num=status.all_prefills_tokens_num + dispatch_tokens,
        projected_tokens=projected,
        projected_kv_usage=(
            projected / status.num_total_gpu_tokens
            if status.num_total_gpu_tokens > 0
            else float("inf")
        ),
        decode_batch_size=status.decode_batch_size
        + dispatch_requests
        + migration_requests,
        all_decode_tokens=status.all_decode_tokens + dispatch_tokens + migration_tokens,
        inflight_dispatch_requests=dispatch_requests,
        inflight_dispatch_prefill_tokens=dispatch_tokens,
        pending_migration_requests=migration_requests,
        pending_migration_tokens=migration_tokens,
    )
