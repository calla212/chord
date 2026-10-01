from __future__ import annotations

import os
from dataclasses import dataclass

from .types import MigrationLimit, MigrationPolicy


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


@dataclass(frozen=True, slots=True)
class LlumnixConfig:


    enabled: bool = False
    status_interval_ms: int = 40
    status_stale_ms: int = 200
    decision_interval_ms: int = 500
    overload_threshold: float = 1.0
    underload_threshold: float = 1.0
    load_balance_threshold: float = 0.7
    policy: MigrationPolicy = MigrationPolicy.SR
    limit: MigrationLimit = MigrationLimit.TOKEN
    limit_value: float = 1024.0
    migrate_extra_tokens: int = 20
    migration_reservation_timeout_s: float = 10.0
    max_migrate_in_requests: int = 1
    max_migrate_in_tokens: int = 10_000
    max_migrate_in_ratio: float = 0.3
    rpc_port: int = 0
    worker_id: int = -1
    system_port: int = -1
    target_worker_ids: tuple[int, ...] = ()

    @classmethod
    def from_env(cls) -> LlumnixConfig:
        targets = tuple(
            int(item)
            for item in os.getenv("DYN_LLUMNIX_TARGET_WORKER_IDS", "").split(",")
            if item.strip()
        )
        return cls(
            enabled=_bool_env("DYN_LLUMNIX_ENABLE", False),
            status_interval_ms=_int_env("DYN_LLUMNIX_STATUS_INTERVAL_MS", 40),
            status_stale_ms=_int_env("DYN_LLUMNIX_STATUS_STALE_MS", 200),
            decision_interval_ms=_int_env("DYN_LLUMNIX_DECISION_INTERVAL_MS", 500),
            overload_threshold=_float_env("DYN_LLUMNIX_OVERLOAD_THRESHOLD", 1.0),
            underload_threshold=_float_env("DYN_LLUMNIX_UNDERLOAD_THRESHOLD", 1.0),
            load_balance_threshold=_float_env(
                "DYN_LLUMNIX_LOAD_BALANCE_THRESHOLD", 0.7
            ),
            policy=MigrationPolicy(
                os.getenv("DYN_LLUMNIX_MIGRATION_POLICY", "SR").upper()
            ),
            limit=MigrationLimit(
                os.getenv("DYN_LLUMNIX_MIGRATION_LIMIT", "TOKEN").upper()
            ),
            limit_value=_float_env("DYN_LLUMNIX_MIGRATION_VALUE", 1024.0),
            migrate_extra_tokens=_int_env("DYN_LLUMNIX_MIGRATE_EXTRA_TOKENS", 20),
            migration_reservation_timeout_s=_float_env(
                "DYN_LLUMNIX_MIGRATION_RESERVATION_TIMEOUT_S", 10.0
            ),
            max_migrate_in_requests=_int_env("DYN_LLUMNIX_MAX_MIGRATE_IN_REQUESTS", 1),
            max_migrate_in_tokens=_int_env("DYN_LLUMNIX_MAX_MIGRATE_IN_TOKENS", 10_000),
            max_migrate_in_ratio=_float_env("DYN_LLUMNIX_MAX_MIGRATE_IN_RATIO", 0.3),
            rpc_port=_int_env("DYN_LLUMNIX_RPC_PORT", 0),
            worker_id=_int_env("DYN_WORKER_ID", _int_env("DYN_FPM_WORKER_ID", -1)),
            system_port=_int_env("DYN_SYSTEM_PORT", -1),
            target_worker_ids=targets,
        )

    def validate(self) -> None:
        if self.status_interval_ms <= 0 or self.decision_interval_ms <= 0:
            raise ValueError("Llumnix collection intervals must be positive")
        if self.status_stale_ms < self.status_interval_ms:
            raise ValueError("status_stale_ms must be >= status_interval_ms")
        if not 0 <= self.underload_threshold <= self.overload_threshold:
            raise ValueError("invalid underload/overload thresholds")
        if self.limit_value == 0 or self.limit_value < -1:
            raise ValueError("migration value must be positive or -1")
        if self.load_balance_threshold < 0:
            raise ValueError("load_balance_threshold must be non-negative")
        if self.migrate_extra_tokens < 0:
            raise ValueError("migrate_extra_tokens must be non-negative")
        if self.migration_reservation_timeout_s <= 0:
            raise ValueError("migration_reservation_timeout_s must be positive")
        if self.rpc_port < 0 or self.rpc_port > 65535:
            raise ValueError("invalid KVT RPC port")
