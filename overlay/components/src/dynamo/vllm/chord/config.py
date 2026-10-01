from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass


TBF_PRECISION = 1e-3
WAITING_RETURN_ACK_RETRY_S = 2.0
TRACE_FLUSH_INTERVAL_S = 1.0


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int_env(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _float_env(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def _urls_env(name: str) -> tuple[str, ...]:
    return tuple(
        item.rstrip("/")
        for item in re.split(r"[\s,]+", os.getenv(name, "").strip())
        if item
    )


@dataclass(frozen=True, slots=True)
class ChordConfig:


    enabled: bool = False
    method: str = "chord"
    reference_policy: bool = False
    monitor_interval_ms: int = 50
    redispatch_interval_ms: int = 25
    dispatch_threshold_s: float = 2.0
    admission_on_worker_waiting: bool = True
    running_decision_interval_ms: int = 500
    status_stale_ms: int = 1_000
    tpot_profile_path: str = ""
    urgency_coefficient: float = 3.0
    tbf_max_t: float = 1.0
    max_plan_requests: int = 256
    waiting_return_timeout_s: float = 1.0
    skip_unfit: bool = False
    startup_timeout_s: float = 1_800.0
    running_migration_enabled: bool = True
    expected_workers: int = 0
    worker_control_urls: tuple[str, ...] = ()
    frontend_control_url: str = "http://127.0.0.1:18080"
    worker_id: int = -1
    trace_path: str = ""

    @property
    def waiting_enabled(self) -> bool:
        return self.method == "chord"

    @property
    def migration_enabled(self) -> bool:
        return self.method != "dynamo" and self.running_migration_enabled

    @classmethod
    def from_env(cls) -> ChordConfig:
        config = cls(
            method=os.getenv("DYN_ALIGNED_METHOD", "chord"),
            enabled=_bool_env("DYN_CHORD_ENABLE", False),
            reference_policy=_bool_env("DYN_CHORD_REFERENCE_POLICY", False),
            monitor_interval_ms=_int_env("DYN_CHORD_MONITOR_INTERVAL_MS", 50),
            redispatch_interval_ms=_int_env("DYN_CHORD_REDISPATCH_INTERVAL_MS", 25),
            dispatch_threshold_s=_float_env("DYN_CHORD_DISPATCH_THRESHOLD_S", 2.0),
            admission_on_worker_waiting=_bool_env("DYN_CHORD_ADMISSION_ON_WORKER_WAITING", True),
            running_decision_interval_ms=_int_env(
                "DYN_CHORD_RUNNING_DECISION_INTERVAL_MS", 500
            ),
            status_stale_ms=_int_env("DYN_CHORD_STATUS_STALE_MS", 1_000),
            tpot_profile_path=os.getenv("DYN_CHORD_TPOT_PROFILE_PATH", ""),
            urgency_coefficient=_float_env("DYN_CHORD_URGENCY_COEFFICIENT", 3.0),
            tbf_max_t=_float_env("DYN_CHORD_TBF_MAX_T", 1.0),
            max_plan_requests=_int_env("DYN_CHORD_MAX_PLAN_REQUESTS", 256),
            waiting_return_timeout_s=_float_env(
                "DYN_CHORD_WAITING_RETURN_TIMEOUT_S", 1.0
            ),
            skip_unfit=_bool_env("DYN_CHORD_SKIP_UNFIT", False),
            startup_timeout_s=_float_env("DYN_CHORD_STARTUP_TIMEOUT_S", 1_800.0),
            running_migration_enabled=_bool_env(
                "DYN_CHORD_RUNNING_MIGRATION_ENABLE", True
            ),
            expected_workers=_int_env("DYN_CHORD_EXPECTED_WORKERS", 0),
            worker_control_urls=_urls_env("DYN_CHORD_WORKER_CONTROL_URLS"),
            frontend_control_url=os.getenv(
                "DYN_CHORD_FRONTEND_CONTROL_URL", "http://127.0.0.1:18080"
            ).rstrip("/"),
            worker_id=_int_env("DYN_WORKER_ID", _int_env("DYN_FPM_WORKER_ID", -1)),
            trace_path=os.getenv("DYN_CHORD_TRACE_PATH", ""),
        )

        if "DYN_ALIGNED_METHOD" in os.environ:
            if not (config.enabled and _bool_env("DYN_LLUMNIX_ENABLE", False)
                    and _bool_env("DYN_ALIGNED_ENABLE", False)):
                raise ValueError("Unified methods require all three runtime enable flags")
            if not config.reference_policy:
                raise ValueError("Unified methods require the shared reference runtime")
        if config.method not in ("dynamo", "llumnix-sr", "llumnix-fcwsr", "chord"):
            raise ValueError(f"Unknown aligned method: {config.method}")
        return config

    def validate(
        self, *, require_worker_id: bool = True, require_worker_urls: bool = False
    ) -> None:
        if self.method not in ("dynamo", "llumnix-sr", "llumnix-fcwsr", "chord"):
            raise ValueError(f"Unknown aligned method: {self.method}")
        for name, value in {
            "monitor_interval_ms": self.monitor_interval_ms,
            "redispatch_interval_ms": self.redispatch_interval_ms,
            "running_decision_interval_ms": self.running_decision_interval_ms,
        }.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.running_decision_interval_ms < self.monitor_interval_ms:
            raise ValueError(
                "running_decision_interval_ms must be >= monitor_interval_ms"
            )
        if not math.isfinite(self.dispatch_threshold_s) or (
            self.dispatch_threshold_s < 0 and self.dispatch_threshold_s != -1
        ):
            raise ValueError("dispatch_threshold_s must be -1 or finite and nonnegative")
        if self.status_stale_ms < self.monitor_interval_ms:
            raise ValueError("status_stale_ms must cover one monitor interval")
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive")
        if self.urgency_coefficient <= 0:
            raise ValueError("urgency_coefficient must be positive")
        if self.tbf_max_t < 1:
            raise ValueError("tbf_max_t must be at least 1")
        if self.max_plan_requests <= 0:
            raise ValueError("max_plan_requests must be positive")
        if self.waiting_return_timeout_s <= 0:
            raise ValueError("waiting_return_timeout_s must be positive")
        if self.expected_workers < 0:
            raise ValueError("expected_workers must be non-negative")
        if require_worker_id and self.worker_id < 0:
            raise ValueError(
                "DYN_WORKER_ID or DYN_FPM_WORKER_ID must identify this worker"
            )
        if require_worker_urls and not self.worker_control_urls:
            raise ValueError(
                "DYN_CHORD_WORKER_CONTROL_URLS is required by ChordCoordinator"
            )

    def validate_frontend_router_mode(self, router_mode: str) -> None:


        if self.enabled and router_mode != "kv":
            raise ValueError(
                "DYN_CHORD_ENABLE=1 requires Frontend load-aware/kv routing; "
                f"got {router_mode!r}. Pooled requests pin the chosen worker "
                "while retaining native tracking."
            )
