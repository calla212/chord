from __future__ import annotations

import time
import threading
from uuid import uuid4
from collections.abc import Iterable
from typing import Any

from vllm.v1.request import Request

from .config import LlumnixConfig
from ..chord.config import ChordConfig
from .types import InstanceStatus
from .selection import select_requests
from .types import (
    MigrationLimit,
    MigrationPolicy,
    RequestPhase,
    RequestSnapshot,
)


class SchedulerStatusAdapter:


    def __init__(self, scheduler: Any, config: LlumnixConfig) -> None:
        self.scheduler = scheduler
        self.config = config
        self.method = ChordConfig.from_env().method
        self.step_id = 0
        self.migrating_out: set[str] = set()
        self._migration_lock = threading.Lock()
        self._migration_tickets: dict[str, str] = {}
        self._transferring: set[str] = set()
        self.last_num_scheduled_tokens: dict[str, int] = {}

    @staticmethod
    def _snapshot(
        request: Request, phase: RequestPhase, migratable: bool
    ) -> RequestSnapshot:
        return RequestSnapshot(
            request_id=request.request_id,
            phase=phase,
            prompt_tokens=request.num_prompt_tokens,
            output_tokens=request.num_output_tokens,
            computed_tokens=request.num_computed_tokens,
            arrival_time=request.arrival_time,
            migratable=migratable,
        )

    def _hybrid(self) -> Any | None:
        connector = self.scheduler.get_kv_connector()
        return getattr(connector, "_sched", None)

    def _scheduler_waiting(self) -> list[Request]:
        return list(self.scheduler.waiting) + list(
            getattr(self.scheduler, "skipped_waiting", ())
        )

    def _connector_state(
        self,
    ) -> tuple[list[tuple[Request, bool, bool]], list[Request]]:
        hybrid = self._hybrid()
        if hybrid is None:
            return [], []
        pending = list(getattr(hybrid, "_waiting", ()))
        loading = [item._req for item in getattr(hybrid, "_loading", {}).values()]
        return pending, loading

    def record_schedule(self, scheduler_output: Any) -> None:
        self.step_id += 1
        async_scheduling = bool(
            getattr(self.scheduler.scheduler_config, "async_scheduling", False)
        )
        self.last_num_scheduled_tokens = (
            dict(scheduler_output.num_scheduled_tokens) if async_scheduling else {}
        )

    def request_snapshots(self) -> list[RequestSnapshot]:
        snapshots: list[RequestSnapshot] = []
        seen: set[str] = set()

        def add(requests: Iterable[Request], phase: RequestPhase) -> None:
            for request in requests:
                if request.request_id in seen:
                    continue
                seen.add(request.request_id)
                snapshots.append(
                    self._snapshot(
                        request,
                        phase,
                        request.request_id not in self.migrating_out,
                    )
                )

        add(self._scheduler_waiting(), RequestPhase.WAITING)
        pending, loading = self._connector_state()
        add((item[0] for item in pending), RequestPhase.WAITING)
        add(loading, RequestPhase.LOADING)
        add(self.scheduler.running, RequestPhase.RUNNING)
        return snapshots

    def _computed_prefix_tokens(self, request: Request) -> int:
        try:
            _, tokens = self.scheduler.kv_cache_manager.get_computed_blocks(request)
            return int(tokens)
        except Exception:
            return request.num_computed_tokens

    @staticmethod
    def _will_decode(request: Request) -> bool:
        params = request.sampling_params
        return bool(params is not None and params.max_tokens > 1) or (
            request.num_tokens - request.num_computed_tokens == 1
        )

    def collect(self, *, include_requests: bool = True) -> InstanceStatus:
        cache_config = self.scheduler.cache_config
        block_size = int(cache_config.block_size)
        total_blocks = int(cache_config.num_gpu_blocks or 0)
        free_blocks = self.scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        used_tokens = max(0, total_blocks - free_blocks) * block_size

        scheduler_waiting = self._scheduler_waiting()
        connector_waiting, loading = self._connector_state()

        scheduler_waiting_prefill_tokens = 0
        scheduler_waiting_decode_requests = 0
        scheduler_waiting_decode_tokens = 0
        for request in scheduler_waiting:
            if request.num_tokens - request.num_computed_tokens > 1:
                scheduler_waiting_prefill_tokens += max(
                    0, request.num_tokens - self._computed_prefix_tokens(request)
                )
            if self._will_decode(request):
                scheduler_waiting_decode_requests += 1
                scheduler_waiting_decode_tokens += request.num_tokens

        hybrid_waiting_prefill_tokens = 0
        hybrid_waiting_decode_unallocated_tokens = 0
        hybrid_waiting_decode_requests = 0
        hybrid_waiting_decode_tokens = 0
        for request, load_count, save_count in connector_waiting:
            unallocated = max(
                0, request.num_tokens - self._computed_prefix_tokens(request)
            )
            if save_count:
                hybrid_waiting_prefill_tokens += unallocated
            if load_count:
                hybrid_waiting_decode_unallocated_tokens += unallocated
                hybrid_waiting_decode_requests += 1
                hybrid_waiting_decode_tokens += request.num_tokens

        running_prefill_uncomputed_tokens = 0
        running_prefill_unallocated_tokens = 0
        scheduler_running_decode_requests = 0
        scheduler_running_decode_tokens = 0
        for request in self.scheduler.running:


            pending = max(0, request.num_tokens - request.num_computed_tokens)
            scheduled = self.last_num_scheduled_tokens.get(request.request_id, 0)
            if (
                request.num_computed_tokens < request.num_prompt_tokens
                or pending > 1
                or scheduled > 1
            ):
                running_prefill_unallocated_tokens += pending
                running_prefill_uncomputed_tokens += pending + (
                    scheduled if scheduled > 1 else 0
                )

            params = request.sampling_params
            if (
                params is not None
                and params.max_tokens > 1
                and request.request_id not in self.migrating_out
            ):
                scheduler_running_decode_requests += 1
                scheduler_running_decode_tokens += request.num_tokens

        migrate_in = 0
        try:
            from blade_kvt.hybrid_connector.migration import _g_migrate_in_req_ids

            migrate_in = len(_g_migrate_in_req_ids)
        except ImportError:
            pass

        all_waiting = scheduler_waiting + [
            request for request, _, _ in connector_waiting
        ]
        unique_waiting = {request.request_id: request for request in all_waiting}
        return InstanceStatus(
            worker_id=self.config.worker_id,
            timestamp_ms=int(time.time() * 1000),
            step_id=self.step_id,
            schedulable=True,
            block_size=block_size,
            num_total_gpu_tokens=total_blocks * block_size,
            num_used_gpu_tokens=used_tokens,
            num_uncomputed_tokens_all_waiting_prefills=(
                scheduler_waiting_prefill_tokens + hybrid_waiting_prefill_tokens
            ),
            num_uncomputed_tokens_scheduler_running_prefills=(
                running_prefill_uncomputed_tokens
            ),
            num_unallocated_tokens_scheduler_running_prefills=(
                running_prefill_unallocated_tokens
            ),
            num_unallocated_tokens_hybrid_scheduler_waiting_decodes=(
                hybrid_waiting_decode_unallocated_tokens
            ),
            hybrid_scheduler_waiting_to_decode_requests_num=(
                hybrid_waiting_decode_requests
            ),
            hybrid_scheduler_waiting_to_decode_tokens_num=(
                hybrid_waiting_decode_tokens
            ),
            scheduler_waiting_to_decode_requests_num=(
                scheduler_waiting_decode_requests
            ),
            scheduler_waiting_to_decode_tokens_num=scheduler_waiting_decode_tokens,
            scheduler_running_to_decode_requests_num=(
                scheduler_running_decode_requests
            ),
            scheduler_running_to_decode_tokens_num=scheduler_running_decode_tokens,
            num_running_requests=len(self.scheduler.running),
            num_waiting_requests=len(unique_waiting),
            num_loading_requests=len(loading),
            num_tokens_loading_requests=sum(request.num_tokens for request in loading),
            num_migrate_in_reqs=migrate_in,
            num_migrate_out_reqs=len(self.migrating_out),
            requests=self.request_snapshots() if include_requests else [],
            rpc_host=_rpc_host(),
            rpc_port=self.config.rpc_port,
        )

    def prepare_migration(self, body: dict[str, Any]) -> dict[str, Any]:
        if self.method == "dynamo":
            return {"status": "disabled", "reason": "dynamo policy", "requests": []}
        policy = MigrationPolicy(
            str(body.get("policy", self.config.policy.value)).upper()
        )
        limit = MigrationLimit(str(body.get("limit", self.config.limit.value)).upper())
        value = float(body.get("value", self.config.limit_value))
        status = self.collect()
        excluded_request_ids = body.get("excluded_request_ids", ())
        if isinstance(excluded_request_ids, str):
            excluded_request_ids = (excluded_request_ids,)
        selected = select_requests(
            status.requests,
            policy,
            limit,
            value,
            status.num_total_gpu_tokens,
            excluded_request_ids=excluded_request_ids,
        )
        max_requests = int(
            body.get("max_requests", self.config.max_migrate_in_requests)
        )
        if max_requests >= 0:
            selected = selected[:max_requests]
        trigger = str(body.get("trigger_policy", "llumnix-load-balance"))
        requests = []
        with self._migration_lock:
            for request in selected:
                ticket = f"{trigger}:{uuid4().hex}"
                self.migrating_out.add(request.request_id)
                self._migration_tickets[request.request_id] = ticket
                requests.append({**request.to_dict(), "migration_ticket": ticket})
        return {
            "status": "ok",
            "worker_id": self.config.worker_id,
            "rpc_host": status.rpc_host,
            "rpc_port": status.rpc_port,
            "policy": policy.value,
            "migrate_extra_tokens": self.config.migrate_extra_tokens,
            "requests": requests,
        }

    def rollback(self, body: dict[str, Any]) -> dict[str, Any]:
        request_ids = body.get("request_ids", [])
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        rolled_back, in_flight = [], []
        with self._migration_lock:
            for request_id in map(str, request_ids):
                if request_id in self._transferring:
                    in_flight.append(request_id)
                    continue
                self.migrating_out.discard(request_id)
                self._migration_tickets.pop(request_id, None)
                rolled_back.append(request_id)
        return {"status": "ok", "request_ids": rolled_back,
                "in_flight_request_ids": in_flight}

    def begin_transfer(self, request_id: str, ticket: str) -> bool:


        with self._migration_lock:
            if (self._migration_tickets.get(request_id) != ticket
                    or request_id in self._transferring):
                return False
            self._transferring.add(request_id)
            return True

    def cleanup(self) -> None:
        live = set(self.scheduler.requests)
        with self._migration_lock:
            self.migrating_out.intersection_update(live)
            self._transferring.intersection_update(live)
            for request_id in self._migration_tickets.keys() - live:
                del self._migration_tickets[request_id]


def _rpc_host() -> str:
    try:
        from vllm.utils.network_utils import get_ip

        value = get_ip()
        return value[0] if isinstance(value, tuple) else value
    except Exception:
        return "127.0.0.1"
