from __future__ import annotations


import os as _ttft_diag_os
if _ttft_diag_os.environ.get("DYN_TTFT_DIAG_ENABLE") == "1":
    import ttft_diag as _ttft_diag
else:
    _ttft_diag = None


import asyncio
import json
import logging
import time
from collections import Counter
from dataclasses import asdict
from typing import Any

import aiohttp

from ..llumnix.config import LlumnixConfig
from ..llumnix.migration import choose_migration_pairs
from ..llumnix.predictor import TpotPredictor
from ..llumnix.types import InstanceStatus
from .config import ChordConfig
from .planner import derive_redispatching_plan
from .protocol import (
    BACKEND_REQUEST_ID_KEY,
    ChordControlType,
    make_backend_request_id,
    split_backend_request_id,
)
from .state import (
    AsyncTrace,
    CoordinatorEvent,
    DispatchLease,
    GlobalPool,
    InstanceState,
    RequestRecord,
    Reservation,
)
from .types import ChordRequestPhase, ChordRequestState, ChordWorkerSnapshot

logger = logging.getLogger(__name__)


class ChordCoordinator:


    def __init__(
        self,
        config: ChordConfig | None = None,
        *,
        block_size: int = 16,
        native_router: Any = None,
    ) -> None:
        self.config = config or ChordConfig.from_env()
        self.config.validate(require_worker_id=False, require_worker_urls=True)
        self.llumnix_config = LlumnixConfig.from_env()
        self.block_size = block_size
        self._native = native_router
        self._latest_reports: dict[str, dict[str, Any]] = {}
        self._worker_runtime: dict[int, dict[str, Any]] = {}
        self._plan_pending = False
        self._admissions: dict[str, asyncio.Task[Any]] = {}
        logger.info(
            "Chord coordinator implementation=%s initial_policy=load-aware+track "
            "TPOT_profile=%s (diagnostic only)",
            __file__,
            self.config.tpot_profile_path or "unavailable",
        )
        self._predictor = (
            TpotPredictor.from_file(self.config.tpot_profile_path)
            if self.config.tpot_profile_path
            else TpotPredictor.unavailable()
        )

        self._records: dict[str, RequestRecord] = {}
        self._instances: dict[int, InstanceState] = {}
        self._worker_by_url: dict[str, int] = {}
        self._pool = GlobalPool(output_priority=not self.config.reference_policy)
        self._events: asyncio.Queue[CoordinatorEvent] = asyncio.Queue(maxsize=65536)
        self._probe_view: dict[str, tuple[dict[str, Any], ...]] = {}

        self._start_lock = asyncio.Lock()
        self._ready = asyncio.Event()
        self._started = False
        self._startup_error: str | None = None
        self._session: aiohttp.ClientSession | None = None
        self._tasks: set[asyncio.Task[Any]] = set()
        self._trace = AsyncTrace(self.config.trace_path)
        self.counters: Counter[str] = Counter()

        self._topology_signature: tuple[Any, ...] | None = None
        self._last_monitor_trace = 0.0
        self._last_plan_trace = 0.0
        self._complete_detail_round = -1
        self._last_migration_decision_round = -1
        self._migration_in_flight: set[tuple[int, int, int]] = set()

    async def submit(self, request_id: str, prompt_tokens: int) -> DispatchLease:


        await self._ensure_started()
        if not request_id:
            raise ValueError("Chord request_id must be non-empty")
        if prompt_tokens <= 0:
            raise ValueError("Chord prompt_tokens must be positive")

        loop = asyncio.get_running_loop()
        assignment: asyncio.Future[DispatchLease] = loop.create_future()
        accepted: asyncio.Future[None] = loop.create_future()
        now = time.time()
        record = RequestRecord(
            request_id=request_id,
            arrival_time=now,
            last_progress_time=now,
            prompt_tokens=prompt_tokens,
            assignment_waiter=assignment,
        )
        if _ttft_diag is not None:
            _ttft_diag.emit("coordinator_submit", request_id)
        await self._events.put(
            CoordinatorEvent("new_request", {"record": record}, accepted)
        )
        try:
            await accepted
            return await assignment
        except BaseException:
            if not assignment.done():
                assignment.cancel()
            raise

    async def redispatch_waiting(self, control: dict[str, Any]) -> DispatchLease:


        if not self.config.waiting_enabled:
            raise RuntimeError(f"Waiting redispatch is disabled for {self.config.method}")
        loop = asyncio.get_running_loop()
        assignment: asyncio.Future[DispatchLease] = loop.create_future()
        stored: asyncio.Future[None] = loop.create_future()
        await self._events.put(
            CoordinatorEvent(
                "waiting_return",
                {"control": dict(control), "assignment": assignment},
                stored,
            )
        )
        try:
            await stored
            return await assignment
        except BaseException:
            if not assignment.done():
                assignment.cancel()
            raise

    async def redispatch_open_failure(
        self,
        *,
        request_id: str,
        epoch: int,
        worker_id: int,
        error: BaseException,
    ) -> DispatchLease:


        loop = asyncio.get_running_loop()
        assignment: asyncio.Future[DispatchLease] = loop.create_future()
        stored: asyncio.Future[None] = loop.create_future()
        await self._events.put(
            CoordinatorEvent(
                "dispatch_failed",
                {
                    "request_id": request_id,
                    "epoch": epoch,
                    "worker_id": worker_id,
                    "error": error,
                    "assignment": assignment,
                },
                stored,
            )
        )
        try:
            await stored
            return await assignment
        except BaseException:
            if not assignment.done():
                assignment.cancel()
            raise

    async def finish(self, request_id: str, epoch: int, reason: str) -> None:
        if not self._started:
            return
        done = asyncio.get_running_loop().create_future()
        await self._events.put(
            CoordinatorEvent(
                "finished",
                {"request_id": request_id, "epoch": epoch, "reason": reason},
                done,
            )
        )
        await done

    async def cancel(self, request_id: str) -> None:
        if not self._started:
            return
        done = asyncio.get_running_loop().create_future()
        await self._events.put(
            CoordinatorEvent("cancelled", {"request_id": request_id}, done)
        )
        await done

    async def start(self) -> None:
        await self._ensure_started()

    async def _ensure_started(self) -> None:
        if not self._started:
            async with self._start_lock:
                if not self._started:
                    self._session = aiohttp.ClientSession(
                        timeout=aiohttp.ClientTimeout(total=5.0)
                    )
                    self._started = True
                    self._spawn(self._coordinator_loop(), "chord-coordinator")
                    self._spawn(self._status_loop(), "chord-monitor")
                    if not self.config.reference_policy:
                        self._spawn(self._planner_loop(), "chord-planner")
        try:
            await asyncio.wait_for(
                self._ready.wait(), timeout=self.config.startup_timeout_s
            )
            if self._startup_error is not None:
                raise RuntimeError(self._startup_error)
        except TimeoutError as error:
            raise RuntimeError(
                "Chord Coordinator timed out waiting for connected workers: "
                f"have={sum(item.connected for item in self._instances.values())} "
                f"expected={self._expected_workers()}"
            ) from error

    def _spawn(self, coroutine: Any, name: str) -> asyncio.Task[Any]:
        task = asyncio.create_task(coroutine, name=name)
        self._tasks.add(task)

        def done(completed: asyncio.Task[Any]) -> None:
            self._tasks.discard(completed)
            if completed.cancelled():
                return
            error = completed.exception()
            if error is not None:
                logger.error(
                    "Chord task %s failed",
                    name,
                    exc_info=(type(error), error, error.__traceback__),
                )

        task.add_done_callback(done)
        return task

    async def state(self) -> dict[str, Any]:
        future = asyncio.get_running_loop().create_future()
        await self._events.put(CoordinatorEvent("query_state", future=future))
        return await future

    def _worker_load_payloads(self, *, include_accounts: bool) -> list[dict[str, Any]]:
        now = time.monotonic()
        workers = []
        for item in sorted(self._instances.values(), key=lambda item: item.worker_id):
            compact = item.compact
            if compact is None:
                continue
            metrics = item.effective_metrics(
                now=now,
                migration_timeout_s=self.llumnix_config.migration_reservation_timeout_s,
            )
            value = {
                "worker_id": item.worker_id,
                "control_url": item.control_url,
                "connected": item.connected,
                "fresh": item.connected
                and not compact.is_stale(self.config.status_stale_ms),
                "step_id": compact.step_id,
                "round_id": compact.round_id,
                "timestamp_ms": compact.timestamp_ms,
                "running": compact.num_running_requests,
                "waiting": compact.num_waiting_requests,
                "loading": compact.num_loading_requests,
                "connector_waiting": compact.num_connector_waiting_requests,
                "waiting_return_pending": compact.num_waiting_return_pending,
                "waiting_return_ack_pending": compact.waiting_return_ack_pending,
                "handoff_pending": compact.handoff_pending,
                "migrate_in": compact.num_migrate_in_reqs,
                "migrate_out": compact.num_migrate_out_reqs,
                "actual_used_blocks": compact.actual_used_blocks,
                "effective": metrics.to_dict(),
                "predicted_tpot_next_zero_prompt": self._predictor.predict_next_decode(
                    metrics, 0
                ).to_dict(),
            }
            if include_accounts:
                value["unobserved_dispatches"] = [
                    dict(asdict(r), age_s=max(0, now - r.created_at))
                    for r in item.unobserved_dispatches.values()
                ]
                value["migration_reservations"] = [
                    dict(
                        asdict(r),
                        age_s=max(0, now - r.created_at),
                        accounting_active=now - r.created_at
                        < self.llumnix_config.migration_reservation_timeout_s,
                    )
                    for r in item.migration_reservations.values()
                ]
                value["forbidden_dispatches"] = sorted(item.forbidden_dispatches)
            workers.append(value)
        return workers

    def _state_payload(self) -> dict[str, Any]:
        return {
            "ready": self._ready.is_set(),
            "runtime": {
                "method": self.config.method,
                "controller": type(self).__name__,
                "source": __file__,
                "monitor_interval_ms": self.config.monitor_interval_ms,
                "running_decision_interval_ms": self.config.running_decision_interval_ms,
                "status_stale_ms": self.config.status_stale_ms,
                "workers": self._worker_runtime,
            },
            "active_requests": len(self._records),
            "global_waiting": len(self._pool),
            "migration_in_flight": len(self._migration_in_flight),
            "native_admissions": len(self._admissions),
            "counters": dict(self.counters),
            "native_loads": self._native.loads() if self._native is not None else [],
            "policy": {
                "initial_dispatch_metric": "native-load-aware+track",
                "redispatch_interval_ms": self.config.redispatch_interval_ms,
                "dispatch_threshold_s": self.config.dispatch_threshold_s,
                "urgency_coefficient": self.config.urgency_coefficient,
                "skip_unfit": self.config.skip_unfit or self.config.dispatch_threshold_s == -1,
                "top_k": 1,
                "migration_metric": "projected_kv_usage",
                "migration_policy": self.llumnix_config.policy.value,
                "migration_limit": self.llumnix_config.limit.value,
                "migration_value": self.llumnix_config.limit_value,
                "overload_threshold": self.llumnix_config.overload_threshold,
                "underload_threshold": self.llumnix_config.underload_threshold,
                "load_balance_threshold": self.llumnix_config.load_balance_threshold,
                "waiting_return_timeout_s": self.config.waiting_return_timeout_s,
                "pool_nonempty_suppresses_running_migration": self.config.waiting_enabled,
                "waiting_enabled": self.config.waiting_enabled,
                "running_migration_enabled": self.config.migration_enabled,
                "predicted_tpot_used_for_selection": False,
                "monitor_interval_ms": self.config.monitor_interval_ms,
                "running_decision_interval_ms": self.config.running_decision_interval_ms,
            },
            "workers": self._worker_load_payloads(include_accounts=True),
        }

    async def close(self) -> None:
        if not self._started and self._session is None:
            self._trace.close()
            return
        self._started = False
        tasks = tuple(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._tasks.clear()

        error = RuntimeError("Chord Coordinator closed")
        for record in self._records.values():
            waiter = record.assignment_waiter
            if waiter is not None and not waiter.done():
                waiter.set_exception(error)
        self._records.clear()

        if self._session is not None:
            await self._session.close()
            self._session = None
        self._trace.close()

    def _expected_workers(self) -> int:
        return self.config.expected_workers or len(self.config.worker_control_urls)

    async def _post(
        self, base_url: str, control: str, body: dict[str, Any]
    ) -> dict[str, Any]:
        assert self._session is not None
        url = f"{base_url.rstrip('/')}/engine/control/{control}"
        async with self._session.post(url, json=body) as response:
            text = await response.text()
            if response.status != 200:
                raise RuntimeError(f"{url} returned {response.status}: {text}")
            value = json.loads(text)
            if not isinstance(value, dict):
                raise TypeError(f"{control} returned non-object JSON")
            return value

    async def _poll_worker(self, base_url: str, body: dict[str, Any]) -> dict[str, Any]:
        value = await self._post(base_url, "chord_status", body)
        if value.get("status") != "ok":
            raise RuntimeError(f"worker reports Chord unavailable: {value}")
        return value

    def _router(self) -> Any:
        if self._native is None:
            from dynamo._core import AlignedRouter
            self._native = AlignedRouter()
        return self._native

    def report(self, base_url: str, value: dict[str, Any]) -> list[dict[str, Any]]:
        if base_url not in self.config.worker_control_urls:
            raise ValueError(f"Unknown Chord reporting worker: {base_url}")
        if base_url in self._latest_reports:
            self.counters["reports_replaced"] += 1
        self._latest_reports[base_url] = value
        return list(self._probe_view.get(base_url, ()))

    async def migration_committed(self, body: dict[str, Any]) -> None:
        done = asyncio.get_running_loop().create_future()
        await self._events.put(CoordinatorEvent("migration_committed", body, done))
        await done

    async def _planner_loop(self) -> None:
        interval = self.config.redispatch_interval_ms / 1000
        while True:
            await asyncio.sleep(interval)
            if not self._plan_pending:
                self._plan_pending = True
                await self._events.put(CoordinatorEvent("plan_tick"))

    async def _status_loop(self) -> None:


        interval = (self.config.monitor_interval_ms if self.config.reference_policy
                    else self.config.running_decision_interval_ms) / 1000
        detail_every = max(1, round(self.config.running_decision_interval_ms / (interval * 1000)))
        round_id = 0
        while True:
            started = time.monotonic()
            round_id += 1
            include_migration = round_id == 1 or round_id % detail_every == 0
            values = await asyncio.gather(*(
                self._poll_worker(base_url, {
                    "round_id": round_id, "include_migration": include_migration,
                    "probes": list(self._probe_view.get(base_url, ())),
                }) for base_url in self.config.worker_control_urls
            ), return_exceptions=True)
            await self._events.put(CoordinatorEvent("status_round", {
                "round_id": round_id, "include_migration": include_migration,
                "results": list(zip(self.config.worker_control_urls, values, strict=True)),
            }))
            await asyncio.sleep(max(0.001, interval - (time.monotonic() - started)))

    async def _coordinator_loop(self) -> None:
        if _ttft_diag is not None:
            _ttft_diag.start_loop_probe("coordinator")
        while True:
            event = await self._events.get()
            try:
                if event.kind == "query_state":
                    if event.future is not None and not event.future.done():
                        event.future.set_result(self._state_payload())
                else:
                    await self._handle_event(event)
                    if self.config.reference_policy:
                        await self._schedule_pool()
                    if event.kind == "status_round":
                        self._maybe_start_migration()
            except Exception as error:
                self.counters["coordinator_errors"] += 1
                self._trace.emit(
                    "coordinator_error", kind=event.kind, error=repr(error)
                )
                logger.exception("Chord coordinator event failed: %s", event.kind)
                if event.future is not None and not event.future.done():
                    event.future.set_exception(error)
            else:
                if event.future is not None and not event.future.done():
                    event.future.set_result(None)
            finally:
                self._refresh_probe_view()

    async def _handle_event(self, event: CoordinatorEvent) -> None:
        kind = event.kind
        if kind == "new_request":
            await self._handle_new_request(event.data["record"])
        elif kind == "reserved":
            await self._handle_reserved(event.data)
        elif kind == "status_round":
            self._handle_status_round(event.data)
        elif kind == "plan_tick":
            self._plan_pending = False
            reports, self._latest_reports = self._latest_reports, {}
            for base_url, value in reports.items():
                self._handle_status_round({
                    "round_id": value["round_id"], "include_migration": False,
                    "push": True, "results": [(base_url, value)],
                })
            await self._schedule_pool()
        elif kind in {"dispatch_failed", "waiting_return", "finished", "cancelled"}:
            data = event.data["control"] if kind == "waiting_return" else event.data
            record = self._records.get(str(data["request_id"]))
            epoch = int(data.get("epoch", record.epoch if record else 0))
            getattr(self, f"_handle_{kind}")(event.data)
            if epoch > 0:
                await self._router().free(make_backend_request_id(str(data["request_id"]), epoch))
        elif kind == "migration_result":
            self._handle_migration_result(event.data)
        elif kind == "migration_committed":
            self._handle_migration_committed(event.data)
        else:
            raise ValueError(f"unknown Chord event: {kind}")

    async def _handle_new_request(self, record: RequestRecord) -> None:
        if _ttft_diag is not None:
            _ttft_diag.emit("coordinator_handle", record.request_id)
        if record.request_id in self._records or record.request_id in self._admissions:
            raise RuntimeError(
                f"duplicate active Chord request ID: {record.request_id}"
            )
        self._records[record.request_id] = record
        worker_waiting = self.config.admission_on_worker_waiting and any(
            item.connected and item.compact is not None
            and not item.compact.is_stale(self.config.status_stale_ms)
            and item.compact.num_waiting_requests > 0
            for item in self._instances.values()
        )
        fast_path = not self.config.waiting_enabled or (
            not (self._pool or worker_waiting) and bool(self._planning_workers()))
        if not fast_path:
            self._pool.add(record)
            if _ttft_diag is not None:
                _ttft_diag.emit("global_pool_enter", record.request_id, epoch=record.epoch)
            self.counters["global_enqueued"] += 1
        else:
            await self._launch(record)
            self.counters["fast_path"] += 1
        self._trace.emit(
            "request_arrived",
            request_id=record.request_id,
            prompt_tokens=record.prompt_tokens,
            fast_path=fast_path,
        )

    def _handle_status_round(self, data: dict[str, Any]) -> None:
        round_id = int(data["round_id"])
        include_migration = bool(data["include_migration"])
        pushed = bool(data.get("push", False))
        for base_url, result in data["results"]:
            if isinstance(result, BaseException):
                worker_id = self._worker_by_url.get(base_url)
                if worker_id is not None and worker_id in self._instances:
                    self._instances[worker_id].connected = False
                self.counters["status_errors"] += 1
                continue

            runtime = result.get("runtime")
            if runtime is not None and runtime["method"] != self.config.method:
                self._startup_error = f"Worker method mismatch: {runtime['method']} != {self.config.method}"
                self._ready.set()
                raise RuntimeError(self._startup_error)
            snapshot = ChordWorkerSnapshot.from_dict(result)
            if snapshot.round_id != round_id:
                self.counters["old_status_rounds"] += 1
                continue
            previous_id = self._worker_by_url.get(base_url)
            if previous_id is not None and previous_id != snapshot.worker_id:
                old = self._instances.get(previous_id)
                if old is not None and old.control_url == base_url:
                    old.connected = False
                self.counters["worker_identity_changes"] += 1
            self._worker_by_url[base_url] = snapshot.worker_id
            if runtime is not None:
                self._worker_runtime[snapshot.worker_id] = runtime

            instance = self._instances.get(snapshot.worker_id)
            if instance is None:
                instance = InstanceState(snapshot.worker_id, base_url)
                self._instances[snapshot.worker_id] = instance
            if not pushed and round_id <= instance.latest_round:
                self.counters["old_status_rounds"] += 1
                continue

            known = {
                item.compact.block_size
                for item in self._instances.values()
                if item.compact is not None and item.worker_id != snapshot.worker_id
            }
            if known and known != {snapshot.block_size}:
                raise RuntimeError(
                    "Chord workers disagree on vLLM block size: "
                    f"existing={sorted(known)} worker={snapshot.worker_id}:"
                    f"{snapshot.block_size}"
                )
            self.block_size = snapshot.block_size
            instance.control_url = base_url
            instance.connected = True
            if not pushed:
                instance.latest_round = round_id
            if instance.compact is None or snapshot.timestamp_ms >= instance.compact.timestamp_ms:
                instance.compact = snapshot
                self._reconcile_instance(instance, snapshot)
            detailed = result.get("llumnix")
            if include_migration and isinstance(detailed, dict):
                instance.detailed = InstanceStatus.from_dict(detailed)
                instance.detail_round = round_id

        connected = [item for item in self._instances.values() if item.connected]
        if len(connected) >= self._expected_workers():
            self._ready.set()
        else:
            self._ready.clear()

        if (
            include_migration
            and len(connected) >= self._expected_workers()
            and all(item.detail_round == round_id for item in connected)
        ):
            self._complete_detail_round = round_id

        connected_signature = tuple(sorted(item.worker_id for item in connected))
        signature = (connected_signature, self._expected_workers())
        if signature != self._topology_signature:
            self._topology_signature = signature
            self._trace.emit(
                "worker_topology",
                expected_workers=self._expected_workers(),
                active_worker_ids=connected_signature,
                fresh_worker_ids=connected_signature,
            )
        self._emit_monitor_trace(connected)

    def _reconcile_instance(
        self, instance: InstanceState, snapshot: ChordWorkerSnapshot
    ) -> None:
        probes = {(item.request_id, item.epoch): item.state for item in snapshot.probes}
        for key in tuple(instance.unobserved_dispatches):
            state = probes.get(key)
            if (
                state is not None
                and state != "absent"
                and snapshot.step_id
                > instance.unobserved_dispatches[key].baseline_step_id
            ):
                instance.unobserved_dispatches.pop(key, None)
                self._apply_observed_state(instance.worker_id, key, state)
        for key in tuple(instance.migration_reservations):
            state = probes.get(key)
            if (
                state is not None
                and state != "absent"
                and snapshot.step_id
                > instance.migration_reservations[key].baseline_step_id
            ):
                instance.migration_reservations.pop(key, None)
                self._apply_observed_state(instance.worker_id, key, state)

        for waiting in snapshot.waiting:
            self._apply_observed_state(
                instance.worker_id,
                (waiting.request_id, waiting.epoch),
                "waiting",
            )

        for key in tuple(instance.forbidden_dispatches):
            if probes.get(key) != "absent":
                continue
            instance.forbidden_dispatches.discard(key)
            record = self._records.get(key[0])
            if record is None:
                continue
            epochs = record.forbidden_epochs.get(instance.worker_id)
            if epochs is None:
                continue
            epochs.discard(key[1])
            if not epochs:
                record.forbidden_epochs.pop(instance.worker_id, None)

    def _apply_observed_state(
        self, worker_id: int, key: tuple[str, int], state: str
    ) -> None:
        request_id, epoch = key
        record = self._records.get(request_id)
        if (
            record is None
            or record.epoch != epoch
            or record.compute_worker_id != worker_id
            or record.state in {ChordRequestState.FINISHED, ChordRequestState.CANCELLED}
        ):
            return
        if state == "waiting":
            record.state = ChordRequestState.LOCAL_WAITING
        elif state == "waiting_returning":
            record.state = ChordRequestState.WAITING_RETURNING
        elif (
            state in {"running", "active", "loading", "connector_waiting"}
            and record.state != ChordRequestState.MIGRATING
        ):
            record.state = ChordRequestState.RUNNING

    def _emit_monitor_trace(self, connected: list[InstanceState]) -> None:
        now = time.monotonic()
        if now - self._last_monitor_trace < 0.5:
            return
        self._last_monitor_trace = now
        wall_now = time.time()
        ages = [
            max(0.0, wall_now - waiting.waiting_since)
            for instance in connected
            if instance.compact is not None
            for waiting in instance.compact.waiting
        ]
        self._trace.emit(
            "monitor",
            connected_workers=len(connected),
            active_requests=len(self._records),
            global_waiting=len(self._pool),
            local_waiting=sum(
                item.compact.num_waiting_requests
                for item in connected
                if item.compact is not None
            ),
            ordinary_local_waiting=sum(
                item.compact.num_ordinary_waiting_requests
                for item in connected
                if item.compact is not None
            ),
            waiting_return_pending=sum(
                item.compact.num_waiting_return_pending
                for item in connected
                if item.compact is not None
            ),
            max_local_waiting_age_s=max(ages, default=0.0),
            mean_local_waiting_age_s=(sum(ages) / len(ages) if ages else 0.0),
            waiting_relocation_enabled=False,
            token_events=0,
            trace_dropped=self._trace.dropped,
            workers=self._worker_load_payloads(include_accounts=False),
        )

    def _planning_workers(self) -> list[ChordWorkerSnapshot]:
        now = time.monotonic()
        return [
            snapshot
            for instance in self._instances.values()
            if instance.compact is not None
            and not instance.compact.is_stale(self.config.status_stale_ms)
            and (
                snapshot := instance.planning_snapshot(
                    now=now,
                    migration_reservation_timeout_s=(
                        self.llumnix_config.migration_reservation_timeout_s
                    ),
                )
            )
            is not None
        ]

    async def _schedule_pool(self) -> None:

        if not self._pool or self._admissions:
            return
        workers = self._planning_workers()
        if not workers:
            return
        now = time.time()
        candidates = self._pool.candidates(
            self._records,
            now=now,
            coefficient=self.config.urgency_coefficient,
            limit=self.config.max_plan_requests,
        )
        if not candidates:
            return
        if not self.config.waiting_enabled:

            for record in sorted(candidates, key=lambda r: (r.arrival_time, r.request_id)):
                await self._launch(record)
            return
        forbidden = {
            record.request_id: tuple(record.forbidden_epochs)
            for record in candidates
            if record.forbidden_epochs
        }
        plan = derive_redispatching_plan(
            [record.snapshot(self.block_size) for record in candidates],
            workers,
            now=now,
            urgency_coefficient=self.config.urgency_coefficient,
            max_t=self.config.tbf_max_t,
            max_plan_requests=self.config.max_plan_requests,
            skip_unfit=self.config.skip_unfit,
            dispatch_threshold_s=self.config.dispatch_threshold_s,
            output_priority=not self.config.reference_policy,
            forbidden_targets=forbidden,
        )
        assignments = 0
        for assignment in plan.assignments:
            record = self._records.get(assignment.request.request_id)
            if record is None or record.state != ChordRequestState.GLOBAL_WAITING:
                continue
            await self._launch(record, assignment.target_worker_id)
            assignments += 1

        monotonic_now = time.monotonic()
        if assignments or monotonic_now - self._last_plan_trace >= 0.5:
            self._last_plan_trace = monotonic_now
            self._trace.emit(
                "redispatch_plan",
                candidates=len(candidates),
                global_waiting=len(self._pool),
                assignments=assignments,
                resource_limit=plan.resource_limit,
                all_placed=plan.all_placed,
                stopped_request_id=plan.stopped_request_id,
            )

    async def _launch(self, record: RequestRecord, target_worker_id: int | None = None) -> None:
        if record.state != ChordRequestState.GLOBAL_WAITING:
            raise RuntimeError(f"Cannot dispatch {record.request_id} from {record.state.value}")
        waiter = record.assignment_waiter
        if waiter is None or waiter.done():
            self._pool.discard(record.request_id)
            self._records.pop(record.request_id, None)
            return
        allowed = [worker.worker_id for worker in self._planning_workers()
                   if worker.worker_id not in record.forbidden_epochs]
        if not allowed or (target_worker_id is not None and target_worker_id not in allowed):
            self._pool.add(record)
            return
        epoch = record.next_epoch
        record.next_epoch += 1
        record.epoch = epoch
        record.state = ChordRequestState.DISPATCHING
        self._pool.discard(record.request_id)
        self._admissions[record.request_id] = self._spawn(
            self._reserve(record.request_id, epoch, record.context_tokens, allowed, target_worker_id),
            f"chord-admit-{record.request_id}-{epoch}",
        )

    async def _reserve(self, request_id: str, epoch: int, input_tokens: int,
                       allowed: list[int], target: int | None) -> None:
        backend_id = make_backend_request_id(request_id, epoch)
        error = None
        try:
            reservation = asyncio.ensure_future(self._router().reserve(
                backend_id, input_tokens, allowed, target))
            target, _ = await asyncio.shield(reservation)
        except asyncio.CancelledError:
            await reservation
            await self._router().free(backend_id)
            raise
        except Exception as caught:
            error = caught
        await self._events.put(CoordinatorEvent("reserved", {
            "request_id": request_id, "epoch": epoch, "target": target, "error": error,
        }))

    async def _handle_reserved(self, data: dict[str, Any]) -> None:
        request_id, epoch = str(data["request_id"]), int(data["epoch"])
        self._admissions.pop(request_id, None)
        record = self._records.get(request_id)
        backend_request_id = make_backend_request_id(request_id, epoch)
        waiter = record.assignment_waiter if record is not None else None
        if record is None or record.epoch != epoch or waiter is None or waiter.done():
            if data["error"] is None:
                await self._router().free(backend_request_id)
            self._records.pop(request_id, None)
            return
        if data["error"] is not None:
            self._records.pop(request_id, None)
            waiter.set_exception(data["error"])
            self.counters["admission_errors"] += 1
            return
        target_worker_id = int(data["target"])
        instance = self._instances[target_worker_id]

        returned_from = record.last_return_worker_id
        same_source_redispatch = returned_from == target_worker_id
        if returned_from is not None:
            self.counters["waiting_redispatches"] += 1
            if same_source_redispatch:
                self.counters["same_source_redispatches"] += 1
        record.last_return_worker_id = None

        record.epoch = epoch
        record.compute_worker_id = target_worker_id
        record.stream_owner_worker_id = target_worker_id
        record.state = ChordRequestState.DISPATCHING
        self._pool.discard(record.request_id)

        instance.unobserved_dispatches[(record.request_id, epoch)] = Reservation(
            request_id=record.request_id,
            epoch=epoch,
            blocks=record.block_demand(self.block_size),
            context_tokens=record.context_tokens,
            baseline_step_id=instance.compact.step_id if instance.compact else -1,
            created_at=time.monotonic(),
        )
        metadata = {
            "request_id": record.request_id,
            BACKEND_REQUEST_ID_KEY: backend_request_id,
            "epoch": epoch,
            "arrival_time": record.arrival_time,
            "last_progress_time": record.last_progress_time,
            "num_preemptions": record.num_preemptions,
            "output_token_ids": list(record.resume_output_token_ids),
            "stream_owner_worker_id": target_worker_id,
            "native_reserved": True,
        }
        lease = DispatchLease(
            request_id=record.request_id,
            backend_request_id=backend_request_id,
            epoch=epoch,
            worker_id=target_worker_id,
            metadata=metadata,
        )
        record.assignment_waiter = None
        if _ttft_diag is not None:
            _ttft_diag.emit("global_dispatch", record.request_id, epoch=epoch, target_worker_id=target_worker_id, resume_tokens=len(record.resume_output_token_ids))
        waiter.set_result(lease)
        self.counters["dispatches"] += 1
        self._trace.emit(
            "dispatch",
            request_id=record.request_id,
            epoch=epoch,
            worker_id=target_worker_id,
            resume_tokens=len(record.resume_output_token_ids),
            returned_from_worker_id=returned_from,
            same_source_redispatch=same_source_redispatch,
        )

    def _current_record(self, data: dict[str, Any]) -> RequestRecord | None:
        record = self._records.get(str(data["request_id"]))
        if record is None or record.epoch != int(data["epoch"]):
            return None
        return record

    def _handle_dispatch_failed(self, data: dict[str, Any]) -> None:
        record = self._current_record(data)
        if record is None or record.state in {
            ChordRequestState.FINISHED,
            ChordRequestState.CANCELLED,
        }:
            raise RuntimeError(
                f"stale dispatch failure request={data['request_id']} "
                f"epoch={data['epoch']}"
            )
        assignment = data["assignment"]
        if not hasattr(assignment, "set_result"):
            raise TypeError("dispatch failure requires an assignment future")

        worker_id = int(data["worker_id"])
        old_epoch = record.epoch
        self._remove_reservation(record.request_id, old_epoch)
        record.forbidden_epochs.setdefault(worker_id, set()).add(old_epoch)
        instance = self._instances.get(worker_id)
        if instance is not None:
            instance.forbidden_dispatches.add((record.request_id, old_epoch))
        record.compute_worker_id = None
        record.stream_owner_worker_id = None
        record.state = ChordRequestState.GLOBAL_WAITING
        record.assignment_waiter = assignment
        self._pool.add(record)
        if _ttft_diag is not None:
            _ttft_diag.emit("global_pool_enter", record.request_id, epoch=record.epoch)
        self.counters["ambiguous_dispatches"] += 1
        self._trace.emit(
            "dispatch_ambiguous",
            request_id=record.request_id,
            epoch=old_epoch,
            worker_id=worker_id,
            error=repr(data["error"]),
        )
        if instance is not None:
            self._spawn(
                self._drop_old_epoch(
                    instance.control_url, record.request_id, old_epoch
                ),
                f"chord-drop-{record.request_id[:8]}-{old_epoch}",
            )

    async def _drop_old_epoch(self, base_url: str, request_id: str, epoch: int) -> None:
        try:
            await self._post(
                base_url,
                "chord_drop_epoch",
                {"request_id": request_id, "epoch": epoch},
            )
        except Exception:
            logger.warning(
                "Chord stale-epoch cleanup failed: request=%s epoch=%d worker=%s",
                request_id,
                epoch,
                base_url,
                exc_info=True,
            )

    def _handle_waiting_return(self, data: dict[str, Any]) -> None:
        control = data["control"]
        if control.get("type") != ChordControlType.WAITING_RETURN.value:
            raise ValueError(f"unexpected Chord control: {control}")
        request_id = str(control["request_id"])
        epoch = int(control["epoch"])
        owner_id = int(control["stream_owner_worker_id"])
        compute_worker_id = int(control["compute_worker_id"])
        phase = ChordRequestPhase(str(control["phase"]))
        reason = str(control.get("reason", "waiting_timeout"))
        waiting_since = float(control["waiting_since"])
        wait_age_s = float(control["wait_age_s"])
        if reason != "waiting_timeout" or waiting_since <= 0 or wait_age_s < 0:
            raise ValueError(f"invalid Chord waiting return: {control}")

        instance = self._instances.get(owner_id)
        if instance is not None:
            self._spawn(
                self._ack_waiting_return(instance.control_url, request_id, epoch),
                f"chord-waiting-return-ack-{request_id[:8]}-{epoch}",
            )

        record = self._records.get(request_id)
        if (
            record is None
            or record.epoch != epoch
            or record.state in {ChordRequestState.FINISHED, ChordRequestState.CANCELLED}
        ):
            self.counters["stale_waiting_returns"] += 1
            raise RuntimeError(
                f"stale waiting return request={request_id} epoch={epoch}"
            )
        if record.state == ChordRequestState.GLOBAL_WAITING:
            self.counters["duplicate_waiting_returns"] += 1
            raise RuntimeError(
                f"duplicate waiting return request={request_id} epoch={epoch}"
            )
        tokens = control.get("output_token_ids")
        if not isinstance(tokens, list):
            raise TypeError("WAITING_RETURN requires output_token_ids")
        if phase == ChordRequestPhase.NEVER_RUN and tokens:
            raise ValueError("never-run WAITING_RETURN cannot contain output tokens")
        assignment = data["assignment"]
        if not hasattr(assignment, "set_result"):
            raise TypeError("waiting return requires an assignment future")

        record.state = ChordRequestState.WAITING_RETURNING
        record.resume_output_token_ids = [int(token) for token in tokens]
        record.last_progress_time = float(
            control.get("last_progress_time", record.last_progress_time)
        )
        record.num_preemptions = max(
            record.num_preemptions, int(control.get("num_preemptions", 0))
        )
        record.was_preempted = (
            record.was_preempted or phase == ChordRequestPhase.PREEMPTED
        )
        record.last_return_worker_id = compute_worker_id
        record.compute_worker_id = None
        record.stream_owner_worker_id = None
        self._remove_reservation(record.request_id, epoch)
        record.state = ChordRequestState.GLOBAL_WAITING
        record.assignment_waiter = assignment
        self._pool.add(record)
        if _ttft_diag is not None:
            _ttft_diag.emit("global_pool_enter", record.request_id, epoch=record.epoch)
        self.counters["waiting_returns"] += 1
        self.counters[f"waiting_returns_{phase.value}"] += 1
        self._trace.emit(
            "waiting_return",
            request_id=request_id,
            epoch=epoch,
            phase=phase.value,
            reason=reason,
            waiting_since=waiting_since,
            wait_age_s=wait_age_s,
            compute_worker_id=compute_worker_id,
            stream_owner_worker_id=owner_id,
            resume_tokens=len(tokens),
            num_preemptions=record.num_preemptions,
        )

    async def _ack_waiting_return(
        self, base_url: str, request_id: str, epoch: int
    ) -> None:
        try:
            await self._post(
                base_url,
                "chord_waiting_return_stored",
                {"request_id": request_id, "epoch": epoch},
            )
        except Exception:
            logger.warning(
                "Chord waiting-return ACK failed: request=%s epoch=%d worker=%s",
                request_id,
                epoch,
                base_url,
                exc_info=True,
            )

    def _handle_finished(self, data: dict[str, Any]) -> None:
        record = self._current_record(data)
        if record is None:
            return
        record.state = ChordRequestState.FINISHED
        self._pool.discard(record.request_id)
        self._clear_forbidden(record)
        self._remove_reservation(record.request_id)
        self._cancel_assignment(record)
        self._records.pop(record.request_id, None)
        self.counters["finished"] += 1
        self._trace.emit(
            "finished",
            request_id=record.request_id,
            epoch=record.epoch,
            reason=str(data.get("reason", "finished")),
        )

    def _handle_cancelled(self, data: dict[str, Any]) -> None:
        request_id = str(data["request_id"])
        record = self._records.pop(request_id, None)
        if record is None:
            return
        record.state = ChordRequestState.CANCELLED
        self._pool.discard(request_id)
        self._clear_forbidden(record)
        self._remove_reservation(request_id)
        self._cancel_assignment(record)
        self.counters["cancelled"] += 1
        self._trace.emit(
            "cancelled",
            request_id=request_id,
            epoch=record.epoch,
        )

    @staticmethod
    def _cancel_assignment(record: RequestRecord) -> None:
        waiter = record.assignment_waiter
        record.assignment_waiter = None
        if waiter is not None and not waiter.done():
            waiter.cancel()

    def _clear_forbidden(self, record: RequestRecord) -> None:
        for worker_id, epochs in record.forbidden_epochs.items():
            instance = self._instances.get(worker_id)
            if instance is None:
                continue
            instance.forbidden_dispatches.difference_update(
                (record.request_id, epoch) for epoch in epochs
            )
        record.forbidden_epochs.clear()

    def _remove_reservation(self, request_id: str, epoch: int | None = None) -> None:
        for instance in self._instances.values():
            for mapping in (
                instance.unobserved_dispatches,
                instance.migration_reservations,
            ):
                for key in tuple(mapping):
                    if key[0] == request_id and (epoch is None or key[1] == epoch):
                        mapping.pop(key, None)

    def _refresh_probe_view(self) -> None:
        by_url: dict[str, tuple[dict[str, Any], ...]] = {}
        for instance in self._instances.values():
            keys = set(instance.unobserved_dispatches)
            keys.update(instance.migration_reservations)
            keys.update(instance.forbidden_dispatches)
            by_url[instance.control_url] = tuple(
                {"request_id": request_id, "epoch": epoch}
                for request_id, epoch in sorted(keys)
            )
        self._probe_view = by_url

    def _migration_accounting_overlays(
        self,
        instances: list[InstanceState],
        *,
        now: float | None = None,
    ) -> tuple[dict[int, int], dict[int, int]]:
        now = time.monotonic() if now is None else now
        tokens_by_worker: dict[int, int] = {}
        requests_by_worker: dict[int, int] = {}
        for instance in instances:
            detailed = instance.detailed
            if detailed is None:
                continue
            overlay = instance.accounting_overlay(
                now=now,
                migration_timeout_s=self.llumnix_config.migration_reservation_timeout_s,
                block_size=detailed.block_size,
            )
            tokens = overlay["dispatch_tokens"] + overlay["migration_tokens"]
            if tokens:
                tokens_by_worker[instance.worker_id] = tokens
            if overlay["migration_requests"]:
                requests_by_worker[instance.worker_id] = overlay["migration_requests"]
        return tokens_by_worker, requests_by_worker

    def _maybe_start_migration(self) -> None:
        round_id = self._complete_detail_round
        if (
            not self.config.migration_enabled
            or self._migration_in_flight
            or (self.config.waiting_enabled and self._pool)
            or round_id <= self._last_migration_decision_round
        ):
            return
        instances = [
            item
            for item in self._instances.values()
            if item.connected
            and item.detail_round == round_id
            and item.latest_round == round_id
            and item.detailed is not None
            and not item.detailed.is_stale(self.config.status_stale_ms)
        ]
        if len(instances) < max(2, self._expected_workers()):
            return
        self._last_migration_decision_round = round_id
        reserved_tokens, reserved_requests = self._migration_accounting_overlays(
            instances
        )
        pairs = choose_migration_pairs(
            [item.detailed for item in instances if item.detailed is not None],
            self.llumnix_config,
            filter_stale=False,
            reserved_tokens_by_worker=reserved_tokens,
            reserved_migrate_in_by_worker=reserved_requests,
        )
        if not pairs:
            return

        launched = 0
        for pair in pairs:
            source = pair.source
            target = pair.target
            source_instance = self._instances.get(source.worker_id)
            if source_instance is None:
                continue
            body = {
                "target_worker_id": str(target.worker_id),
                "target_rpc_host": target.rpc_host,
                "target_rpc_port": target.rpc_port,
                "policy": self.llumnix_config.policy.value,
                "limit": self.llumnix_config.limit.value,
                "value": self.llumnix_config.limit_value,
                "max_requests": max(
                    0,
                    self.llumnix_config.max_migrate_in_requests
                    - target.num_migrate_in_reqs
                    - reserved_requests.get(target.worker_id, 0),
                ),
                "trigger_policy": "llumnix-load-balance",
            }
            key = (round_id, source.worker_id, target.worker_id)
            self._migration_in_flight.add(key)
            launched += 1
            self._spawn(
                self._run_migration(
                    round_id,
                    source_instance.control_url,
                    source.worker_id,
                    target.worker_id,
                    body,
                ),
                f"chord-migrate-{source.worker_id}-{target.worker_id}-{round_id}",
            )
        if launched:
            self._trace.emit(
                "running_migration_batch",
                round_id=round_id,
                pairs=launched,
            )

    async def _run_migration(
        self,
        round_id: int,
        source_url: str,
        source_worker_id: int,
        target_worker_id: int,
        body: dict[str, Any],
    ) -> None:
        try:
            result = await self._post(source_url, "llumnix_prepare_migration", body)
            error = None
        except Exception as caught:
            result = None
            error = caught
        await self._events.put(
            CoordinatorEvent(
                "migration_result",
                {
                    "round_id": round_id,
                    "source_worker_id": source_worker_id,
                    "target_worker_id": target_worker_id,
                    "result": result,
                    "error": error,
                },
            )
        )

    def _handle_migration_result(self, data: dict[str, Any]) -> None:
        round_id = int(data.get("round_id", -1))
        source_worker_id = int(data["source_worker_id"])
        target_worker_id = int(data["target_worker_id"])
        self._migration_in_flight.discard(
            (round_id, source_worker_id, target_worker_id)
        )
        error = data.get("error")
        if error is not None:
            self.counters["migration_errors"] += 1
            self._trace.emit(
                "running_migration_error",
                round_id=round_id,
                source_worker_id=source_worker_id,
                target_worker_id=target_worker_id,
                error=repr(error),
            )
            return

        result = data.get("result") or {}
        migrations = result.get("migrations") or ()
        target = self._instances.get(target_worker_id)
        for migration in migrations:
            backend_request_id = str(migration["request_id"])
            output_tokens_hint = max(
                int(migration.get("output_tokens", 0)),
                int(migration.get("output_tokens_hint", 0)),
            )
            identity = split_backend_request_id(backend_request_id)
            if identity is None:
                self.counters["migration_identity_mismatches"] += 1
                continue
            request_id, migration_epoch = identity
            record = self._records.get(request_id)
            if (
                record is None
                or record.epoch != migration_epoch
                or record.compute_worker_id != source_worker_id
                or record.state
                in {ChordRequestState.FINISHED, ChordRequestState.CANCELLED}
            ):
                continue

            record.state = ChordRequestState.MIGRATING
            if target is not None:
                target.migration_reservations[(request_id, record.epoch)] = Reservation(
                    request_id=request_id,
                    epoch=record.epoch,
                    blocks=record.block_demand(
                        self.block_size,
                        output_tokens_hint=output_tokens_hint,
                    ),
                    baseline_step_id=target.compact.step_id if target.compact else -1,
                    created_at=time.monotonic(),
                )
        self.counters["accepted_migrations" if self.config.method == "llumnix-fcwsr"
                      else "running_migrations"] += len(migrations)
        self._trace.emit(
            "migration_accepted" if self.config.method == "llumnix-fcwsr" else "running_migration",
            source_worker_id=source_worker_id,
            target_worker_id=target_worker_id,
            migrations=len(migrations),
            rejected=len(result.get("rejected_request_ids") or ()),
        )

    def _handle_migration_committed(self, data: dict[str, Any]) -> None:
        identity = split_backend_request_id(str(data["backend_request_id"]))
        if identity is None:
            raise ValueError("Committed Chord migration lacks an epoch")
        request_id, epoch = identity
        record = self._records.get(request_id)
        if record is None or record.epoch != epoch:
            self.counters["stale_migration_commits"] += 1
            return
        source, target = int(data["source_worker_id"]), int(data["target_worker_id"])
        if record.compute_worker_id != source:
            if self.config.method == "llumnix-fcwsr" and record.compute_worker_id == target:
                self.counters["duplicate_migration_commits"] += 1
                return
            raise ValueError(f"Migration source does not own {request_id}:{epoch}")
        source_instance = self._instances.get(source)
        if source_instance is not None:
            source_instance.unobserved_dispatches.pop((request_id, epoch), None)
        record.compute_worker_id = target
        record.state = ChordRequestState.RUNNING
        self.counters["committed_migrations" if self.config.method == "llumnix-fcwsr"
                      else "committed_running_migrations"] += 1
        self._trace.emit("migration_committed" if self.config.method == "llumnix-fcwsr"
                         else "running_migration_committed", request_id=request_id, epoch=epoch,
                         source_worker_id=source, target_worker_id=target)
