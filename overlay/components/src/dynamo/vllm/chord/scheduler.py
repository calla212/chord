from __future__ import annotations

import heapq
import logging
import sys
import time
from collections import Counter
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request, RequestStatus

from .config import ChordConfig
from .protocol import make_backend_request_id, split_backend_request_id
from .types import ChordRequestPhase


logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WaitingReturn:
    request_id: str
    epoch: int
    client_index: int
    phase: ChordRequestPhase
    reason: str
    waiting_since: float
    wait_age_s: float
    num_preemptions: int


@dataclass(frozen=True, slots=True)
class _PendingWaitingReturn:
    request: Request
    phase: ChordRequestPhase
    reason: str
    waiting_since: float
    wait_age_s: float


class ChordUrgencyRequestQueue(RequestQueue):


    def __init__(self, adapter: ChordSchedulerAdapter) -> None:
        self._adapter = adapter
        self._never_run: list[tuple[float, float, str, int, Request]] = []
        self._preempted: list[tuple[float, float, str, int, Request]] = []
        self._active: dict[str, tuple[int, Request, ChordRequestPhase]] = {}
        self._version = 0
        self._now = time.time()

    def set_now(self, now: float) -> None:
        self._now = now

    def add_request(self, request: Request) -> None:
        self._version += 1
        version = self._version
        phase = self._adapter.priority_phase(request)
        self._active[request.request_id] = (version, request, phase)
        baseline = (
            float(request.arrival_time)
            if phase == ChordRequestPhase.NEVER_RUN
            else float(
                request.chord_last_progress_time
                if request.chord_last_progress_time is not None
                else request.arrival_time
            )
        )
        entry = (
            baseline,
            float(request.arrival_time),
            request.request_id,
            version,
            request,
        )
        heapq.heappush(
            self._preempted
            if phase == ChordRequestPhase.PREEMPTED
            else self._never_run,
            entry,
        )

    def _valid(
        self,
        entry: tuple[float, float, str, int, Request],
        phase: ChordRequestPhase,
    ) -> bool:
        _, _, request_id, version, request = entry
        current = self._active.get(request_id)
        return current == (version, request, phase)

    def _clean(
        self,
        heap: list[tuple[float, float, str, int, Request]],
        phase: ChordRequestPhase,
    ) -> None:
        while heap and not self._valid(heap[0], phase):
            heapq.heappop(heap)

    def _phase_head(self, phase: ChordRequestPhase) -> Request | None:
        heap = (
            self._preempted if phase == ChordRequestPhase.PREEMPTED else self._never_run
        )
        self._clean(heap, phase)
        return heap[0][4] if heap else None

    def peek_request(self) -> Request:
        never_run = self._phase_head(ChordRequestPhase.NEVER_RUN)
        preempted = self._phase_head(ChordRequestPhase.PREEMPTED)
        if never_run is None and preempted is None:
            raise IndexError("peek from an empty Chord queue")
        if never_run is None:
            assert preempted is not None
            return preempted
        if preempted is None:
            return never_run
        return (
            preempted
            if self._adapter.prefer(preempted, never_run, self._now)
            else never_run
        )

    def refresh(self, request: Request) -> None:
        if request.request_id in self._active:
            self.add_request(request)

    def pop_request(self) -> Request:
        request = self.peek_request()
        self._active.pop(request.request_id, None)
        return request

    def prepend_request(self, request: Request) -> None:
        self.add_request(request)

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.add_request(request)

    def remove_request(self, request: Request) -> None:
        current = self._active.get(request.request_id)
        if current is None or current[1] is not request:
            raise ValueError(f"request is not queued: {request.request_id}")
        self._active.pop(request.request_id, None)

    def remove_requests(self, requests: Iterable[Request]) -> None:
        for request in requests:
            current = self._active.get(request.request_id)
            if current is not None and current[1] is request:
                self._active.pop(request.request_id, None)

    def __bool__(self) -> bool:
        return bool(self._active)

    def __len__(self) -> int:
        return len(self._active)

    def __iter__(self) -> Iterator[Request]:
        yield from sorted(
            (value[1] for value in self._active.values()),
            key=lambda request: self._adapter.priority_key(request, self._now),
        )


class ChordSchedulerAdapter:


    def __init__(self, scheduler: Any, config: ChordConfig) -> None:
        self.scheduler = scheduler
        self.config = config
        self.epochs: dict[str, int] = {}
        self.inflight_refs: Counter[str] = Counter()
        self.waiting_since: dict[str, float] = {}
        self.waiting_return_pending: dict[str, _PendingWaitingReturn] = {}
        self.counters: Counter[str] = Counter()
        self.schedule_now = time.time()
        if self.config.waiting_enabled:
            self._install_priority_queues()
        logger.info(
            "Chord worker implementation scheduler=%s request=%s",
            __file__,
            sys.modules[Request.__module__].__file__,
        )

    def _install_priority_queues(self) -> None:
        old_waiting = tuple(self.scheduler.waiting)
        old_skipped = tuple(self.scheduler.skipped_waiting)
        waiting = ChordUrgencyRequestQueue(self)
        skipped = ChordUrgencyRequestQueue(self)
        for request in old_waiting:
            waiting.add_request(request)
        for request in old_skipped:
            skipped.add_request(request)
        self.scheduler.waiting = waiting
        self.scheduler.skipped_waiting = skipped

    @staticmethod
    def _managed(request: Request) -> bool:
        return request.chord_last_progress_time is not None

    @staticmethod
    def phase(request: Request) -> ChordRequestPhase:
        if (
            request.status == RequestStatus.PREEMPTED
            or request.num_preemptions > 0
            or request.num_output_tokens > 0
        ):
            return ChordRequestPhase.PREEMPTED
        return ChordRequestPhase.NEVER_RUN

    def priority_phase(self, request: Request) -> ChordRequestPhase:
        if self.config.reference_policy:
            return self.phase(request)
        return (ChordRequestPhase.PREEMPTED if request.num_output_tokens > 0
                else ChordRequestPhase.NEVER_RUN)

    def priority_key(self, request: Request, now: float) -> tuple[float, float, str]:
        phase = self.priority_phase(request)
        baseline = (
            float(request.arrival_time)
            if phase == ChordRequestPhase.NEVER_RUN
            else float(
                request.chord_last_progress_time
                if request.chord_last_progress_time is not None
                else request.arrival_time
            )
        )
        coefficient = (
            1.0
            if phase == ChordRequestPhase.NEVER_RUN
            else self.config.urgency_coefficient
        )
        urgency = coefficient * max(0.0, now - baseline)
        return -urgency, float(request.arrival_time), request.request_id

    def prefer(self, left: Request, right: Request, now: float | None = None) -> bool:
        frozen = self.schedule_now if now is None else now
        return self.priority_key(left, frozen) < self.priority_key(right, frozen)

    def begin_schedule(self) -> float:
        self.schedule_now = time.time()
        for queue in (self.scheduler.waiting, self.scheduler.skipped_waiting):
            if isinstance(queue, ChordUrgencyRequestQueue):
                queue.set_now(self.schedule_now)
        return self.schedule_now

    def select_waiting_queue(self) -> RequestQueue | None:
        waiting = self.scheduler.waiting
        skipped = self.scheduler.skipped_waiting
        if waiting and skipped:
            return (
                waiting
                if self.prefer(
                    waiting.peek_request(),
                    skipped.peek_request(),
                    self.schedule_now,
                )
                else skipped
            )
        return waiting or skipped or None

    def _queued_requests(self) -> list[Request]:
        values: dict[str, Request] = {}
        for request in (*self.scheduler.skipped_waiting, *self.scheduler.waiting):
            values[request.request_id] = request
        return sorted(
            values.values(),
            key=lambda request: self.priority_key(request, self.schedule_now),
        )

    def _connector_requests(self) -> tuple[list[Request], list[Request]]:
        adapter = getattr(self.scheduler, "_llumnix_status_adapter", None)
        if adapter is None:
            return [], []
        pending, loading = adapter._connector_state()
        return [item[0] for item in pending], loading

    @staticmethod
    def _canonical_identity(request: Request) -> tuple[str, int]:
        epoch = int(request.chord_request_version)
        identity = split_backend_request_id(request.request_id)
        if identity is None or identity[1] != epoch:
            return request.request_id, epoch
        return identity

    def _find_epoch_request(
        self,
        request_id: str,
        epoch: int,
        connector_requests: tuple[list[Request], list[Request]] | None = None,
    ) -> tuple[Request | None, str, bool]:
        backend_request_id = make_backend_request_id(request_id, epoch)
        candidate_ids = (backend_request_id, request_id)
        for candidate_id in candidate_ids:
            request = self.scheduler.requests.get(candidate_id)
            if request is not None:
                return request, candidate_id, True

        if connector_requests is None:
            pending, loading = self._connector_requests()
        else:
            pending, loading = connector_requests
        for candidate_id in candidate_ids:
            request = next(
                (
                    item
                    for item in (*pending, *loading)
                    if item.request_id == candidate_id
                ),
                None,
            )
            if request is not None:
                return request, candidate_id, False
        return None, backend_request_id, False

    def _remove_from_waiting(self, request: Request) -> None:
        self.scheduler.waiting.remove_requests((request,))
        self.scheduler.skipped_waiting.remove_requests((request,))

    def on_request_added(self, request: Request) -> None:
        if not self._managed(request):
            return
        self.epochs[request.request_id] = int(request.chord_request_version)
        if request.status in {RequestStatus.WAITING, RequestStatus.PREEMPTED}:
            self.waiting_since[request.request_id] = time.time()
        if request.num_output_tokens > 0:
            request.prefill_stats = None
            self.counters["resumed_prefill_stats_suppressed"] += 1
        self.counters["request_added"] += 1

    def _eligible_for_waiting_return(
        self,
        request: Request,
        connector_ids: set[str],
    ) -> bool:
        return (
            self._managed(request)
            and request.request_id not in self.waiting_return_pending
            and request.request_id not in connector_ids
            and not self._llumnix_migrating(request.request_id)
            and request.status in {RequestStatus.WAITING, RequestStatus.PREEMPTED}
        )

    def on_schedule(self, output: Any) -> None:


        self.counters["schedule_updates"] += 1
        now = self.schedule_now
        scheduled_ids = set(output.num_scheduled_tokens)
        preempted_ids = set(output.preempted_req_ids or ())

        for request_id in scheduled_ids:
            request = self.scheduler.requests.get(request_id)
            if request is None or not self._managed(request):
                continue
            self.inflight_refs[request_id] += 1
            if request_id in self.waiting_since:
                phase = self.phase(request)
                self.counters[
                    "local_preempted_resumes"
                    if phase == ChordRequestPhase.PREEMPTED
                    else "local_initial_admissions"
                ] += 1
            self.waiting_since.pop(request_id, None)

        for request_id in preempted_ids:
            request = self.scheduler.requests.get(request_id)
            if request is None or not self._managed(request):
                continue
            self.waiting_since[request_id] = now
            self.counters["native_preemptions_retained"] += 1

        pending_connector, loading_connector = self._connector_requests()
        connector_ids = {
            request.request_id for request in (*pending_connector, *loading_connector)
        }
        queued = self._queued_requests()
        queued_ids = {request.request_id for request in queued}
        for request in queued:
            request_id = request.request_id
            if self._eligible_for_waiting_return(request, connector_ids):
                self.waiting_since.setdefault(request_id, now)
            elif request_id not in self.waiting_return_pending:


                self.waiting_since.pop(request_id, None)

        for request in queued if self.config.waiting_enabled else ():
            request_id = request.request_id
            if not self._eligible_for_waiting_return(request, connector_ids):
                continue
            waiting_since = self.waiting_since[request_id]
            wait_age_s = max(0.0, now - waiting_since)
            if wait_age_s < self.config.waiting_return_timeout_s:
                continue
            phase = self.phase(request)
            self._remove_from_waiting(request)
            self.waiting_return_pending[request_id] = _PendingWaitingReturn(
                request=request,
                phase=phase,
                reason="waiting_timeout",
                waiting_since=waiting_since,
                wait_age_s=wait_age_s,
            )
            self.counters["waiting_returns_captured"] += 1
            self.counters[f"waiting_returns_{phase.value}"] += 1

        active_ids = set(self.scheduler.requests)
        for request_id in tuple(self.waiting_since):
            if (
                request_id not in queued_ids
                and request_id not in active_ids
                and request_id not in self.waiting_return_pending
            ):
                self.waiting_since.pop(request_id, None)

    def complete_batch(self, output: Any) -> list[WaitingReturn]:


        for request_id in output.num_scheduled_tokens:
            if self.inflight_refs[request_id] <= 1:
                self.inflight_refs.pop(request_id, None)
            else:
                self.inflight_refs[request_id] -= 1

        retired = [
            request_id
            for request_id in output.num_scheduled_tokens
            if request_id not in self.scheduler.requests
            and request_id not in self.waiting_return_pending
        ]
        self.forget_requests(retired)

        ready: list[WaitingReturn] = []
        for request_id, pending in tuple(self.waiting_return_pending.items()):
            if self.inflight_refs.get(request_id, 0) > 0:
                continue
            self.waiting_return_pending.pop(request_id, None)
            request = pending.request
            ready.append(
                WaitingReturn(
                    request_id=request_id,
                    epoch=self.epochs.get(
                        request_id, int(request.chord_request_version)
                    ),
                    client_index=int(request.client_index),
                    phase=pending.phase,
                    reason=pending.reason,
                    waiting_since=pending.waiting_since,
                    wait_age_s=pending.wait_age_s,
                    num_preemptions=int(request.num_preemptions),
                )
            )
        self.counters["waiting_returns_ready"] += len(ready)
        return ready

    def _llumnix_migrating(self, request_id: str) -> bool:
        adapter = getattr(self.scheduler, "_llumnix_status_adapter", None)
        return bool(adapter is not None and request_id in adapter.migrating_out)

    def snapshot_output_counts(self) -> dict[str, int]:
        return {
            request.request_id: request.num_output_tokens
            for request in self.scheduler.running
            if self._managed(request)
        }

    def update_progress(self, before: dict[str, int]) -> None:
        now = time.time()
        for request in self.scheduler.requests.values():
            previous = before.get(request.request_id)
            if previous is not None and request.num_output_tokens > previous:
                request.chord_last_progress_time = now
                if self.config.waiting_enabled:
                    self.scheduler.waiting.refresh(request)
                    self.scheduler.skipped_waiting.refresh(request)

    def _probe_state(self, request_id: str, epoch: int) -> str:
        pending, loading = self._connector_requests()
        request, backend_request_id, _ = self._find_epoch_request(
            request_id, epoch, (pending, loading)
        )
        if request is None or int(request.chord_request_version) != epoch:
            return "absent"
        if backend_request_id in self.waiting_return_pending:
            return "waiting_returning"
        if request in self.scheduler.running:
            return "running"
        if request in loading:
            return "loading"
        if request in pending:
            return "connector_waiting"
        if (
            request in self.scheduler.waiting
            or request in self.scheduler.skipped_waiting
        ):
            return "waiting"
        return "active"

    def collect(self, body: dict[str, Any]) -> dict[str, Any]:


        include_migration = bool(body.get("include_migration", False))
        self.counters["status_collections"] += 1
        if include_migration:
            self.counters["migration_detail_collections"] += 1
        llumnix = getattr(self.scheduler, "_llumnix_status_adapter", None)
        if llumnix is None:
            raise RuntimeError("Chord requires Llumnix status projection")
        status = llumnix.collect(include_requests=include_migration)
        block_size = int(status.block_size)
        total_blocks = status.num_total_gpu_tokens // block_size

        now = time.time()
        waiting = [
            request
            for request in self._queued_requests()
            if request.request_id not in self.waiting_return_pending
            and self._managed(request)
        ]
        pending_connector, loading_connector = self._connector_requests()
        connector_ids = {
            request.request_id for request in (*pending_connector, *loading_connector)
        }
        ordinary_waiting = [
            request
            for request in waiting
            if self._eligible_for_waiting_return(request, connector_ids)
        ]
        waiting_entries = []
        wait_ages = []
        for request in ordinary_waiting:
            request_id, epoch = self._canonical_identity(request)
            waiting_since = self.waiting_since.get(request.request_id, 0.0)
            if waiting_since > 0:
                wait_ages.append(max(0.0, now - waiting_since))
            if include_migration or self.config.reference_policy:
                waiting_entries.append(
                    {
                        "request_id": request_id,
                        "epoch": epoch,
                        "phase": self.phase(request).value,
                        "arrival_time": float(request.arrival_time),
                        "last_progress_time": float(
                            request.chord_last_progress_time or request.arrival_time
                        ),
                        "waiting_since": waiting_since,
                    }
                )

        probes = []
        for item in body.get("probes", ()):
            try:
                request_id = str(item["request_id"])
                epoch = int(item["epoch"])
            except (KeyError, TypeError, ValueError):
                continue
            probes.append(
                {
                    "request_id": request_id,
                    "epoch": epoch,
                    "state": self._probe_state(request_id, epoch),
                }
            )

        value = {
            "status": "ok",
            "runtime": {
                "method": self.config.method,
                "scheduler": type(self.scheduler).__name__,
                "local_state": type(self).__name__,
                "migration_state": type(llumnix).__name__,
                "connector": type(self.scheduler.get_kv_connector()).__name__,
                "waiting_queue": type(self.scheduler.waiting).__name__,
                "source": __file__,
            },
            "policy": {
                "dispatch_threshold_s": self.config.dispatch_threshold_s,
                "waiting_return_timeout_s": self.config.waiting_return_timeout_s,
                "urgency_coefficient": self.config.urgency_coefficient,
                "skip_unfit": self.config.skip_unfit or self.config.dispatch_threshold_s == -1,
            },
            "worker_id": self.config.worker_id,
            "round_id": int(body.get("round_id", 0)),
            "schedulable": True,
            "block_size": block_size,
            "total_blocks": total_blocks,
            "actual_used_blocks": status.num_used_gpu_tokens // block_size,
            "projected_tokens": status.projected_tokens,
            "step_id": status.step_id,
            "timestamp_ms": status.timestamp_ms,
            "all_prefills_tokens_num": status.all_prefills_tokens_num,
            "decode_batch_size": status.decode_batch_size,
            "all_decode_tokens": status.all_decode_tokens,
            "num_connector_waiting_requests": len(pending_connector),
            "num_running_requests": status.num_running_requests,
            "num_waiting_requests": len(waiting),
            "num_ordinary_waiting_requests": len(ordinary_waiting),
            "num_waiting_return_pending": len(self.waiting_return_pending),
            "num_loading_requests": status.num_loading_requests,
            "num_migrate_in_reqs": status.num_migrate_in_reqs,
            "num_migrate_out_reqs": status.num_migrate_out_reqs,
            "max_waiting_age_s": max(wait_ages, default=0.0),
            "mean_waiting_age_s": (
                sum(wait_ages) / len(wait_ages) if wait_ages else 0.0
            ),
            "waiting": waiting_entries,
            "probes": probes,
            "counters": dict(self.counters),
        }
        if include_migration:
            value["llumnix"] = status.to_dict()
        return value

    def drop_epoch(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            request_id = str(body["request_id"])
            epoch = int(body["epoch"])
        except (KeyError, TypeError, ValueError) as error:
            return {"status": "error", "message": f"invalid epoch drop: {error}"}

        request, backend_request_id, native = self._find_epoch_request(
            request_id, epoch
        )
        if request is None:
            return {"status": "absent", "request_id": request_id, "epoch": epoch}
        current = int(request.chord_request_version)
        if current != epoch:
            return {
                "status": "stale",
                "request_id": request_id,
                "epoch": epoch,
                "current_epoch": current,
            }
        if native:
            self.scheduler._llumnix_finish_requests_native(backend_request_id)
        else:
            self.scheduler.finish_requests(
                backend_request_id, RequestStatus.FINISHED_ABORTED
            )
        self.forget_requests((backend_request_id,))
        self.counters["epochs_dropped"] += 1
        return {"status": "ok", "request_id": request_id, "epoch": epoch}

    def forget_requests(self, request_ids: Iterable[str]) -> None:
        for request_id in request_ids:
            self.epochs.pop(request_id, None)
            self.inflight_refs.pop(request_id, None)
            self.waiting_since.pop(request_id, None)
            self.waiting_return_pending.pop(request_id, None)
