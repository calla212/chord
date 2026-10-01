from __future__ import annotations

import json
import os
import queue
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable


TRACE_SCHEMA_VERSION = 2
TRACE_EVENT_NAME = "scheduler_iteration"
ENV_TRACE_PATH = "DYN_SCHED_ITER_TRACE_PATH"
ENV_TRACE_WORKER_ID = "DYN_SCHED_ITER_WORKER_ID"


def summarize_scheduler_output(
    output: Any,
    *,
    new_request_is_decode: Callable[[str], bool] | None = None,
) -> dict[str, int]:


    is_decode = new_request_is_decode or (lambda _request_id: False)
    num_scheduled = output.num_scheduled_tokens
    num_prefill_requests = 0
    sum_prefill_tokens = 0
    num_decode_requests = 0
    sum_decode_tokens = 0

    for request in output.scheduled_new_reqs:
        tokens = int(num_scheduled.get(request.req_id, 0))
        if is_decode(request.req_id):
            num_decode_requests += 1
            sum_decode_tokens += tokens
        else:
            num_prefill_requests += 1
            sum_prefill_tokens += tokens

    cached = output.scheduled_cached_reqs
    for request_id in cached.req_ids:
        tokens = int(num_scheduled.get(request_id, 0))
        if cached.is_context_phase(request_id):
            num_prefill_requests += 1
            sum_prefill_tokens += tokens
        else:
            num_decode_requests += 1
            sum_decode_tokens += tokens

    return {
        "num_prefill_requests": num_prefill_requests,
        "sum_prefill_tokens": sum_prefill_tokens,
        "num_decode_requests": num_decode_requests,
        "sum_decode_tokens": sum_decode_tokens,
        "num_preempted_requests": len(set(output.preempted_req_ids or ())),
        "total_num_scheduled_tokens": int(output.total_num_scheduled_tokens),
    }


def summarize_scheduler_state(scheduler: Any) -> dict[str, int | None]:


    try:
        free_kv_blocks = int(
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        )
    except (AttributeError, TypeError, ValueError):
        free_kv_blocks = None

    return {
        "num_waiting_requests": len(scheduler.waiting),
        "num_skipped_waiting_requests": len(scheduler.skipped_waiting),
        "num_running_requests": len(scheduler.running),
        "num_tracked_requests": len(scheduler.requests),
        "num_free_kv_blocks": free_kv_blocks,
    }


class SchedulerIterationTraceSink:


    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        worker_id: str,
        dp_rank: int,
        max_queue_size: int = 100_000,
    ) -> None:
        self._path = Path(path)
        self._worker_id = worker_id
        self._dp_rank = int(dp_rank)
        self._hostname = socket.gethostname()
        self._process_id = os.getpid()
        self._dropped = 0
        self._closed = False
        self._queue: queue.Queue[dict[str, Any] | None] = queue.Queue(
            maxsize=max_queue_size
        )
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._thread = threading.Thread(
            target=self._writer,
            daemon=True,
            name="scheduler-iteration-trace",
        )
        self._thread.start()

    @property
    def dropped(self) -> int:
        return self._dropped

    def emit(
        self,
        *,
        iteration_id: int,
        monotonic_ns: int,
        schedule_duration_ns: int,
        counts: dict[str, int],
    ) -> None:
        if self._closed:
            return
        record = {
            "schema_version": TRACE_SCHEMA_VERSION,
            "event_name": TRACE_EVENT_NAME,
            "worker_id": self._worker_id,
            "dp_rank": self._dp_rank,
            "hostname": self._hostname,
            "process_id": self._process_id,
            "iteration_id": int(iteration_id),
            "monotonic_ns": int(monotonic_ns),
            "realtime_ns": time.time_ns(),
            "schedule_duration_ns": int(schedule_duration_ns),
            "trace_drop_count": self._dropped,
            **counts,
        }
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            self._dropped += 1

    def close(self, timeout: float = 2.0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put(None, timeout=timeout)
        except queue.Full:
            return
        self._thread.join(timeout=timeout)

    def _writer(self) -> None:
        with self._path.open("a", encoding="utf-8", buffering=1) as handle:
            while True:
                record = self._queue.get()
                if record is None:
                    break
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
