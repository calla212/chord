from __future__ import annotations

import asyncio
import contextlib
import copy
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from dynamo._core import Context

from .types import MigrationDecision, MigrationPolicy
from .audit import migration_audit

if TYPE_CHECKING:
    from dynamo.runtime import Endpoint

logger = logging.getLogger(__name__)

MIGRATION_METADATA_KEY = "dynamo_llumnix_migration"
TRANSPORT_REQUEST_ID_METADATA_KEY = "dynamo.llumnix.request_id"
_STREAM_END = object()
_TARGET_DISCOVERY_TIMEOUT_S = 2.0
_TARGET_DISCOVERY_POLL_S = 0.01


@dataclass(frozen=True, slots=True)
class MigrationEnvelope:


    request_id: str
    source_worker_id: int
    target_worker_id: int
    source_rpc_host: str
    source_rpc_port: int
    output_tokens: int
    output_tokens_hint: int
    policy: MigrationPolicy
    trigger_policy: str
    source_phase: str = "unknown"
    source_computed_tokens: int = 0

    @classmethod
    def from_decision(cls, decision: MigrationDecision) -> MigrationEnvelope:
        return cls(**asdict(decision))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["policy"] = self.policy.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> MigrationEnvelope:
        return cls(
            request_id=str(value["request_id"]),
            source_worker_id=int(value["source_worker_id"]),
            target_worker_id=int(value["target_worker_id"]),
            source_rpc_host=str(value["source_rpc_host"]),
            source_rpc_port=int(value["source_rpc_port"]),
            output_tokens=int(value.get("output_tokens", 0)),
            output_tokens_hint=int(value.get("output_tokens_hint", 0)),
            policy=MigrationPolicy(str(value.get("policy", "SR")).upper()),
            trigger_policy=str(value.get("trigger_policy", "llumnix-load-balance")),
            source_phase=str(value.get("source_phase", "unknown")),
            source_computed_tokens=int(value.get("source_computed_tokens", 0)),
        )


def migration_envelope(request: dict[str, Any]) -> MigrationEnvelope | None:
    extra_args = request.get("extra_args")
    if not isinstance(extra_args, dict):
        return None
    value = extra_args.get(MIGRATION_METADATA_KEY)
    if not isinstance(value, dict):
        return None
    return MigrationEnvelope.from_dict(value)


def apply_resume_metadata(request: dict[str, Any], sampling_params: Any) -> None:


    if sampling_params.extra_args is None:
        sampling_params.extra_args = {}
    kv_params = dict(sampling_params.extra_args.get("kv_transfer_params") or {})


    kv_params["ali_llumnix_disagg"] = False

    envelope = migration_envelope(request)
    if envelope is not None:
        from blade_kvt.hybrid_connector import PREALLOC_KEY
        from blade_kvt.hybrid_connector.migration import (
            MIGRATION_TRIGGER_POLICY,
            OUTPUT_TOKENS_N,
            SRC_INFO,
        )

        kv_params.update(
            {
                SRC_INFO: (envelope.source_rpc_host, envelope.source_rpc_port),
                OUTPUT_TOKENS_N: envelope.output_tokens,
                PREALLOC_KEY: envelope.output_tokens_hint,
                MIGRATION_TRIGGER_POLICY: envelope.trigger_policy,
            }
        )
    sampling_params.extra_args["kv_transfer_params"] = kv_params


def _detached_context(request_id: str, source: Context) -> Context:


    source_metadata = getattr(source, "metadata", None)
    metadata = source_metadata.copy() if source_metadata is not None else {}
    metadata[TRANSPORT_REQUEST_ID_METADATA_KEY] = request_id
    transport_id = f"{request_id}.llumnix.{uuid4().hex}"
    return Context(id=transport_id, metadata=metadata)


@dataclass(slots=True)
class _RequestEntry:
    request_id: str
    request: dict[str, Any]
    target_context: Context
    envelope: MigrationEnvelope | None = None
    task: asyncio.Task[None] | None = None
    queue: asyncio.Queue[Any] | None = None
    error: BaseException | None = None


class MigrationHandoffRegistry:


    def __init__(
        self,
        rollback: Callable[[str], Awaitable[None]],
        *,
        queue_size: int = 256,
    ) -> None:
        self._entries: dict[str, _RequestEntry] = {}
        self._rollback = rollback
        self._queue_size = queue_size
        self._client: Any | None = None
        self._client_lock = asyncio.Lock()

    async def _endpoint_client(self, endpoint: Endpoint) -> Any:
        if self._client is not None:
            return self._client
        async with self._client_lock:
            if self._client is None:
                self._client = await endpoint.client()
        return self._client

    @staticmethod
    async def _wait_for_target_client(
        client: Any,
        target_worker_id: int,
        *,
        timeout_s: float = _TARGET_DISCOVERY_TIMEOUT_S,
    ) -> None:


        initial_ids = {int(item) for item in await client.wait_for_instances()}
        if target_worker_id in initial_ids:
            return

        instance_ids = getattr(client, "instance_ids", None)
        if not callable(instance_ids):
            raise RuntimeError(
                "Dynamo client cannot inspect target discovery state: "
                f"target_worker_id={target_worker_id} initial_ids={sorted(initial_ids)}"
            )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + timeout_s
        observed_ids = initial_ids
        while loop.time() < deadline:
            await asyncio.sleep(_TARGET_DISCOVERY_POLL_S)
            observed_ids = {int(item) for item in instance_ids()}
            if target_worker_id in observed_ids:
                return
        raise TimeoutError(
            "Llumnix migration target was not discovered before timeout: "
            f"target_worker_id={target_worker_id} observed_ids={sorted(observed_ids)}"
        )

    async def wait_for_target(
        self,
        target_worker_id: int,
        endpoint: Endpoint,
        *,
        timeout_s: float = _TARGET_DISCOVERY_TIMEOUT_S,
    ) -> None:
        client = await self._endpoint_client(endpoint)
        await self._wait_for_target_client(
            client, target_worker_id, timeout_s=timeout_s
        )

    def register(
        self, request_id: str, request: dict[str, Any], context: Context
    ) -> _RequestEntry:
        if request_id in self._entries:
            raise RuntimeError(f"duplicate active request ID: {request_id}")
        entry = _RequestEntry(
            request_id=request_id,
            request=copy.deepcopy(request),
            target_context=_detached_context(request_id, context),
        )
        self._entries[request_id] = entry
        return entry

    def _retire_local(self, entry: _RequestEntry) -> None:
        if self._entries.get(entry.request_id) is entry:
            self._entries.pop(entry.request_id)

    def start(
        self,
        decision: MigrationDecision,
        endpoint: Endpoint,
    ) -> bool:
        entry = self._entries.get(decision.request_id)
        if entry is None or entry.task is not None:
            return False

        envelope = MigrationEnvelope.from_decision(decision)
        request = copy.deepcopy(entry.request)
        extra_args = dict(request.get("extra_args") or {})
        extra_args[MIGRATION_METADATA_KEY] = envelope.to_dict()
        request["extra_args"] = extra_args

        entry.envelope = envelope
        entry.queue = asyncio.Queue(maxsize=self._queue_size)
        entry.task = asyncio.create_task(
            self._pump_target(entry, request, endpoint),
            name=f"llumnix-handoff-{decision.request_id}",
        )
        return True

    @staticmethod
    def active(entry: _RequestEntry) -> bool:
        return entry.task is not None

    async def _pump_target(
        self,
        entry: _RequestEntry,
        request: dict[str, Any],
        endpoint: Endpoint,
    ) -> None:
        assert entry.envelope is not None
        assert entry.queue is not None
        cancelled = False
        try:
            client = await self._endpoint_client(endpoint)
            await self._wait_for_target_client(
                client, entry.envelope.target_worker_id
            )
            stream = await client.direct(
                request,
                entry.envelope.target_worker_id,
                annotated=False,
                context=entry.target_context,
            )
            async for chunk in stream:
                await entry.queue.put(chunk)
        except asyncio.CancelledError:
            cancelled = True
            migration_audit("target_cancelled", entry.envelope)
            raise
        except BaseException as error:
            entry.error = error
            migration_audit("target_failed", entry.envelope, error=repr(error))
            logger.exception(
                "Llumnix target handoff failed: request_id=%s target_worker_id=%d",
                entry.envelope.request_id,
                entry.envelope.target_worker_id,
            )


            if self._entries.get(entry.request_id) is entry:
                with contextlib.suppress(Exception):
                    await self._rollback(entry.request_id)
                    migration_audit("rollback_requested", entry.envelope)
        finally:
            if cancelled:


                if entry.queue.full():
                    with contextlib.suppress(asyncio.QueueEmpty):
                        entry.queue.get_nowait()
                entry.queue.put_nowait(_STREAM_END)
            else:
                await entry.queue.put(_STREAM_END)

    async def drain(
        self, entry: _RequestEntry, source_output_tokens: int
    ) -> AsyncIterator[dict[str, Any]]:


        self._retire_local(entry)
        if entry.queue is None:
            return

        committed = False
        while True:
            item = await entry.queue.get()
            if item is _STREAM_END:
                break
            if not isinstance(item, dict):
                raise TypeError(
                    f"target handoff returned {type(item).__name__}, expected dict"
                )
            chunk = dict(item)


            if not committed and chunk.get("token_ids"):
                assert entry.envelope is not None
                extra = dict(chunk.get("extra_args") or {})
                commits = list(extra.get("dynamo_migration_committed") or ())
                commits.insert(0, {
                    "source_worker_id": entry.envelope.source_worker_id,
                    "target_worker_id": entry.envelope.target_worker_id,
                })
                extra["dynamo_migration_committed"] = commits
                chunk["extra_args"] = extra
                logger.info(
                    "Llumnix migration committed: request_id=%s source_worker_id=%d target_worker_id=%d",
                    entry.request_id, entry.envelope.source_worker_id,
                    entry.envelope.target_worker_id,
                )
                migration_audit("committed", entry.envelope, source_output_tokens=source_output_tokens)
                committed = True
            usage = chunk.get("completion_usage")
            if isinstance(usage, dict):
                usage = dict(usage)
                completion = int(usage.get("completion_tokens", 0))
                prompt = int(usage.get("prompt_tokens", 0))
                usage["completion_tokens"] = completion + source_output_tokens
                usage["total_tokens"] = prompt + completion + source_output_tokens
                chunk["completion_usage"] = usage
            yield chunk

        if entry.error is not None:
            raise RuntimeError(
                f"target migration failed for request {entry.request_id}: {entry.error}"
            ) from entry.error

    async def rollback_current(self, request_id: str) -> None:
        entry = self._entries.get(request_id)
        if entry is None:
            return
        migration_audit("rolled_back", entry.envelope)
        await self._stop_target(entry)
        entry.target_context = _detached_context(request_id, entry.target_context)
        entry.envelope = entry.task = entry.queue = entry.error = None

    async def close(self, entry: _RequestEntry) -> None:
        self._retire_local(entry)
        await self._stop_target(entry)

    async def _stop_target(self, entry: _RequestEntry) -> None:
        if entry.task is None or entry.task.done():
            return


        entry.target_context.stop_generating()
        entry.task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await entry.task
