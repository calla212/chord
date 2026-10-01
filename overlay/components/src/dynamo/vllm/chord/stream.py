from __future__ import annotations

import asyncio
import time
from array import array
from collections import Counter
from collections.abc import AsyncIterator
from typing import Any

from ..llumnix.handoff import migration_envelope
from .config import WAITING_RETURN_ACK_RETRY_S
from .protocol import (
    ChordControlType,
    control_message,
    parse_waiting_return_stop_reason,
    request_metadata,
)


class ChordWaitingReturnAckRegistry:


    def __init__(self) -> None:
        self._pending: dict[tuple[str, int], asyncio.Event] = {}
        self.counters: Counter[str] = Counter()

    def arm(self, request_id: str, epoch: int) -> asyncio.Event:
        return self._pending.setdefault((request_id, epoch), asyncio.Event())

    def acknowledge(self, request_id: str, epoch: int) -> bool:
        event = self._pending.get((request_id, epoch))
        if event is None:
            return False
        event.set()
        return True

    def discard(self, request_id: str, epoch: int) -> None:
        self._pending.pop((request_id, epoch), None)


def _append_tokens(history: array[int], values: Any) -> bool:
    if not isinstance(values, list) or not values:
        return False
    tokens = [int(value) for value in values]
    if any(value < 0 or value > 0xFFFFFFFF for value in tokens):
        raise ValueError("Chord token IDs must fit in uint32")
    history.extend(tokens)
    return True


async def chord_worker_stream(
    owner: Any,
    request: dict[str, Any],
    context: Any,
    source_stream: AsyncIterator[dict[str, Any]],
) -> AsyncIterator[dict[str, Any]]:


    metadata = request_metadata(request)
    if not getattr(owner, "_chord_enabled", False) or metadata is None:
        async for chunk in source_stream:
            yield chunk
        return

    request_id = str(metadata["request_id"])
    epoch = int(metadata["epoch"])
    stream_owner_worker_id = int(
        metadata.get("stream_owner_worker_id", owner._chord_worker_id)
    )
    is_migration_target = migration_envelope(request) is not None
    if epoch <= 0 or stream_owner_worker_id < 0:
        raise ValueError(
            f"invalid Chord stream identity request={request_id} "
            f"epoch={epoch} owner={stream_owner_worker_id}"
        )


    counters = owner._chord_waiting_return_acks.counters
    counters["streams_opened"] += 1
    history = array("I")
    if not is_migration_target:
        _append_tokens(history, list(metadata.get("output_token_ids", ())))
    last_progress_time = float(
        metadata.get("last_progress_time", metadata.get("arrival_time", time.time()))
    )

    async for chunk in source_stream:
        if not is_migration_target and _append_tokens(history, chunk.get("token_ids")):
            counters["history_tokens"] += len(chunk["token_ids"])
            last_progress_time = time.time()

        marker = parse_waiting_return_stop_reason(chunk.get("stop_reason"))
        finish_reason = str(chunk.get("finish_reason") or "").lower()
        if marker is None or finish_reason not in {"abort", "cancelled"}:
            yield chunk
            continue

        if is_migration_target:
            yield chunk
            return
        if marker.epoch != epoch:
            raise RuntimeError(
                f"Chord waiting-return epoch mismatch request={request_id} "
                f"stream={epoch} marker={marker.epoch}"
            )

        ack = owner._chord_waiting_return_acks.arm(request_id, epoch)
        payload = control_message(
            ChordControlType.WAITING_RETURN,
            request_id=request_id,
            epoch=epoch,
            phase=marker.phase.value,
            reason=marker.reason,
            waiting_since=marker.waiting_since,
            wait_age_s=marker.wait_age_s,
            compute_worker_id=marker.worker_id,
            stream_owner_worker_id=stream_owner_worker_id,
            output_token_ids=list(history),
            last_progress_time=last_progress_time,
            num_preemptions=marker.num_preemptions,
        )
        try:
            while not ack.is_set() and not context.is_stopped():
                yield payload
                try:
                    await asyncio.wait_for(
                        asyncio.shield(ack.wait()),
                        timeout=WAITING_RETURN_ACK_RETRY_S,
                    )
                except TimeoutError:
                    pass
        finally:
            owner._chord_waiting_return_acks.discard(request_id, epoch)
        return
