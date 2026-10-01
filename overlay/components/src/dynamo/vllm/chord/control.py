from __future__ import annotations

from typing import Any

from .stream import ChordWaitingReturnAckRegistry


class ChordControlMixin:


    engine_client: Any
    _chord_waiting_return_acks: ChordWaitingReturnAckRegistry

    async def _chord_core_call(self, method: str, *args: Any) -> Any:
        if self.engine_client is None:
            raise RuntimeError("Engine not initialized")
        return await self.engine_client.engine_core.call_utility_async(method, *args)

    async def chord_status(self, body: dict[str, Any]) -> dict[str, Any]:
        value = await self._chord_core_call("chord_status", body)
        registry = getattr(self, "_chord_waiting_return_acks", None)
        value["waiting_return_ack_pending"] = len(registry._pending) if registry else 0
        value["stream_counters"] = dict(registry.counters) if registry else {}
        handoffs = getattr(self, "_llumnix_handoffs", None)
        value["handoff_pending"] = len(handoffs._entries) if handoffs else 0
        value["runtime"]["handoff_registry"] = type(handoffs).__name__
        return value

    async def chord_drop_epoch(self, body: dict[str, Any]) -> dict[str, Any]:
        return await self._chord_core_call("chord_drop_epoch", body)

    async def chord_waiting_return_stored(self, body: dict[str, Any]) -> dict[str, Any]:
        try:
            request_id = str(body["request_id"])
            epoch = int(body["epoch"])
        except (KeyError, TypeError, ValueError) as error:
            return {"status": "error", "message": f"invalid ACK: {error}"}
        acknowledged = self._chord_waiting_return_acks.acknowledge(request_id, epoch)
        return {
            "status": "ok" if acknowledged else "already_released",
            "request_id": request_id,
            "epoch": epoch,
        }
