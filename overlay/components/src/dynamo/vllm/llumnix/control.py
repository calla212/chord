from __future__ import annotations

from typing import Any

from .types import MigrationDecision, MigrationPolicy
from .audit import migration_audit


class LlumnixControlMixin:


    engine_client: Any
    _endpoint: Any
    _llumnix_handoffs: Any
    _llumnix_worker_id: int

    async def _llumnix_core_call(self, method: str, *args: Any) -> Any:
        if self.engine_client is None:
            raise RuntimeError("Engine not initialized")
        return await self.engine_client.engine_core.call_utility_async(method, *args)

    async def _llumnix_rollback_request(self, request_id: str) -> None:
        if self.engine_client is None:
            return
        await self._llumnix_core_call(
            "llumnix_rollback_migration", {"request_ids": [request_id]}
        )

    async def llumnix_status(self, body: dict[str, Any]) -> dict[str, Any]:
        del body
        return await self._llumnix_core_call("llumnix_status")

    async def llumnix_prepare_migration(self, body: dict[str, Any]) -> dict[str, Any]:
        registry = getattr(self, "_llumnix_handoffs", None)
        if registry is None or self._endpoint is None:
            return {"status": "error", "message": "Llumnix handoff is unavailable"}

        try:
            target_worker_id = int(body["target_worker_id"])
            target_rpc_host = str(body["target_rpc_host"])
            target_rpc_port = int(body["target_rpc_port"])
        except (KeyError, TypeError, ValueError) as error:
            return {
                "status": "error",
                "message": f"invalid migration target: {error}",
            }
        if (
            target_worker_id == self._llumnix_worker_id
            or not target_rpc_host
            or target_rpc_port <= 0
        ):
            return {"status": "error", "message": "invalid migration target"}


        try:
            await registry.wait_for_target(target_worker_id, self._endpoint)
        except Exception as error:
            return {
                "status": "error",
                "message": f"migration target unavailable: {error}",
            }

        prepared = await self._llumnix_core_call("llumnix_prepare_migration", body)
        if prepared.get("status") != "ok":
            return prepared

        requests = prepared.get("requests", [])
        try:
            policy = MigrationPolicy(
                str(prepared.get("policy", body.get("policy", "SR"))).upper()
            )
        except ValueError as error:
            await self._llumnix_core_call(
                "llumnix_rollback_migration",
                {"request_ids": [str(item["request_id"]) for item in requests]},
            )
            return {"status": "error", "message": str(error)}

        accepted: list[dict[str, Any]] = []
        rejected: list[str] = []
        for request in requests:
            request_id = str(request["request_id"])
            observed_output_tokens = int(request.get("output_tokens", 0))
            output_tokens_hint = observed_output_tokens
            if observed_output_tokens > 0:
                output_tokens_hint += int(prepared.get("migrate_extra_tokens", 0))
            decision = MigrationDecision(
                request_id=request_id,
                source_worker_id=int(prepared["worker_id"]),
                target_worker_id=target_worker_id,
                source_rpc_host=str(prepared["rpc_host"]),
                source_rpc_port=int(prepared["rpc_port"]),
                output_tokens=observed_output_tokens,
                output_tokens_hint=output_tokens_hint,
                policy=policy,
                trigger_policy=str(request["migration_ticket"]),
                source_phase=str(request.get("phase", "unknown")),
                source_computed_tokens=int(request.get("computed_tokens", 0)),
            )
            migration_audit("selected", decision)
            if registry.start(decision, self._endpoint):
                migration_audit("accepted", decision)
                migration = decision.to_dict()
                if "prompt_tokens" in request:
                    migration["prompt_tokens"] = int(request["prompt_tokens"])
                accepted.append(migration)
            else:
                migration_audit("rejected", decision)
                rejected.append(request_id)

        if rejected:
            await self._llumnix_core_call(
                "llumnix_rollback_migration", {"request_ids": rejected}
            )
        return {
            "status": "ok",
            "source_worker_id": int(prepared["worker_id"]),
            "target_worker_id": target_worker_id,
            "migrations": accepted,
            "rejected_request_ids": rejected,
        }

    async def llumnix_rollback_migration(self, body: dict[str, Any]) -> dict[str, Any]:
        request_ids = body.get("request_ids", [])
        if isinstance(request_ids, str):
            request_ids = [request_ids]
        result = await self._llumnix_core_call(
            "llumnix_rollback_migration",
            {"request_ids": [str(item) for item in request_ids]},
        )
        registry = getattr(self, "_llumnix_handoffs", None)
        if registry is not None:
            for request_id in result["request_ids"]:
                await registry.rollback_current(request_id)
        return result
