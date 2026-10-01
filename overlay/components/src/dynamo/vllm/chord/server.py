from __future__ import annotations

import asyncio
import logging
from dataclasses import asdict
from typing import Any
from urllib.parse import urlsplit

from aiohttp import web

from .config import ChordConfig
from .coordinator import ChordCoordinator
from .state import DispatchLease

logger = logging.getLogger(__name__)


def _lease_response(lease: DispatchLease) -> web.Response:
    return web.json_response(asdict(lease))


@web.middleware
async def _error_middleware(request: web.Request, handler: Any) -> web.StreamResponse:
    try:
        return await handler(request)
    except asyncio.CancelledError:
        raise
    except Exception as error:
        logger.exception("Chord control request failed: %s", request.path)
        return web.json_response(
            {"status": "error", "message": str(error)},
            status=500,
        )


class ChordControlServer:


    def __init__(
        self,
        config: ChordConfig | None = None,
        *,
        coordinator: ChordCoordinator | None = None,
    ) -> None:
        self.config = config or ChordConfig.from_env()
        self.config.validate(require_worker_id=False, require_worker_urls=self.config.enabled)
        self.coordinator = coordinator or (ChordCoordinator(self.config) if self.config.enabled else None)
        self._runner: web.AppRunner | None = None
        self._site: web.TCPSite | None = None
        self._host, self._port = self._parse_bind(self.config.frontend_control_url)

    @staticmethod
    def _parse_bind(url: str) -> tuple[str, int]:
        parsed = urlsplit(url)
        if parsed.scheme != "http" or parsed.hostname is None or parsed.port is None:
            raise ValueError(
                "DYN_CHORD_FRONTEND_CONTROL_URL must be an explicit HTTP host:port"
            )
        if parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
            raise ValueError("Chord Frontend control endpoint must bind to loopback")
        if parsed.path not in {"", "/"} or parsed.query or parsed.fragment:
            raise ValueError("Chord Frontend control URL must not contain a path")
        return parsed.hostname, parsed.port

    async def start(self) -> None:
        if self._runner is not None:
            return
        if self.coordinator is not None:
            await self.coordinator.start()
        app = web.Application(middlewares=[_error_middleware])
        app.add_routes(
            [
                web.get("/health", self._health),
                web.get("/state", self._state),
                web.get("/native-loads", self._native_loads),
                web.post("/submit", self._submit),
                web.post("/waiting-return", self._waiting_return),
                web.post("/dispatch-failed", self._dispatch_failed),
                web.post("/finish", self._finish),
                web.post("/cancel", self._cancel),
                web.post("/report", self._report),
                web.post("/migration-committed", self._migration_committed),
            ]
        )
        runner = web.AppRunner(app, access_log=None, handler_cancellation=True)
        await runner.setup()
        site = web.TCPSite(runner, self._host, self._port, backlog=4096)
        try:
            await site.start()
        except BaseException:
            await runner.cleanup()
            if self.coordinator is not None:
                await self.coordinator.close()
            raise
        self._runner = runner
        self._site = site
        logger.info(
            "Chord Frontend control plane ready at %s",
            self.config.frontend_control_url,
        )

    async def close(self) -> None:
        runner = self._runner
        self._runner = None
        self._site = None
        if runner is not None:
            await runner.cleanup()
        if self.coordinator is not None:
            await self.coordinator.close()

    async def _health(self, request: web.Request) -> web.Response:
        del request
        return web.json_response({"status": "ok"})

    async def _state(self, request: web.Request) -> web.Response:
        if self.coordinator is None:
            return await self._native_loads(request)
        return web.json_response(await self.coordinator.state())

    async def _submit(self, request: web.Request) -> web.Response:
        body = await request.json()
        request_id = str(body["request_id"])
        try:
            lease = await self.coordinator.submit(
                request_id, int(body["prompt_tokens"])
            )
        except asyncio.CancelledError:
            await asyncio.shield(self.coordinator.cancel(request_id))
            raise
        return _lease_response(lease)

    async def _waiting_return(self, request: web.Request) -> web.Response:
        body = await request.json()
        request_id = str(body["request_id"])
        try:
            lease = await self.coordinator.redispatch_waiting(dict(body))
        except asyncio.CancelledError:
            await asyncio.shield(self.coordinator.cancel(request_id))
            raise
        return _lease_response(lease)

    async def _dispatch_failed(self, request: web.Request) -> web.Response:
        body = await request.json()
        request_id = str(body["request_id"])
        try:
            lease = await self.coordinator.redispatch_open_failure(
                request_id=request_id,
                epoch=int(body["epoch"]),
                worker_id=int(body["worker_id"]),
                error=RuntimeError(str(body.get("error", "native route failure"))),
            )
        except asyncio.CancelledError:
            await asyncio.shield(self.coordinator.cancel(request_id))
            raise
        return _lease_response(lease)

    async def _finish(self, request: web.Request) -> web.Response:
        body = await request.json()
        await self.coordinator.finish(
            str(body["request_id"]),
            int(body["epoch"]),
            str(body.get("reason", "finished")),
        )
        return web.json_response({"status": "ok"})

    async def _cancel(self, request: web.Request) -> web.Response:
        body = await request.json()
        await self.coordinator.cancel(str(body["request_id"]))
        return web.json_response({"status": "ok"})

    async def _report(self, request: web.Request) -> web.Response:
        body = await request.json()
        probes = self.coordinator.report(str(body.pop("worker_url")), body)
        return web.json_response({"probes": probes})

    async def _migration_committed(self, request: web.Request) -> web.Response:
        await self.coordinator.migration_committed(await request.json())
        return web.json_response({"status": "ok"})

    async def _native_loads(self, request: web.Request) -> web.Response:
        from dynamo._core import AlignedRouter
        return web.json_response({"loads": AlignedRouter().loads()})
