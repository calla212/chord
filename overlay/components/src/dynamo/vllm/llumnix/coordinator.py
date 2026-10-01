from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import signal
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import aiohttp

from .config import LlumnixConfig
from .types import InstanceStatus
from .migration import MigrationPair, choose_migration_pairs

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class WorkerControl:
    base_url: str

    def url(self, control: str) -> str:
        return f"{self.base_url.rstrip('/')}/engine/control/{control}"


@dataclass(frozen=True, slots=True)
class MigrationReservation:
    request_id: str
    target_worker_id: int
    reserved_tokens: int
    created_at: float


def choose_migration_pair(
    statuses: list[InstanceStatus],
    config: LlumnixConfig,
    *,
    now_ms: int | None = None,
    filter_stale: bool = True,
    reserved_tokens_by_worker: Mapping[int, int] | None = None,
    reserved_migrate_in_by_worker: Mapping[int, int] | None = None,
) -> tuple[InstanceStatus, InstanceStatus] | None:


    pairs = choose_migration_pairs(
        statuses,
        config,
        now_ms=now_ms,
        filter_stale=filter_stale,
        reserved_tokens_by_worker=reserved_tokens_by_worker,
        reserved_migrate_in_by_worker=reserved_migrate_in_by_worker,
    )
    if not pairs:
        return None
    pair = pairs[0]
    return pair.source, pair.target


class LlumnixCoordinator:
    def __init__(self, workers: list[WorkerControl], config: LlumnixConfig) -> None:
        if len(workers) < 2:
            raise ValueError("Llumnix requires at least two aggregate workers")
        config.validate()
        self.workers = workers
        self.config = config
        self.statuses: dict[str, InstanceStatus] = {}
        self._migration_reservations: dict[tuple[int, str], MigrationReservation] = {}
        self._stop = asyncio.Event()

    async def _post(
        self,
        session: aiohttp.ClientSession,
        worker: WorkerControl,
        control: str,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        async with session.post(worker.url(control), json=body) as response:
            text = await response.text()
            if response.status != 200:
                raise RuntimeError(
                    f"{worker.base_url} {control} returned {response.status}: {text}"
                )
            value = json.loads(text)
            if not isinstance(value, dict):
                raise TypeError(f"{control} returned non-object JSON")
            return value

    async def _poll_one(
        self, session: aiohttp.ClientSession, worker: WorkerControl
    ) -> None:
        try:
            value = await self._post(session, worker, "llumnix_status", {})
            if value.get("status") == "disabled":
                raise RuntimeError("worker reports Llumnix disabled")
            self.statuses[worker.base_url] = InstanceStatus.from_dict(value)
        except Exception:
            self.statuses.pop(worker.base_url, None)
            logger.warning(
                "Llumnix status poll failed: %s", worker.base_url, exc_info=True
            )

    async def poll(self, session: aiohttp.ClientSession) -> None:
        await asyncio.gather(
            *(self._poll_one(session, worker) for worker in self.workers)
        )
        self._reconcile_migration_reservations()

    def _reconcile_migration_reservations(self, *, now: float | None = None) -> None:
        now = time.monotonic() if now is None else now
        statuses_by_worker = {
            status.worker_id: status for status in self.statuses.values()
        }
        for key, reservation in tuple(self._migration_reservations.items()):
            target = statuses_by_worker.get(reservation.target_worker_id)
            observed = target is not None and any(
                request.request_id == reservation.request_id
                for request in target.requests
            )
            expired = (
                now - reservation.created_at
                >= self.config.migration_reservation_timeout_s
            )
            if not observed and not expired:
                continue
            self._migration_reservations.pop(key, None)
            logger.debug(
                "Llumnix migration reservation released request=%s target=%d "
                "tokens=%d reason=%s",
                reservation.request_id,
                reservation.target_worker_id,
                reservation.reserved_tokens,
                "observed" if observed else "expired",
            )

    def _reservation_overlays(self) -> tuple[dict[int, int], dict[int, int]]:
        tokens_by_worker: dict[int, int] = {}
        requests_by_worker: dict[int, int] = {}
        for reservation in self._migration_reservations.values():
            worker_id = reservation.target_worker_id
            tokens_by_worker[worker_id] = (
                tokens_by_worker.get(worker_id, 0) + reservation.reserved_tokens
            )
            requests_by_worker[worker_id] = requests_by_worker.get(worker_id, 0) + 1
        return tokens_by_worker, requests_by_worker

    def _record_migration_reservations(
        self,
        source: InstanceStatus,
        target: InstanceStatus,
        result: dict[str, Any],
        *,
        now: float | None = None,
    ) -> None:
        now = time.monotonic() if now is None else now
        source_requests = {request.request_id: request for request in source.requests}
        for migration in result.get("migrations") or ():
            request_id = str(migration.get("request_id", ""))
            if not request_id:
                logger.warning(
                    "Llumnix accepted migration omitted request_id: %r", migration
                )
                continue

            prompt_tokens_value = migration.get("prompt_tokens")
            if prompt_tokens_value is None:
                snapshot = source_requests.get(request_id)
                if snapshot is not None:
                    prompt_tokens_value = snapshot.prompt_tokens

            if prompt_tokens_value is None:
                reserved_tokens = 0
                logger.warning(
                    "Llumnix migration reservation has count-only accounting: "
                    "request=%s target=%d",
                    request_id,
                    target.worker_id,
                )
            else:
                prompt_tokens = max(0, int(prompt_tokens_value))
                output_tokens = max(
                    0,
                    int(migration.get("output_tokens", 0)),
                    int(
                        migration.get(
                            "output_tokens_hint",
                            migration.get("output_tokens", 0),
                        )
                    ),
                )
                block_size = max(1, int(target.block_size))
                total_tokens = prompt_tokens + output_tokens
                reserved_tokens = (
                    (total_tokens + block_size - 1) // block_size
                ) * block_size

            reservation = MigrationReservation(
                request_id=request_id,
                target_worker_id=target.worker_id,
                reserved_tokens=reserved_tokens,
                created_at=now,
            )
            self._migration_reservations[(target.worker_id, request_id)] = reservation
            logger.debug(
                "Llumnix migration reservation created request=%s target=%d tokens=%d",
                request_id,
                target.worker_id,
                reserved_tokens,
            )

    async def _rebalance_pair(
        self,
        session: aiohttp.ClientSession,
        source_control: WorkerControl,
        pair: MigrationPair,
        reserved_migrate_in_requests: int,
    ) -> dict[str, Any]:
        source = pair.source
        target = pair.target
        body = {
            "target_worker_id": str(target.worker_id),
            "target_rpc_host": target.rpc_host,
            "target_rpc_port": target.rpc_port,
            "policy": self.config.policy.value,
            "limit": self.config.limit.value,
            "value": self.config.limit_value,
            "max_requests": max(
                0,
                self.config.max_migrate_in_requests
                - target.num_migrate_in_reqs
                - reserved_migrate_in_requests,
            ),
            "trigger_policy": "llumnix-load-balance",
        }
        result = await self._post(
            session, source_control, "llumnix_prepare_migration", body
        )
        self._record_migration_reservations(source, target, result)
        logger.info(
            "Llumnix rebalance source=%d target=%d source_load=%.4f "
            "target_load=%.4f migrations=%d rejected=%d",
            source.worker_id,
            target.worker_id,
            pair.source_load,
            pair.target_load,
            len(result.get("migrations", [])),
            len(result.get("rejected_request_ids", [])),
        )
        return result

    async def rebalance(self, session: aiohttp.ClientSession) -> list[dict[str, Any]]:
        self._reconcile_migration_reservations()
        reserved_tokens, reserved_requests = self._reservation_overlays()
        pairs = choose_migration_pairs(
            list(self.statuses.values()),
            self.config,
            reserved_tokens_by_worker=reserved_tokens,
            reserved_migrate_in_by_worker=reserved_requests,
        )
        if not pairs:
            return []

        controls_by_worker: dict[int, WorkerControl] = {}
        for worker in self.workers:
            status = self.statuses.get(worker.base_url)
            if status is not None:
                controls_by_worker[status.worker_id] = worker

        scheduled_pairs: list[MigrationPair] = []
        tasks: list[Any] = []
        for pair in pairs:
            source_control = controls_by_worker.get(pair.source.worker_id)
            if source_control is None:
                logger.warning(
                    "Llumnix source control missing for worker=%d",
                    pair.source.worker_id,
                )
                continue
            scheduled_pairs.append(pair)
            tasks.append(
                self._rebalance_pair(
                    session,
                    source_control,
                    pair,
                    reserved_requests.get(pair.target.worker_id, 0),
                )
            )
        if not tasks:
            return []

        logger.info("Llumnix rebalance batch pairs=%d", len(tasks))
        outcomes = await asyncio.gather(*tasks, return_exceptions=True)
        results: list[dict[str, Any]] = []
        for pair, outcome in zip(scheduled_pairs, outcomes):
            if isinstance(outcome, BaseException):
                logger.warning(
                    "Llumnix rebalance pair failed source=%d target=%d error=%r",
                    pair.source.worker_id,
                    pair.target.worker_id,
                    outcome,
                )
                continue
            results.append(outcome)
        return results

    async def run(self) -> None:
        timeout = aiohttp.ClientTimeout(total=2.0)
        next_decision = 0.0
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while not self._stop.is_set():
                started = time.monotonic()
                await self.poll(session)
                if started >= next_decision:
                    try:
                        await self.rebalance(session)
                    except Exception:
                        logger.warning("Llumnix rebalance failed", exc_info=True)
                    next_decision = started + self.config.decision_interval_ms / 1000
                remaining = self.config.status_interval_ms / 1000 - (
                    time.monotonic() - started
                )
                try:
                    await asyncio.wait_for(
                        self._stop.wait(), timeout=max(0.001, remaining)
                    )
                except TimeoutError:
                    pass

    def stop(self) -> None:
        self._stop.set()


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dynamo Llumnix coordinator")
    parser.add_argument(
        "--worker",
        action="append",
        default=[],
        help="worker system-server base URL; repeat once per worker",
    )
    return parser.parse_args(argv)


async def _main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    worker_urls = args.worker or [
        item.strip()
        for item in os.getenv("DYN_LLUMNIX_WORKERS", "").split(",")
        if item.strip()
    ]
    coordinator = LlumnixCoordinator(
        [WorkerControl(url) for url in worker_urls],
        LlumnixConfig.from_env(),
    )
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGINT, signal.SIGTERM):
        loop.add_signal_handler(signum, coordinator.stop)
    await coordinator.run()


if __name__ == "__main__":
    logging.basicConfig(level=os.getenv("DYN_LLUMNIX_LOG_LEVEL", "INFO"))
    asyncio.run(_main())
