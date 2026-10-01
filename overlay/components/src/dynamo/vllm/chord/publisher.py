from __future__ import annotations

import asyncio
import logging
import os
import time
from collections import Counter
from typing import Any

import aiohttp

from .config import ChordConfig

logger = logging.getLogger(__name__)


class StatusPublisher:
    def __init__(self, handler: Any, config: ChordConfig) -> None:
        self.handler = handler
        self.config = config
        self.worker_url = os.environ["DYN_CHORD_WORKER_CONTROL_URL"].rstrip("/")
        self.latest: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)
        self.probes: list[dict[str, Any]] = []
        self.counters: Counter[str] = Counter()

    async def collect(self) -> None:
        round_id = 0
        while True:
            started = time.monotonic()
            round_id += 1
            value = await self.handler.chord_status({
                "round_id": round_id, "include_migration": False, "probes": self.probes,
            })
            if value.get("status") != "ok":
                raise RuntimeError(f"Chord compact sampling failed: {value}")
            self.counters["samples"] += 1
            if self.latest.full():
                self.latest.get_nowait()
                self.counters["replaced"] += 1
            self.latest.put_nowait(dict(value, worker_url=self.worker_url))
            await asyncio.sleep(max(0.001, self.config.monitor_interval_ms / 1000
                                    - (time.monotonic() - started)))

    async def send(self, session: aiohttp.ClientSession) -> None:
        last_log = 0.0
        url = self.config.frontend_control_url + "/report"
        while True:
            value = await self.latest.get()
            if time.time() * 1000 - value["timestamp_ms"] > self.config.status_stale_ms:
                self.counters["expired"] += 1
                continue
            try:
                async with session.post(url, json=value) as response:
                    response.raise_for_status()
                    self.probes = (await response.json())["probes"]
                self.counters["sent"] += 1
            except (aiohttp.ClientError, TimeoutError):

                self.counters["send_errors"] += 1
            if time.monotonic() - last_log >= 10:
                logger.info("Chord publisher worker=%s counters=%s", self.worker_url,
                            dict(self.counters))
                last_log = time.monotonic()

    async def run(self) -> None:
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=1)) as session:
            tasks = [asyncio.create_task(self.collect()), asyncio.create_task(self.send(session))]
            try:
                await asyncio.gather(*tasks)
            finally:
                for task in tasks:
                    task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)


def start_publisher(handler: Any) -> asyncio.Task | None:
    config = ChordConfig.from_env()
    if not config.enabled or config.reference_policy:
        return None
    return asyncio.create_task(StatusPublisher(handler, config).run(), name="chord-publisher")
