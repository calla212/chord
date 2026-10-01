from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .metrics import EffectiveMetrics


@dataclass(frozen=True, slots=True)
class InitialSelection:
    worker_id: int
    metrics: EffectiveMetrics
    fallback: bool


def choose_initial_worker(
    eligible: Iterable[tuple[int, EffectiveMetrics]], *, threshold: int = 8192
) -> InitialSelection | None:

    views = list(eligible)
    if not views:
        return None
    under = [view for view in views if view[1].all_prefills_tokens_num < threshold]
    worker_id, metrics = min(
        under or views, key=lambda view: (view[1].all_prefills_tokens_num, view[0])
    )
    return InitialSelection(worker_id, metrics, fallback=not under)
