from __future__ import annotations

import bisect
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .metrics import EffectiveMetrics


@dataclass(frozen=True, slots=True)
class Prediction:
    value_ms: float
    reason: str | None = None

    @property
    def finite(self) -> bool:
        return math.isfinite(self.value_ms)

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["finite"] = self.finite
        if not self.finite:
            value["value_ms"] = None
        return value


class TpotPredictor:


    def __init__(
        self,
        batch_sizes: list[float],
        tokens_per_request: list[float],
        samples: dict[tuple[float, float], float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self.batch_sizes = sorted(set(batch_sizes))
        self.tokens_per_request = sorted(set(tokens_per_request))
        self.samples = dict(samples)
        self.metadata = dict(metadata or {})

    @classmethod
    def unavailable(cls) -> TpotPredictor:
        return cls([], [], {}, {"status": "profile_not_configured"})

    @classmethod
    def from_file(cls, path: str | Path) -> TpotPredictor:
        value = json.loads(Path(path).read_text())
        if int(value.get("schema_version", -1)) != 1:
            raise ValueError("unsupported Llumnix TPOT profile schema")
        axes = value.get("axes")
        if not isinstance(axes, dict):
            raise ValueError("TPOT profile is missing axes")
        batch_sizes = [float(item) for item in axes.get("decode_batch_size", [])]
        tokens = [float(item) for item in axes.get("tokens_per_request", [])]
        if not batch_sizes or not tokens:
            raise ValueError("TPOT profile axes must be non-empty")
        if batch_sizes != sorted(set(batch_sizes)) or tokens != sorted(set(tokens)):
            raise ValueError("TPOT profile axes must be sorted and unique")

        samples: dict[tuple[float, float], float] = {}
        for point in value.get("points", []):
            if not point.get("valid", True):
                continue
            key = (
                float(point["decode_batch_size"]),
                float(point["tokens_per_request"]),
            )
            result = float(point["p50_tpot_ms"])
            if not math.isfinite(result) or result <= 0:
                continue
            samples[key] = result
        return cls(batch_sizes, tokens, samples, value.get("metadata", {}))

    def predict(self, batch_size: float, tokens_per_request: float) -> Prediction:
        if not self.batch_sizes or not self.tokens_per_request:
            return Prediction(float("inf"), "profile_not_configured")
        bounds_a = self._bounds(self.batch_sizes, batch_size)
        bounds_b = self._bounds(self.tokens_per_request, tokens_per_request)
        if bounds_a is None or bounds_b is None:
            return Prediction(float("inf"), "outside_profile_grid")
        a1, a2 = bounds_a
        b1, b2 = bounds_b
        corners = [(a1, b1), (a1, b2), (a2, b1), (a2, b2)]
        if any(point not in self.samples for point in corners):
            return Prediction(float("inf"), "missing_profile_point")

        z11, z12, z21, z22 = (self.samples[point] for point in corners)
        if a1 == a2 and b1 == b2:
            return Prediction(z11)
        if a1 == a2:
            wb = (tokens_per_request - b1) / (b2 - b1)
            return Prediction((1 - wb) * z11 + wb * z12)
        if b1 == b2:
            wa = (batch_size - a1) / (a2 - a1)
            return Prediction((1 - wa) * z11 + wa * z21)

        wa = (batch_size - a1) / (a2 - a1)
        wb = (tokens_per_request - b1) / (b2 - b1)
        return Prediction(
            (1 - wa) * (1 - wb) * z11
            + (1 - wa) * wb * z12
            + wa * (1 - wb) * z21
            + wa * wb * z22
        )

    def predict_next_decode(
        self, metrics: EffectiveMetrics, candidate_prompt_tokens: int
    ) -> Prediction:
        batch_size = metrics.decode_batch_size + 1
        tokens_per_request = max(
            8.0,
            (metrics.all_decode_tokens + max(0, candidate_prompt_tokens)) / batch_size,
        )
        return self.predict(float(batch_size), tokens_per_request)

    @staticmethod
    def _bounds(axis: list[float], value: float) -> tuple[float, float] | None:
        if not axis or value < axis[0] or value > axis[-1]:
            return None
        upper = bisect.bisect_left(axis, value)
        if upper < len(axis) and axis[upper] == value:
            return axis[upper], axis[upper]
        return axis[upper - 1], axis[upper]
