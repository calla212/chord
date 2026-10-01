from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .types import ChordRequestPhase

REQUEST_METADATA_KEY = "dynamo_chord"
BACKEND_REQUEST_ID_KEY = "backend_request_id"
CONTROL_KEY = "dynamo_chord_control"
WAITING_RETURN_STOP_PREFIX = "dynamo_chord_waiting_return"
_BACKEND_REQUEST_ID_SEPARATOR = ":dynamo-chord-epoch:"


def make_backend_request_id(request_id: str, epoch: int) -> str:


    if not request_id:
        raise ValueError("Chord request_id must be non-empty")
    if epoch <= 0:
        raise ValueError("Chord request epoch must be positive")
    return f"{request_id}{_BACKEND_REQUEST_ID_SEPARATOR}{epoch}"


def split_backend_request_id(value: str) -> tuple[str, int] | None:


    request_id, separator, raw_epoch = value.rpartition(_BACKEND_REQUEST_ID_SEPARATOR)
    if not separator or not request_id:
        return None
    try:
        epoch = int(raw_epoch)
    except ValueError:
        return None
    if epoch <= 0:
        return None
    return request_id, epoch


class ChordControlType(str, Enum):
    WAITING_RETURN = "WAITING_RETURN"


@dataclass(frozen=True, slots=True)
class WaitingReturnMarker:
    worker_id: int
    epoch: int
    phase: ChordRequestPhase
    reason: str
    waiting_since: float
    wait_age_s: float
    num_preemptions: int


def request_metadata(request: dict[str, Any]) -> dict[str, Any] | None:
    extra_args = request.get("extra_args")
    for value in (
        request.get(REQUEST_METADATA_KEY),
        extra_args.get(REQUEST_METADATA_KEY) if isinstance(extra_args, dict) else None,
    ):
        if isinstance(value, dict):
            return value
    return None


def control_message(kind: ChordControlType, **fields: Any) -> dict[str, Any]:


    return {
        "token_ids": [],
        "tokens": None,
        "text": None,
        "cum_log_probs": None,
        "log_probs": None,
        "top_logprobs": None,
        "finish_reason": None,
        "stop_reason": None,
        "index": 0,
        "extra_args": {CONTROL_KEY: {"type": kind.value, **fields}},
    }


def parse_control(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, dict):
        return None
    extra_args = value.get("extra_args")
    for control in (
        value.get(CONTROL_KEY),
        extra_args.get(CONTROL_KEY) if isinstance(extra_args, dict) else None,
    ):
        if isinstance(control, dict):
            return control
    return None


def waiting_return_stop_reason(
    *,
    worker_id: int,
    epoch: int,
    phase: ChordRequestPhase,
    reason: str,
    waiting_since: float,
    wait_age_s: float,
    num_preemptions: int,
) -> str:
    if ":" in reason:
        raise ValueError("Chord waiting-return reason must not contain ':'")
    return ":".join(
        (
            WAITING_RETURN_STOP_PREFIX,
            str(worker_id),
            str(epoch),
            phase.value,
            reason,
            format(waiting_since, ".17g"),
            format(wait_age_s, ".17g"),
            str(num_preemptions),
        )
    )


def parse_waiting_return_stop_reason(value: Any) -> WaitingReturnMarker | None:
    if not isinstance(value, str):
        return None
    fields = value.split(":")
    if len(fields) != 8 or fields[0] != WAITING_RETURN_STOP_PREFIX:
        return None
    try:
        marker = WaitingReturnMarker(
            worker_id=int(fields[1]),
            epoch=int(fields[2]),
            phase=ChordRequestPhase(fields[3]),
            reason=fields[4],
            waiting_since=float(fields[5]),
            wait_age_s=float(fields[6]),
            num_preemptions=int(fields[7]),
        )
    except (TypeError, ValueError):
        return None
    if (
        marker.worker_id < 0
        or marker.epoch <= 0
        or marker.waiting_since <= 0
        or marker.wait_age_s < 0
        or marker.num_preemptions < 0
        or not marker.reason
    ):
        return None
    return marker
