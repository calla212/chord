from __future__ import annotations

import math
import time
from typing import Any

from vllm.v1.engine import EngineCoreRequest

from ..llumnix.handoff import migration_envelope
from .config import ChordConfig
from .protocol import (
    BACKEND_REQUEST_ID_KEY,
    make_backend_request_id,
    request_metadata,
)


def _validate_request(
    request: dict[str, Any], prompt: Any, sampling_params: Any
) -> None:
    if not isinstance(prompt, dict) or "prompt_token_ids" not in prompt:
        raise ValueError(
            "Chord supports token-input aggregate requests only; "
            "text and multimodal inputs are not replay-safe"
        )
    if request.get("multi_modal_data") is not None:
        raise ValueError("Chord does not support multimodal request replay")
    if int(getattr(sampling_params, "n", 1)) != 1:
        raise ValueError("Chord requires sampling_params.n == 1")
    temperature = float(getattr(sampling_params, "temperature", 0.0) or 0.0)
    if not math.isclose(temperature, 0.0, abs_tol=1e-12):
        raise ValueError("Chord currently supports greedy temperature=0 requests only")
    if getattr(sampling_params, "structured_outputs", None) is not None:
        raise ValueError("Chord does not support structured-output requests")


async def build_chord_engine_request(
    engine_client: Any,
    request: dict[str, Any],
    prompt: Any,
    sampling_params: Any,
    request_id: str,
    *,
    lora_request: Any = None,
    trace_headers: Any = None,
    priority: int = 0,
    data_parallel_rank: int | None = None,
) -> EngineCoreRequest | None:


    config = ChordConfig.from_env()
    if not config.enabled:
        return None
    metadata = request_metadata(request)
    if metadata is None:
        return None
    _validate_request(request, prompt, sampling_params)

    canonical_id = str(metadata.get("request_id") or "")
    epoch = int(metadata.get("epoch", 0))
    if epoch <= 0:
        raise ValueError("Chord request epoch must be positive")
    backend_request_id = str(metadata.get(BACKEND_REQUEST_ID_KEY) or "")
    expected_backend_request_id = make_backend_request_id(canonical_id, epoch)
    if backend_request_id != expected_backend_request_id:
        raise ValueError(
            "Chord backend request ID does not match its canonical request ID and epoch"
        )
    if request_id != backend_request_id:
        raise ValueError(
            "Chord worker transport request ID does not match the assigned "
            "backend request ID"
        )
    running_transfer = migration_envelope(request)
    if (
        running_transfer is not None
        and running_transfer.request_id != backend_request_id
    ):
        raise ValueError("Chord and Llumnix backend request IDs do not match")
    arrival_time = float(metadata.get("arrival_time", time.time()))
    last_progress_time = float(metadata.get("last_progress_time", arrival_time))
    core_request = engine_client.input_processor.process_inputs(
        backend_request_id,
        prompt,
        sampling_params,
        supported_tasks=await engine_client.get_supported_tasks(),
        arrival_time=arrival_time,
        lora_request=lora_request,
        trace_headers=trace_headers,
        priority=priority,
        data_parallel_rank=data_parallel_rank,
    )
    core_request.chord_request_version = epoch
    core_request.chord_last_progress_time = last_progress_time


    if running_transfer is None:
        core_request.resume_output_token_ids = [
            int(token) for token in metadata.get("output_token_ids", ())
        ]
        core_request.resume_num_preemptions = int(metadata.get("num_preemptions", 0))
    return core_request
