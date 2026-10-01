from __future__ import annotations

import logging
from typing import Any

from dynamo.common.constants import DisaggregationMode

from .config import ChordConfig

logger = logging.getLogger(__name__)


def configure_chord(
    dynamo_config: Any, engine_config: Any, defaults: dict[str, Any]
) -> None:


    chord = ChordConfig.from_env()
    if not chord.enabled:
        return
    chord.validate(require_worker_id=False)
    if dynamo_config.disaggregation_mode != DisaggregationMode.AGGREGATED:
        raise ValueError("DYN_CHORD_ENABLE supports aggregate (non-P/D) workers only")

    from dynamo.vllm.llumnix.config import LlumnixConfig

    llumnix = LlumnixConfig.from_env()
    if not llumnix.enabled:
        raise ValueError(
            "DYN_CHORD_ENABLE requires DYN_LLUMNIX_ENABLE=1 for running-request migration"
        )

    scheduler_cls = getattr(engine_config, "scheduler_cls", None)
    scheduler_names = {
        "dynamo.vllm.instrumented_scheduler.DynamoScheduler",
        "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler",
    }
    if scheduler_cls is not None and str(scheduler_cls) not in scheduler_names:
        raise ValueError(
            "DYN_CHORD_ENABLE requires DynamoScheduler; remove --scheduler-cls "
            f"or use one of {sorted(scheduler_names)}, got {scheduler_cls!r}"
        )
    defaults["scheduler_cls"] = "dynamo.vllm.instrumented_scheduler.DynamoScheduler"

    from vllm.v1.engine import EngineCoreRequest

    required_fields = {
        "resume_output_token_ids",
        "resume_num_preemptions",
        "chord_request_version",
        "chord_last_progress_time",
    }
    missing = required_fields.difference(EngineCoreRequest.__struct_fields__)
    if missing:
        raise RuntimeError(
            "DYN_CHORD_ENABLE requires the pinned vLLM-Chord fork; missing "
            f"EngineCoreRequest fields: {sorted(missing)}"
        )
    logger.info(
        "Chord worker enabled: monitor=%dms running_decision=%dms C=%.3f "
        "Tmax=%.3f max_plan=%d waiting_return_timeout=%.3fs skip_unfit=%s "
        "running_migration=%s",
        chord.monitor_interval_ms,
        chord.running_decision_interval_ms,
        chord.urgency_coefficient,
        chord.tbf_max_t,
        chord.max_plan_requests,
        chord.waiting_return_timeout_s,
        chord.skip_unfit,
        chord.running_migration_enabled,
    )
