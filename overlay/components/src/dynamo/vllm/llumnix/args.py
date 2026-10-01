from __future__ import annotations

import logging
import os
from typing import Any

from dynamo.common.constants import DisaggregationMode

from vllm.config import KVTransferConfig

from .config import LlumnixConfig

logger = logging.getLogger(__name__)


def configure_llumnix(
    dynamo_config: Any, engine_config: Any, defaults: dict[str, Any]
) -> None:


    llumnix = LlumnixConfig.from_env()
    if not llumnix.enabled:
        return
    llumnix.validate()


    os.environ["VLLM_DISABLE_REQUEST_ID_RANDOMIZATION"] = "1"

    if dynamo_config.disaggregation_mode != DisaggregationMode.AGGREGATED:
        raise ValueError(
            "DYN_LLUMNIX_ENABLE currently supports aggregate (non-PD) workers only"
        )
    if llumnix.rpc_port <= 0:
        raise ValueError("DYN_LLUMNIX_RPC_PORT must be set for every Llumnix worker")

    scheduler_cls = getattr(engine_config, "scheduler_cls", None)
    scheduler_names = {
        "dynamo.vllm.instrumented_scheduler.DynamoScheduler",
        "dynamo.vllm.instrumented_scheduler.InstrumentedScheduler",
    }
    if scheduler_cls is not None and str(scheduler_cls) not in scheduler_names:
        raise ValueError(
            "DYN_LLUMNIX_ENABLE requires DynamoScheduler; remove --scheduler-cls "
            f"or use one of {sorted(scheduler_names)}, got {scheduler_cls!r}"
        )
    defaults["scheduler_cls"] = "dynamo.vllm.instrumented_scheduler.DynamoScheduler"

    kv_config = getattr(engine_config, "kv_transfer_config", None)
    if kv_config is None:
        kv_config = KVTransferConfig(
            kv_connector="DynamoHybridConnector",
            kv_connector_module_path="dynamo.vllm.llumnix.connector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "backend": "kvt+migration",
                "naming_url": "fake://",
                "rpc_port": llumnix.rpc_port,
                "kvt_inst_id": f"dynamo-rpc-{llumnix.rpc_port}",
            },
        )
        engine_config.kv_transfer_config = kv_config

    expected = {
        "kv_connector": "DynamoHybridConnector",
        "kv_connector_module_path": "dynamo.vllm.llumnix.connector",
        "kv_role": "kv_both",
    }
    for field, value in expected.items():
        actual = getattr(kv_config, field, None)
        if actual != value:
            raise ValueError(
                f"Llumnix requires kv_transfer_config.{field}={value!r}, got {actual!r}"
            )
    extra = kv_config.kv_connector_extra_config
    if not isinstance(extra, dict):
        raise TypeError("Llumnix kv_connector_extra_config must be a dict")
    extra.setdefault("backend", "kvt+migration")
    extra.setdefault("naming_url", "fake://")
    extra.setdefault("rpc_port", llumnix.rpc_port)
    extra.setdefault("kvt_inst_id", f"dynamo-rpc-{llumnix.rpc_port}")
    if extra["backend"] not in {"kvt+migration", "migration+kvt"}:
        raise ValueError("Llumnix aggregate workers require backend=kvt+migration")
    if int(extra["rpc_port"]) != llumnix.rpc_port:
        raise ValueError(
            "DYN_LLUMNIX_RPC_PORT must match kv_connector_extra_config.rpc_port"
        )

    logger.info(
        "Llumnix enabled: KVT RPC port=%d, scheduler=DynamoScheduler",
        llumnix.rpc_port,
    )
