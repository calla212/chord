from __future__ import annotations

import logging
import os

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorRole

from .runtime import install_vllm_env_compat, install_worker_output_state_compat

install_vllm_env_compat()

from blade_kvt.hybrid_connector import (
    HybridConnector as _BladeHybridConnector,
)
from blade_kvt.hybrid_connector.engine_proxy import worker_init


def _validate_transfer_cache_layout(kv_caches):


    if not kv_caches or any(
        cache.ndim != 5 or cache.shape[1] != 2 or not cache.is_contiguous()
        for cache in kv_caches.values()
    ):
        raise ValueError("Aligned Llumnix requires contiguous [blocks, 2, tokens, heads, dim] KV caches")


class DynamoHybridConnector(_BladeHybridConnector):


    def __init__(self, vllm_config, role, kv_cache_config=None):
        if role == KVConnectorRole.WORKER:


            layout = os.environ.setdefault("BLLM_KVTRANS_CACHE_SHAPE", "3")
            if layout != "3":
                raise ValueError("vLLM 0.23 block-first KV transfer requires BLLM_KVTRANS_CACHE_SHAPE=3")
            from blade_kvt.hybrid_connector import engine_proxy

            if engine_proxy._g_worker_loop is None:
                try:
                    from vllm.distributed.parallel_state import (
                        get_tensor_model_parallel_rank,
                    )

                    local_rank = get_tensor_model_parallel_rank()
                except Exception:
                    local_rank = 0
                worker_init(vllm_config, local_rank)
            install_worker_output_state_compat()
        super().__init__(vllm_config, role, kv_cache_config)


    def register_kv_caches(self, kv_caches):
        _validate_transfer_cache_layout(kv_caches)
        first = next(iter(kv_caches.values()))
        logging.getLogger(__name__).info(
            "Llumnix block-first KV transfer: layout=3 shape=%s stride=%s",
            tuple(first.shape), first.stride(),
        )
        return super().register_kv_caches(kv_caches)



HybridConnector = DynamoHybridConnector
