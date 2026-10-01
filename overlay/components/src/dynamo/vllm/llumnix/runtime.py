from __future__ import annotations

import inspect
import logging
import os
from functools import wraps
from typing import Any

logger = logging.getLogger(__name__)

_PRIOR_OUTPUT_TOKENS_KEY = "dynamo_llumnix_output_token_ids"


def install_vllm_env_compat() -> None:


    from vllm import envs

    defaults: dict[str, Any] = {
        "LLUMNIX_DETAILED_MIG_STATUS": True,
        "LLUMNIX_MAX_KV_CACHE_USAGE_RATIO_MIG_IN": float(
            os.getenv("DYN_LLUMNIX_MAX_MIGRATE_IN_RATIO", "0.3")
        ),
        "LLUMNIX_MAX_BLOCK_RATIO_MIG_IN": float(
            os.getenv("DYN_LLUMNIX_MAX_MIGRATE_IN_RATIO", "0.3")
        ),
        "LLUMNIX_MAX_REQ_MIG_IN": int(
            os.getenv("DYN_LLUMNIX_MAX_MIGRATE_IN_REQUESTS", "1")
        ),
        "LLUMNIX_MAX_TOKEN_MIG_IN": int(
            os.getenv("DYN_LLUMNIX_MAX_MIGRATE_IN_TOKENS", "10000")
        ),
        "VLLM_ATTENTION_BACKEND": None,
        "LLUMNIX_MIGRATE_IN_PD_WAY": True,
        "LLUMNIX_MIGRATE_IN_PD_WAY_EXTRA_TOKENS": int(
            os.getenv("DYN_LLUMNIX_MIGRATE_EXTRA_TOKENS", "20")
        ),
        "VLLM_ENABLE_BYPASS_SUBSTEP": False,
        "VLLM_ENABLE_BYPASS_TASK": False,
        "VLLM_FLASH_ATTN_USE_FAKE_TURBOQUANT": False,
        "VLLM_FLASH_ATTN_USE_FAST_TURBOQUANT": False,
        "VLLM_KVS_IO_TIMEOUT_SECONDS": 30.0,
        "VLLM_KVS_ON_MIN_LENGTH": 1024,
        "VLLM_KVS_USE_REQUEST_HASH": False,
        "VLLM_KVT_GONE_REQ_TTL_S": 30,
        "VLLM_KVT_MAX_DELAY_MS": 0,
        "VLLM_KV_TRANS_PROTOCOL": os.getenv("VLLM_KV_TRANS_PROTOCOL", "rdma"),
        "VLLM_LOG_STATS_INTERVAL": 5.0,
        "VLLM_PD_CONNMANAGER_CAP": 128,
        "VLLM_PD_TRY_CONNECT_TIMEOUT_SECONDS": 30.0,
        "VLLM_USE_MIGRATION_BACKEND_RETURN_TOKEN": False,
        "VLLM_V6D_ASYNC_REGISTER": False,
        "VLLM_V6D_STORE_NAN_CHECK": False,
        "VLLM_USE_SWAP_ENGINE": False,
        "VLLM_ZERO_HYBRID_KV_CACHE": False,
    }
    for name, value in defaults.items():
        try:
            getattr(envs, name)
        except (AttributeError, KeyError):
            setattr(envs, name, value)


def install_vllm_request_compat() -> None:


    from vllm.v1.request import Request

    original = Request.__init__
    if getattr(original, "_dynamo_llumnix", False):
        return

    @wraps(original)
    def request_init(self, *args: Any, **kwargs: Any) -> None:
        original(self, *args, **kwargs)
        if not hasattr(self, "queue_server_address"):
            self.queue_server_address = None
        if not hasattr(self, "num_external_computed_tokens"):
            self.num_external_computed_tokens = 0

    request_init._dynamo_llumnix = True
    Request.__init__ = request_init


def _find_initializing_core() -> Any:
    from vllm.v1.engine.core import EngineCore

    frame = inspect.currentframe()
    try:
        while frame is not None:
            candidate = frame.f_locals.get("self")
            if isinstance(candidate, EngineCore) and hasattr(candidate, "input_queue"):
                return candidate
            frame = frame.f_back
    finally:
        del frame
    raise RuntimeError(
        "DynamoScheduler must be constructed by vLLM EngineCore when "
        "DynamoHybridConnector is enabled"
    )


def _install_transfer_safe_migration_step(hybrid_modules: Any) -> None:


    original = hybrid_modules.HybridScheduler._step_migrating
    if getattr(original, "_dynamo_llumnix", False):
        return

    from blade_kvt.hybrid_connector.kvtbackend import P_KVT_STATE

    @wraps(original)
    def transfer_safe_step(self: Any) -> Any:
        deferred: dict[str, tuple[int, int]] = {}
        if self._cfg.scheduler_config.async_scheduling:
            snapshots = self._migrating_output_tokens_snapshot
            for request_id, (previous_output_tokens, touched) in list(
                snapshots.items()
            ):
                if touched:
                    continue
                request = hybrid_modules.sched_get_req(request_id)
                if (
                    request is None
                    or request.num_output_tokens <= previous_output_tokens
                ):
                    continue
                state = hybrid_modules.get_param(request, P_KVT_STATE, None)
                if state is not None and not getattr(state, "untouched", True):
                    continue

                current_output_tokens = request.num_output_tokens
                snapshots[request_id] = (current_output_tokens, False)
                deferred[request_id] = (
                    previous_output_tokens,
                    current_output_tokens,
                )
                logger.debug(
                    "Deferring Llumnix migration touch until KVT metadata is "
                    "built: request_id=%s previous=%d current=%d",
                    request_id,
                    previous_output_tokens,
                    current_output_tokens,
                )

        try:
            return original(self)
        finally:
            snapshots = self._migrating_output_tokens_snapshot
            for request_id, (
                previous_output_tokens,
                masked_output_tokens,
            ) in deferred.items():
                if snapshots.get(request_id) == (masked_output_tokens, False):
                    snapshots[request_id] = (previous_output_tokens, False)

    transfer_safe_step._dynamo_llumnix = True
    hybrid_modules.HybridScheduler._step_migrating = transfer_safe_step


def _install_migration_completion(kvtbackend: Any, proxy: Any) -> None:

    original_mark = kvtbackend._SendingReq.try_mark_done
    if getattr(original_mark, "_dynamo_llumnix", False):
        return
    original_wait = kvtbackend.PBackend._wait_kvt_state

    @wraps(original_mark)
    def mark_done(self, result):
        if result.ex is not None:
            self._dynamo_migration_error = result.ex
            tokens = getattr(self, "_dynamo_migration_tokens", None)
            if tokens is not None and not tokens.done():
                tokens.set_exception(result.ex)
        return original_mark(self, result)

    @wraps(original_wait)
    async def wait_kvt_state(self, request_id, state, request):
        scheduler = tokens = None
        try:
            transferred = await state._fut
            if transferred.ex is None and kvtbackend.migrate_in_pd_way(request):
                scheduler = proxy._sched().connector._sched
                tokens = scheduler._migrating_fut.get(request_id)

                state._dynamo_migration_tokens = tokens
                error = getattr(state, "_dynamo_migration_error", None)
                if error is not None and tokens is not None and not tokens.done():
                    tokens.set_exception(error)
            return await original_wait(self, request_id, state, request)
        except kvtbackend.CodeError as error:
            logger.info("Llumnix source ended during handoff: request_id=%s code=%s",
                        request_id, error.code)
            return kvtbackend.KVTResp(code=error.code, cached=0, computed=-1,
                                     output_token_ids=[])
        finally:
            self._sending.pop(request_id, None)
            if scheduler is not None and scheduler._migrating_fut.get(request_id) is tokens:
                scheduler._migrating_fut.pop(request_id, None)

    mark_done._dynamo_llumnix = True
    kvtbackend._SendingReq.try_mark_done = mark_done
    kvtbackend.PBackend._wait_kvt_state = wait_kvt_state


def _patch_blade_runtime() -> None:
    import blade_kvt.hybrid_connector as hybrid
    import blade_kvt.hybrid_connector.engine_proxy as proxy
    import blade_kvt.hybrid_connector.migration.backend as migration_backend
    import blade_kvt.hybrid_connector.migration.frontend as migration_frontend
    import msgspec
    from blade_kvt.hybrid_connector import hybrid_modules, kvtbackend, utils

    from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest, FinishReason
    from vllm.v1.serial_utils import MsgpackEncoder as VllmMsgpackEncoder

    _install_transfer_safe_migration_step(hybrid_modules)
    _install_migration_completion(kvtbackend, proxy)

    original_transfer = kvtbackend.PBackend.submit_transfer_kv
    if not getattr(original_transfer, "_dynamo_llumnix", False):
        @wraps(original_transfer)
        async def submit_transfer_kv(self, request):
            if request.migration:
                adapter = proxy._sched()._llumnix_status_adapter
                if not adapter.begin_transfer(request.reqid, request.migration_reason):
                    return kvtbackend.KVTResp(
                        code=kvtbackend.CODE_REQNOTFOUND, cached=0, computed=-1
                    )
            return await original_transfer(self, request)

        submit_transfer_kv._dynamo_llumnix = True
        kvtbackend.PBackend.submit_transfer_kv = submit_transfer_kv

    class OffsetMsgpackEncoder(VllmMsgpackEncoder):
        def encode_into(self, obj: Any, buf: bytearray, offset: int | None = None):
            if offset is None:
                return super().encode_into(obj, buf)
            if len(buf) != offset:
                raise ValueError(
                    f"msgpack offset {offset} does not match buffer length {len(buf)}"
                )
            encoded = list(super().encode(obj))
            buf.extend(encoded[0])
            encoded[0] = buf
            return encoded

    for module in (proxy, hybrid, kvtbackend, migration_backend, migration_frontend):
        if hasattr(module, "MsgpackEncoder"):
            module.MsgpackEncoder = OffsetMsgpackEncoder


    original_core_abort_req = migration_backend.core_abort_req
    if not getattr(original_core_abort_req, "_dynamo_llumnix", False):

        def core_abort_req(reqid: str, reason: str, output: bool):
            if reason == "migration.suspend":
                output = True
            return original_core_abort_req(reqid, reason, output)

        core_abort_req._dynamo_llumnix = True
        migration_backend.core_abort_req = core_abort_req

    original_list = utils.PeerManager._list_instance

    def list_instance(self):
        if self._naming_cli is None:
            return []
        return original_list(self)

    if not getattr(utils.PeerManager._list_instance, "_dynamo_llumnix", False):
        list_instance._dynamo_llumnix = True
        utils.PeerManager._list_instance = list_instance

    def req2corereq(req):
        fields = {
            "request_id": req.request_id,
            "prompt_token_ids": req.prompt_token_ids,
            "mm_features": req.mm_features,
            "sampling_params": req.sampling_params,
            "pooling_params": req.pooling_params,
            "eos_token_id": getattr(req, "eos_token_id", 0),
            "arrival_time": req.arrival_time,
            "lora_request": req.lora_request,
            "cache_salt": req.cache_salt,
            "data_parallel_rank": None,
            "prompt_embeds": req.prompt_embeds,
            "priority": req.priority,
            "trace_headers": req.trace_headers,
            "resume_output_token_ids": list(req.output_token_ids),
            "resume_num_preemptions": req.num_preemptions,
            "chord_request_version": getattr(req, "chord_request_version", 0),
            "chord_last_progress_time": getattr(
                req, "chord_last_progress_time", req.arrival_time
            ),
        }
        accepted = inspect.signature(EngineCoreRequest).parameters
        return EngineCoreRequest(
            **{key: val for key, val in fields.items() if key in accepted}
        )

    proxy.req2corereq = req2corereq


    kvtbackend.req2corereq = req2corereq

    def put_abort_resp(load_output, req):
        load_output[req.client_index].append(
            EngineCoreOutput(
                request_id=req.request_id,


                new_token_ids=[],
                finish_reason=FinishReason.ABORT,
            )
        )

    hybrid._put_abort_resp = put_abort_resp
    hybrid_modules._put_abort_resp = put_abort_resp


    def native_add(req):
        proxy._sched()._llumnix_add_request_native(req)

    def native_finish(request_ids):
        proxy._sched()._llumnix_finish_requests_native(request_ids)

    proxy.sched_add_req = native_add
    proxy.sched_finish_req = native_finish
    hybrid.sched_add_req = native_add
    hybrid.sched_finish_req = native_finish
    hybrid_modules.sched_add_req = native_add
    hybrid_modules.sched_finish_req = native_finish


    probe = bytearray(8)
    buffers = OffsetMsgpackEncoder().encode_into({"probe": 1}, probe, 8)
    if len(buffers) != 1 or msgspec.msgpack.decode(probe[8:]) != {"probe": 1}:
        raise RuntimeError("llumnix-kv msgpack compatibility probe failed")


def prepare_scheduler_runtime(vllm_config: Any) -> Any:


    install_vllm_env_compat()
    install_vllm_request_compat()
    _patch_blade_runtime()

    from blade_kvt.hybrid_connector import engine_proxy

    core = _find_initializing_core()
    if engine_proxy._g_core is None:
        engine_proxy.core_init(core, vllm_config)
    elif engine_proxy._g_core is not core:
        raise RuntimeError("llumnix-kv already belongs to another EngineCore")
    install_engine_core_utilities()
    return core


def attach_scheduler(core: Any, scheduler: Any) -> None:
    core.scheduler = scheduler


def install_engine_core_utilities() -> None:
    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_dynamo_llumnix_utilities", False):
        return

    def llumnix_status(self):
        return self.scheduler.llumnix_status()

    def llumnix_prepare_migration(self, body: dict[str, Any]):
        return self.scheduler.llumnix_prepare_migration(body)

    def llumnix_rollback_migration(self, body: dict[str, Any]):
        return self.scheduler.llumnix_rollback_migration(body)

    EngineCore.llumnix_status = llumnix_status
    EngineCore.llumnix_prepare_migration = llumnix_prepare_migration
    EngineCore.llumnix_rollback_migration = llumnix_rollback_migration
    EngineCore._dynamo_llumnix_utilities = True


def install_worker_output_state_compat() -> None:


    from vllm.v1.worker import gpu_model_runner

    cls = gpu_model_runner.GPUModelRunner
    original = cls._update_states
    if getattr(original, "_dynamo_llumnix", False):
        return

    def update_states(self, scheduler_output):
        prior: dict[str, list[int]] = {}
        for request in scheduler_output.scheduled_new_reqs:
            params = request.sampling_params
            extra_args = getattr(params, "extra_args", None) if params else None
            if extra_args and _PRIOR_OUTPUT_TOKENS_KEY in extra_args:
                prior[request.req_id] = list(extra_args[_PRIOR_OUTPUT_TOKENS_KEY])

        original_state = gpu_model_runner.CachedRequestState

        def state_factory(*args, **kwargs):
            request_id = kwargs.get("req_id", args[0] if args else None)
            if request_id in prior:
                kwargs["output_token_ids"] = list(prior[request_id])
            return original_state(*args, **kwargs)

        gpu_model_runner.CachedRequestState = state_factory
        try:
            return original(self, scheduler_output)
        finally:
            gpu_model_runner.CachedRequestState = original_state

    update_states._dynamo_llumnix = True
    cls._update_states = update_states


def stamp_prior_output_tokens(request: Any) -> None:
    if request.num_output_tokens <= 0 or request.sampling_params is None:
        return
    params = request.sampling_params
    if params.extra_args is None:
        params.extra_args = {}
    params.extra_args[_PRIOR_OUTPUT_TOKENS_KEY] = list(request.output_token_ids)
