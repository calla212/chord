import os as _ttft_diag_os
if _ttft_diag_os.environ.get("DYN_TTFT_DIAG_ENABLE") == "1":
    import ttft_diag as _ttft_diag
else:
    _ttft_diag = None

import asyncio
import base64
import inspect
import logging
import math
import os


import pickle
import struct
import tempfile
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import (
    Any,
    AsyncIterator,
    Callable,
    Dict,
    Final,
    Generic,
    Iterator,
    NoReturn,
    Optional,
    TypeVar,
)

import torch
from dynamo._core import Context
from dynamo.common.backend import logprobs as _shared_logprobs
from dynamo.common.lora.manager import LoRAInfo, get_lora_manager
from dynamo.common.memory.multimodal_embedding_cache_manager import (
    MultimodalEmbeddingCacheManager,
)
from dynamo.common.multimodal.audio_loader import AudioLoader
from dynamo.common.multimodal.embedding_transfer import (
    LocalEmbeddingReceiver,
    NixlReadEmbeddingReceiver,
    NixlWriteEmbeddingReceiver,
)
from dynamo.common.multimodal.image_loader import ImageLoader
from dynamo.common.multimodal.mm_kwargs_transfer import (
    MmKwargsNixlReceiver,
    MmKwargsReceiver,
    MmKwargsShmReceiver,
    MmKwargsShmTransferMetadata,
    MmKwargsTransferMetadata,
)
from dynamo.common.multimodal.video_loader import VideoLoader
from dynamo.common.rl import (
    RLAdminValidationError,
    RLRouteRegistry,
    env_bool,
    require_lora_load_request,
    require_lora_unload_request,
)
from dynamo.common.utils import nvtx_utils as _nvtx
from dynamo.common.utils.engine_response import normalize_finish_reason
from dynamo.common.utils.input_params import InputParamManager
from dynamo.common.utils.structural_tag import serialize_structural_tag
from dynamo.common.utils.time_section import time_and_log_code_section
from dynamo.llm import (
    KvEventPublisher,
    ModelInput,
    ModelRuntimeConfig,
    ModelType,
    WorkerType,
    lora_name_to_id,
    register_model,
    unregister_model,
)
from dynamo.llm.exceptions import EngineShutdown
from dynamo.runtime import Client
from dynamo.runtime.logging import configure_dynamo_logging
from dynamo.vllm.kv_connector_protocols import (
    KvConnectorProtocol,
    make_kv_connector_protocol,
)

from vllm import PoolingParams
from vllm.config import ModelConfig, VllmConfig
from vllm.inputs import EmbedsPrompt, TextPrompt, TokensPrompt
from vllm.lora.request import LoRARequest
from vllm.multimodal.inputs import MultiModalKwargsItem, PlaceholderRange
from vllm.outputs import RequestOutput
from vllm.renderers.embed_utils import safe_load_prompt_embeds
from vllm.sampling_params import (
    RequestOutputKind,
    SamplingParams,
    StructuredOutputsParams,
)
from vllm.v1.engine.exceptions import EngineDeadError

from .args import Config
from .chord.config import ChordConfig
from .chord.control import ChordControlMixin
from .chord.protocol import BACKEND_REQUEST_ID_KEY, request_metadata
from .chord.request import build_chord_engine_request
from .chord.stream import ChordWaitingReturnAckRegistry, chord_worker_stream
from .constants import DisaggregationMode, EmbeddingTransferMode
from .engine_monitor import VllmEngineMonitor
from .llumnix.config import LlumnixConfig
from .llumnix.control import LlumnixControlMixin
from .llumnix.handoff import (
    MigrationHandoffRegistry,
    apply_resume_metadata,
    migration_envelope,
)
from .multimodal_utils.hash_utils import compute_mm_uuids_from_images
from .multimodal_utils.model import (
    ModelFamily,
    construct_qwen_decode_mm_data,
    resolve_model_family,
)
from .multimodal_utils.models.qwen import (
    build_qwen_embedding_params,
    load_qwen_grid_params,
)
from .multimodal_utils.prefill_worker_utils import MultiModalEmbeddingLoader


IMAGE_URL_KEY: Final = "image_url"
VIDEO_URL_KEY: Final = "video_url"
AUDIO_URL_KEY: Final = "audio_url"
URL_VARIANT_KEY: Final = "Url"
DECODED_VARIANT_KEY: Final = "Decoded"

configure_dynamo_logging()
logger = logging.getLogger(__name__)

_GENERATE_REASONING_SUPPORT_CACHE_ATTR = "_dynamo_generate_reasoning_support"
_DELTA_REQUEST_OUTPUT_KIND = RequestOutputKind.DELTA
_RL_INIT_WEIGHTS_TIMEOUT_ENV = "DYN_RL_INIT_WEIGHTS_TIMEOUT_S"
_RL_INIT_WEIGHTS_TIMEOUT_DEFAULT_S = 30.0
_LORA_LOCK_STRIPES = 64
_DISTRIBUTED_WEIGHT_UPDATE_RESERVED_KEYS: Final = frozenset(
    {
        "allow_unpaused",
        "engine_rpc",
        "reset_prefix_cache",
        "weight_version",
    }
)


def _rl_init_weights_timeout_s() -> float:
    return float(
        os.environ.get(
            _RL_INIT_WEIGHTS_TIMEOUT_ENV,
            str(_RL_INIT_WEIGHTS_TIMEOUT_DEFAULT_S),
        )
    )


class _DeferredAbort:


    def __init__(
        self,
        engine_client: Any,
        request_id: str,
        on_engine_dead: Optional[Any] = None,
    ):
        self._engine_client = engine_client
        self._request_id = request_id


        self._on_engine_dead = on_engine_dead
        self._first_token_received = False
        self._first_token_event = asyncio.Event()


        self._abort_task: Optional[asyncio.Task] = None


        self._abort_exc: Optional[BaseException] = None

    def signal_first_token(self) -> None:

        if not self._first_token_received:
            self._first_token_received = True
            self._first_token_event.set()

    async def abort(self) -> None:


        if self._abort_task is None:
            if self._first_token_received:
                logger.debug(
                    f"Deferred abort: first token already received, "
                    f"aborting request {self._request_id} now"
                )
                self._abort_task = asyncio.create_task(self._run_abort())
            else:
                logger.debug(
                    f"Deferred abort: first token not received for request "
                    f"{self._request_id}, spawning background task"
                )
                self._abort_task = asyncio.create_task(self._wait_and_abort())


        if not self._first_token_received:
            return
        try:


            await asyncio.shield(self._abort_task)
        except asyncio.CancelledError:
            logger.debug(
                f"Deferred abort: shielded from cancellation for request "
                f"{self._request_id}, abort continues in background"
            )

    async def _run_abort(self) -> None:

        try:
            await self._engine_client.abort(self._request_id)
            logger.debug(f"Aborted Request ID: {self._request_id}")
        except Exception as e:


            self._abort_exc = e
            logger.warning(
                f"Deferred abort: engine abort raised for request "
                f"{self._request_id}: {e}"
            )
            if isinstance(e, EngineDeadError) and self._on_engine_dead is not None:
                self._on_engine_dead(e)

    async def _wait_and_abort(self) -> None:

        try:
            await self._first_token_event.wait()
        except Exception:
            pass
        await self._run_abort()

    async def close(self) -> None:


        if self._abort_task is None:
            return

        if not self._first_token_received:

            self._abort_task.cancel()

        try:


            await asyncio.shield(self._abort_task)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.warning(
                f"Deferred abort: cleanup observed error for request "
                f"{self._request_id}: {e}"
            )


@asynccontextmanager
async def _deferred_abort_guard(
    engine_client: Any,
    request_id: str,
    is_decode_only: bool,
    registry: Optional[dict[str, "_DeferredAbort"]] = None,
    on_engine_dead: Optional[Any] = None,
) -> AsyncIterator[Optional[_DeferredAbort]]:


    guard = (
        _DeferredAbort(engine_client, request_id, on_engine_dead)
        if is_decode_only
        else None
    )
    if guard is not None and registry is not None:
        registry[request_id] = guard
    try:
        yield guard
    finally:
        if guard is not None:


            try:
                await guard.close()
            finally:
                if registry is not None:
                    registry.pop(request_id, None)


class VllmEnginePauseController:
    def __init__(self, engine_client: Any):
        self._engine_client = engine_client
        self._is_paused = False
        self._generation_paused = False

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    @property
    def needs_resume_recovery(self) -> bool:
        return self._generation_paused

    async def pause(self, *args: object) -> bool:
        if self._is_paused or self._generation_paused:
            return False

        level = args[0] if args else None
        await self._engine_client.pause_generation()
        self._generation_paused = True
        try:
            if level is None:
                await self._engine_client.sleep()
            else:
                await self._engine_client.sleep(level)
        except Exception:
            try:
                await self._engine_client.resume_generation()
                self._generation_paused = False
            except Exception:
                logger.exception(
                    "Failed to resume generation after native vLLM sleep failure"
                )
            raise
        self._is_paused = True
        return True

    async def resume(self, tags: list[str] | None = None) -> bool:
        if not self._is_paused and not self._generation_paused:
            return False

        if self._is_paused:
            if tags is None:
                await self._engine_client.wake_up()
            else:
                await self._engine_client.wake_up(tags)
        if self._generation_paused:
            await self._engine_client.resume_generation()
            self._generation_paused = False
        return True

    def mark_resumed(self) -> None:
        self._is_paused = False
        self._generation_paused = False


def _pad_mm_hashes_to_64(mm_hashes: list[str]) -> list[str]:


    return [
        h.ljust(64, "0") if isinstance(h, str) and len(h) < 64 else h for h in mm_hashes
    ]


def _compute_mm_uuids(
    multi_modal_data: Dict[str, Any] | None,
) -> Dict[str, list[str]] | None:


    if not multi_modal_data or "image" not in multi_modal_data:
        return None
    images = multi_modal_data["image"]


    if isinstance(images, dict):
        return None
    if not isinstance(images, list):
        images = [images]
    if not images:
        return None
    uuids = compute_mm_uuids_from_images(images)
    return {"image": uuids}


_MIN_FINITE_LOGPROB = -1e30


def _finite_logprob(value: Any) -> float:
    lp = float(value)
    return lp if math.isfinite(lp) else _MIN_FINITE_LOGPROB


def _serialize_prompt_logprobs(
    raw_prompt_logprobs: list,
) -> list:


    result: list = []
    for entry in raw_prompt_logprobs:
        if entry is None:
            result.append(None)
        else:
            converted: Dict[str, Dict[str, Any]] = {}
            for token_id, logprob_obj in entry.items():
                try:
                    key = str(int(token_id))
                except (TypeError, ValueError):


                    continue
                lp_dict: Dict[str, Any] = {
                    "logprob": _finite_logprob(logprob_obj.logprob),
                }
                rank = getattr(logprob_obj, "rank", None)
                if rank is not None:
                    lp_dict["rank"] = int(rank)
                decoded = getattr(logprob_obj, "decoded_token", None)
                if decoded is not None:
                    lp_dict["decoded_token"] = decoded
                converted[key] = lp_dict
            result.append(converted)
    return result


def _attach_prompt_logprobs_engine_data(
    tok: Dict[str, Any], prompt_logprobs: list
) -> None:
    engine_data = tok.setdefault("engine_data", {})
    if isinstance(engine_data, dict):
        engine_data["prompt_logprobs"] = prompt_logprobs


def _attach_routed_experts_engine_data(
    tok: Dict[str, Any], routed_experts: Dict[str, Any]
) -> None:


    engine_data = tok.setdefault("engine_data", {})
    if isinstance(engine_data, dict):
        engine_data["routed_experts"] = routed_experts


def _iter_nvext_sources(request: Dict[str, Any]) -> Iterator[Dict[str, Any]]:


    extra_args = request.get("extra_args")
    for source in (
        request.get("nvext"),
        extra_args.get("nvext") if isinstance(extra_args, dict) else None,
    ):
        if isinstance(source, dict):
            yield source


def _nvext_extra_field_requested(request: Dict[str, Any], field: str) -> bool:

    return any(
        isinstance(source.get("extra_fields"), list) and field in source["extra_fields"]
        for source in _iter_nvext_sources(request)
    )


def _apply_nvext_cache_salt(request: Dict[str, Any], prompt: Any) -> None:
    if not isinstance(prompt, dict):
        return
    for source in _iter_nvext_sources(request):
        cache_salt = source.get("cache_salt")
        if cache_salt is not None:
            prompt["cache_salt"] = cache_salt
            return


def _prompt_token_ids_for_engine_data(
    request: Dict[str, Any],
    prompt: Any,
) -> list[int]:
    prompt_token_ids = (
        prompt.get("prompt_token_ids")
        if isinstance(prompt, dict)
        else request.get("token_ids")
    )
    return list(prompt_token_ids or [])


def _flatten_logprobs(
    log_probs: Any,
) -> Optional[list[float]]:


    if log_probs is None:
        return None
    if not isinstance(log_probs, list):
        return None
    out: list[float] = []


    pending: deque = deque(log_probs)
    while pending:
        item = pending.popleft()
        if isinstance(item, bool):

            continue
        if isinstance(item, (int, float)):
            out.append(_finite_logprob(item))
        elif isinstance(item, list):
            pending.extendleft(reversed(item))
        elif isinstance(item, dict) and "logprob" in item:
            try:
                out.append(_finite_logprob(item["logprob"]))
            except (TypeError, ValueError):
                continue
    return out or None


def _is_token_in_request(request: Dict[str, Any]) -> bool:
    return any(
        source.get("token_data") or source.get("token_in")
        for source in _iter_nvext_sources(request)
    )


def _accumulate_engine_data(
    tok: Dict[str, Any],
    request_prompt_token_ids: Optional[list[int]],
    accumulated_token_ids: dict[int, list[int]],
    accumulated_log_probs: dict[int, list[float]],
) -> None:
    output_index = int(tok.get("index") or 0)
    token_accumulator = accumulated_token_ids.setdefault(output_index, [])
    logprob_accumulator = accumulated_log_probs.setdefault(output_index, [])

    new_token_ids = tok.get("token_ids")
    if isinstance(new_token_ids, list):
        for t in new_token_ids:
            try:
                token_accumulator.append(int(t))
            except (TypeError, ValueError):


                continue

    flat_lp = _flatten_logprobs(tok.get("log_probs"))
    if flat_lp:
        logprob_accumulator.extend(flat_lp)

    finish_reason = tok.get("finish_reason")
    if finish_reason is None:
        return


    finished = not (
        isinstance(finish_reason, str) and finish_reason.startswith("error")
    )

    engine_data: Dict[str, Any] = dict(tok.get("engine_data") or {})
    engine_data.update(
        {
            "completion_token_ids": list(token_accumulator),
            "finished": finished,
        }
    )
    if logprob_accumulator:


        if len(logprob_accumulator) == len(token_accumulator):
            engine_data["completion_logprobs"] = list(logprob_accumulator)
        else:
            logger.warning(
                "Dropping completion_logprobs for output index %d: logprob "
                "count %d != token count %d (misaligned)",
                output_index,
                len(logprob_accumulator),
                len(token_accumulator),
            )
    if request_prompt_token_ids:
        engine_data["prompt_token_ids"] = list(request_prompt_token_ids)
    tok["engine_data"] = engine_data


def _serialize_routed_experts(
    routed_experts: Any, start: int = 0
) -> Optional[Dict[str, Any]]:
    if routed_experts is None:
        return None

    shape = getattr(routed_experts, "shape", None)
    tobytes = getattr(routed_experts, "tobytes", None)
    if shape is None or not callable(tobytes):
        logger.warning(
            "Unable to serialize routed_experts of type %s",
            type(routed_experts).__name__,
        )
        return None

    return {

        "data": base64.b64encode(tobytes()).decode("ascii"),
        "shape": [int(dim) for dim in shape],


        "start": int(start),


        "dtype": str(getattr(routed_experts, "dtype", "")),
    }


def build_sampling_params(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
    model_max_len: int | None = None,
    enable_rl: bool = False,
) -> SamplingParams:


    if enable_rl and _is_token_in_request(request):

        sampling_params = SamplingParams()
    else:
        sampling_params = SamplingParams(**default_sampling_params)


    sampling_options = dict(request.get("sampling_options") or {})
    extra_args = request.get("extra_args") or {}
    if isinstance(extra_args, dict):
        passthrough_sampling_options = extra_args.get("sampling_options")
        if isinstance(passthrough_sampling_options, dict):
            sampling_options.update(passthrough_sampling_options)
    guided_decoding = sampling_options.get("guided_decoding")
    if guided_decoding is not None and isinstance(guided_decoding, dict):
        sampling_params.structured_outputs = StructuredOutputsParams(
            json=guided_decoding.get("json"),
            regex=guided_decoding.get("regex"),
            choice=guided_decoding.get("choice"),
            grammar=guided_decoding.get("grammar"),
            whitespace_pattern=guided_decoding.get("whitespace_pattern"),
            structural_tag=serialize_structural_tag(
                guided_decoding.get("structural_tag")
            ),
        )


    for key, value in sampling_options.items():

        if key == "guided_decoding":
            continue
        if key == "bad_words_token_ids" and value is not None:


            if not hasattr(sampling_params, "_bad_words_token_ids"):
                raise AttributeError(
                    "vLLM SamplingParams._bad_words_token_ids missing; TITO "
                    "bad_words_token_ids passthrough needs updating for this "
                    "vLLM version"
                )
            sampling_params._bad_words_token_ids = value
            continue
        if value is not None and hasattr(sampling_params, key):
            setattr(sampling_params, key, value)


    reps = getattr(sampling_params, "routed_experts_prompt_start", None)
    if reps is not None and (
        isinstance(reps, bool) or not isinstance(reps, int) or reps < 0
    ):
        logger.warning(
            "Ignoring invalid routed_experts_prompt_start=%r (want non-negative int)",
            reps,
        )
        sampling_params.routed_experts_prompt_start = 0


    for key, value in request.get("stop_conditions", {}).items():
        if value is not None and hasattr(sampling_params, key):

            if key == "stop":
                continue
            setattr(sampling_params, key, value)
        if (
            key == "stop_token_ids_hidden"
            and value is not None
            and hasattr(sampling_params, "stop_token_ids")
        ):
            existing = sampling_params.stop_token_ids or []
            sampling_params.stop_token_ids = list(set(existing).union(value))


        if (
            key == "max_thinking_tokens"
            and value is not None
            and hasattr(sampling_params, "thinking_token_budget")
        ):
            sampling_params.thinking_token_budget = value


    output_options = request.get("output_options", {}) or {}
    logprobs, prompt_logprobs = _shared_logprobs.parse_logprob_options(output_options)
    if logprobs is not None:
        sampling_params.logprobs = logprobs
    if prompt_logprobs is not None:
        sampling_params.prompt_logprobs = prompt_logprobs


    provided_max_tokens = request.get("stop_conditions", {}).get("max_tokens", None)
    token_ids = request.get("token_ids", [])
    input_length = len(token_ids)
    if model_max_len is not None and provided_max_tokens is None:

        dynamic_default = max(1, model_max_len - input_length)
        configured_default = default_sampling_params.get("max_tokens", dynamic_default)
        sampling_params.max_tokens = min(configured_default, dynamic_default)


    sampling_params.detokenize = False
    sampling_params.output_kind = _DELTA_REQUEST_OUTPUT_KIND

    return sampling_params


def build_sampling_params_openai(
    request: Dict[str, Any],
    default_sampling_params: Dict[str, Any],
) -> SamplingParams:


    sampling_params = SamplingParams(**default_sampling_params)
    sampling_params.detokenize = True


    openai_mapping = {
        "n": "n",
        "temperature": "temperature",
        "top_p": "top_p",
        "presence_penalty": "presence_penalty",
        "frequency_penalty": "frequency_penalty",
        "seed": "seed",
        "top_k": "top_k",
        "repetition_penalty": "repetition_penalty",
        "min_p": "min_p",
        "length_penalty": "length_penalty",
        "use_beam_search": "use_beam_search",
    }

    for req_key, param_key in openai_mapping.items():
        if req_key in request and request[req_key] is not None:
            if hasattr(sampling_params, param_key):
                setattr(sampling_params, param_key, request[req_key])


    if "max_tokens" in request and request["max_tokens"] is not None:
        sampling_params.max_tokens = request["max_tokens"]


    if "stop" in request and request["stop"] is not None:
        sampling_params.stop = request["stop"]


    if "ignore_eos" in request and request["ignore_eos"] is not None:
        sampling_params.ignore_eos = request["ignore_eos"]


    if "min_tokens" in request and request["min_tokens"] is not None:
        sampling_params.min_tokens = request["min_tokens"]

    nvext_max_thinking_tokens = (request.get("nvext") or {}).get("max_thinking_tokens")
    if nvext_max_thinking_tokens is not None and hasattr(
        sampling_params, "thinking_token_budget"
    ):
        sampling_params.thinking_token_budget = nvext_max_thinking_tokens

    return sampling_params


def _engine_generate_reasoning_kwargs(
    engine_client: Any,
    reasoning_ended: bool | None,
    reasoning_parser_kwargs: dict[str, Any] | None,
) -> dict[str, Any]:
    if reasoning_ended is None and reasoning_parser_kwargs is None:
        return {}

    support = _engine_generate_reasoning_support(engine_client)
    if support is None:
        return {}
    accepts_reasoning_ended, accepts_reasoning_parser_kwargs = support

    kwargs: dict[str, Any] = {}
    if accepts_reasoning_ended:
        kwargs["reasoning_ended"] = reasoning_ended
    if accepts_reasoning_parser_kwargs:
        kwargs["reasoning_parser_kwargs"] = reasoning_parser_kwargs

    if not kwargs:
        logger.debug(
            "vLLM generate does not accept reasoning parser kwargs; "
            "running without request-local reasoning parser metadata"
        )
    return kwargs


def _engine_generate_reasoning_support(
    engine_client: Any,
) -> tuple[bool, bool] | None:
    try:
        cached = vars(engine_client).get(_GENERATE_REASONING_SUPPORT_CACHE_ATTR)
    except TypeError:
        cached = None
    if cached is not None:
        return cached

    try:
        parameters = inspect.signature(engine_client.generate).parameters
    except (TypeError, ValueError):
        logger.debug(
            "Unable to inspect vLLM generate signature; dropping reasoning parser kwargs"
        )
        return None

    accepts_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in parameters.values()
    )
    support = (
        accepts_kwargs or "reasoning_ended" in parameters,
        accepts_kwargs or "reasoning_parser_kwargs" in parameters,
    )
    try:
        setattr(engine_client, _GENERATE_REASONING_SUPPORT_CACHE_ATTR, support)
    except Exception:
        pass
    return support


def _request_reasoning_metadata(
    request: Mapping[str, Any],
) -> tuple[bool | None, dict[str, Any] | None]:
    reasoning_ended = request.get("reasoning_ended")
    reasoning_parser_kwargs = request.get("reasoning_parser_kwargs")

    extra_args = request.get("extra_args")
    if isinstance(extra_args, dict):
        if reasoning_ended is None:
            reasoning_ended = extra_args.get("reasoning_ended")
        if reasoning_parser_kwargs is None:
            reasoning_parser_kwargs = extra_args.get("reasoning_parser_kwargs")

    return reasoning_ended, reasoning_parser_kwargs


def get_dp_range_for_worker(vllm_config: VllmConfig) -> tuple[int, int]:


    if vllm_config.parallel_config.data_parallel_external_lb:

        return (vllm_config.parallel_config.data_parallel_rank, 1)
    elif vllm_config.parallel_config.data_parallel_hybrid_lb:

        return (
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size_local,
        )
    else:

        logger.warning(
            "vLLM selects internal DP load balancing. If you are launching multiple workers for DP deployment,"
            " hybrid or external load balancing is recommended."
        )
        return (
            vllm_config.parallel_config.data_parallel_rank,
            vllm_config.parallel_config.data_parallel_size,
        )


RequestT = TypeVar("RequestT")
ResponseT = TypeVar("ResponseT")


class BaseWorkerHandler(
    ChordControlMixin, LlumnixControlMixin, ABC, Generic[RequestT, ResponseT]
):


    _llumnix_enabled = False
    _llumnix_handoffs = None
    _chord_enabled = False
    _chord_waiting_return_acks = None


    _use_unified_vision_chunk: bool = False
    _scale_ep_in_progress: bool = False

    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        model_config: ModelConfig | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Optional[Client] = None,
    ):
        self.runtime = runtime
        self.engine_client = engine
        self.default_sampling_params = default_sampling_params
        self.kv_publishers: list[KvEventPublisher] | None = None
        self.fpm_relays: list | None = None
        self.generate_endpoint = generate_endpoint
        self._llumnix_enabled = LlumnixConfig.from_env().enabled
        self._endpoint = generate_endpoint
        self._llumnix_worker_id = (
            int(generate_endpoint.connection_id())
            if self._llumnix_enabled and generate_endpoint is not None
            else -1
        )
        self._llumnix_handoffs = (
            MigrationHandoffRegistry(self._llumnix_rollback_request)
            if self._llumnix_enabled
            else None
        )
        self._chord_enabled = ChordConfig.from_env().enabled
        self._chord_worker_id = (
            int(generate_endpoint.connection_id())
            if self._chord_enabled and generate_endpoint is not None
            else -1
        )
        self._chord_waiting_return_acks = (
            ChordWaitingReturnAckRegistry() if self._chord_enabled else None
        )
        self.config = config
        self.engine_monitor = VllmEngineMonitor(runtime, engine, shutdown_event)
        self.temp_dirs: list[tempfile.TemporaryDirectory] = []
        self.model_max_len = model_max_len
        self.model_config = model_config
        self.enable_multimodal = enable_multimodal

        self.loaded_loras: dict[str, LoRAInfo] = {}


        self._engine_loaded_loras: set[str] = set()


        self._lora_load_locks = [asyncio.Lock() for _ in range(_LORA_LOCK_STRIPES)]
        self._paused: bool = False
        self._weight_version: str = "initial"

        self.image_loader = ImageLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        self.audio_loader = AudioLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        self.video_loader = VideoLoader(
            enable_frontend_decoding=enable_frontend_decoding
        )
        self.embedding_loader = self.init_embedding_loader(config, encode_worker_client)

        self.use_vllm_tokenizer = use_vllm_tokenizer

        self.dp_range = get_dp_range_for_worker(self.engine_client.vllm_config)
        self._pause_controller = VllmEnginePauseController(self.engine_client)
        self._pause_lock = asyncio.Lock()


        self._deferred_aborts: dict[str, _DeferredAbort] = {}
        self._mm_kwargs_receiver: MmKwargsNixlReceiver | None = None


        self._use_unified_vision_chunk = bool(
            getattr(
                self.engine_client.vllm_config.model_config.hf_config,
                "use_unified_vision_chunk",
                False,
            )
        )


        self._scale_ep_lock = asyncio.Lock()
        self._scale_ep_in_progress = False


        tokenizer = None
        if use_vllm_tokenizer and hasattr(engine, "tokenizer"):
            tokenizer = engine.tokenizer
        self.input_param_manager = InputParamManager(tokenizer)


        self.shutdown_event = shutdown_event


        self.rl_route_registry = RLRouteRegistry(self.runtime, logger_=logger)

    def _shutdown_worker(self) -> NoReturn:
        logger.warning("Initiating Dynamo Runtime shutdown.")
        self.runtime.shutdown()
        os._exit(1)

    def _shutdown_on_engine_dead(self, e: EngineDeadError) -> NoReturn:
        logger.error(f"vLLM EngineDeadError: {e}")
        self._shutdown_worker()

    def init_embedding_loader(
        self, config: Config, encode_worker_client: Optional[Client] = None
    ) -> Optional[MultiModalEmbeddingLoader]:


        if encode_worker_client is None:
            return None
        logger.warning(
            "Separate multimodal encode-worker routing only applies to image_url "
            "inputs. video_url inputs are not sent to the encode worker and will "
            "be processed on the prefill/PD worker instead."
        )


        self.encode_worker_client = encode_worker_client
        if config.embedding_transfer_mode == EmbeddingTransferMode.LOCAL:
            self.embedding_receiver = LocalEmbeddingReceiver()
        elif config.embedding_transfer_mode == EmbeddingTransferMode.NIXL_WRITE:
            self.embedding_receiver = NixlWriteEmbeddingReceiver()
        elif config.embedding_transfer_mode == EmbeddingTransferMode.NIXL_READ:


            self.embedding_receiver = NixlReadEmbeddingReceiver(max_items=0)
        else:
            raise ValueError(
                f"Invalid embedding transfer mode: {config.embedding_transfer_mode}"
            )


        self.embedding_cache_manager: MultimodalEmbeddingCacheManager | None = None
        if config.multimodal_embedding_cache_capacity_gb > 0:
            capacity_bytes = int(
                config.multimodal_embedding_cache_capacity_gb * 1024**3
            )
            self.embedding_cache_manager = MultimodalEmbeddingCacheManager(
                capacity_bytes
            )
        return MultiModalEmbeddingLoader(
            encode_worker_client=self.encode_worker_client,
            receiver=self.embedding_receiver,
            embedding_cache_manager=self.embedding_cache_manager,
        )

    async def sleep(self, body: dict) -> dict:


        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        level = body.get("level", 1)
        async with self._pause_lock:
            if self._pause_controller.is_paused:
                return {
                    "status": "ok",
                    "message": "Engine already sleeping",
                }
            if self._pause_controller.needs_resume_recovery:
                return {
                    "status": "error",
                    "message": "wake_up required before retrying sleep",
                }

            unregistered = False
            try:

                if self.generate_endpoint is not None:
                    await self.generate_endpoint.unregister_endpoint_instance()
                    unregistered = True
                    logger.info(
                        "[Sleep] Unregistered endpoint from discovery - worker removed from routing pool"
                    )


                if not await self._pause_controller.pause(level):
                    return {
                        "status": "ok",
                        "message": "Engine already sleeping",
                    }

                return {
                    "status": "ok",
                    "message": f"Engine slept (level={level})",
                }
            except Exception as e:
                logger.error(f"Failed to sleep engine: {e}")


                if (
                    unregistered
                    and not self._pause_controller.is_paused
                    and not self._pause_controller.needs_resume_recovery
                    and self.generate_endpoint is not None
                ):
                    try:
                        await self.generate_endpoint.register_endpoint_instance()
                        logger.info(
                            "[Sleep] Re-registered endpoint after failed sleep rollback"
                        )
                    except Exception as reg_err:
                        logger.error(
                            f"Failed to re-register endpoint after sleep failure: {reg_err}"
                        )
                return {"status": "error", "message": str(e)}

    async def scale_elastic_ep(self, body: dict) -> dict:


        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        new_dp_size = body.get("new_data_parallel_size")
        if new_dp_size is None:
            return {
                "status": "error",
                "message": "Missing required field: new_data_parallel_size",
            }
        try:
            new_dp_size = int(new_dp_size)
        except (TypeError, ValueError):
            return {
                "status": "error",
                "message": f"new_data_parallel_size must be an integer, got: {new_dp_size!r}",
            }
        if new_dp_size < 2:
            return {
                "status": "error",
                "message": (
                    "new_data_parallel_size must be >= 2 when elastic EP/ePLB is enabled"
                ),
            }

        logger.info(f"[ElasticEP] Scaling to new_data_parallel_size={new_dp_size}")


        async with self._scale_ep_lock:
            if self._scale_ep_in_progress:
                msg = (
                    "A scale_elastic_ep operation is already in progress; "
                    f"rejecting concurrent request for new_data_parallel_size={new_dp_size}"
                )
                logger.warning("[ElasticEP] %s", msg)
                return {"status": "error", "message": msg}
            self._scale_ep_in_progress = True

        try:


            import ray
            import ray.util.state as _ray_util_state

            class _NodeInfo:
                __slots__ = ("node_id", "node_ip")

                def __init__(self, d: dict) -> None:
                    self.node_ip: str = d["NodeManagerAddress"]
                    self.node_id: str = d["NodeID"]

            original_list_nodes = _ray_util_state.list_nodes
            try:
                _ray_util_state.list_nodes = lambda **kw: [
                    _NodeInfo(n) for n in ray.nodes() if n.get("Alive", False)
                ]
                await self.engine_client.scale_elastic_ep(new_dp_size)
            finally:
                _ray_util_state.list_nodes = original_list_nodes

            logger.info(f"[ElasticEP] Scaling to dp={new_dp_size} complete")
            return {
                "status": "ok",
                "message": f"Scaled to data_parallel_size={new_dp_size}",
                "new_data_parallel_size": new_dp_size,
            }
        except Exception as e:
            logger.error(f"[ElasticEP] Scaling failed: {e}")
            return {"status": "error", "message": str(e)}
        finally:
            async with self._scale_ep_lock:
                self._scale_ep_in_progress = False

    async def wake_up(self, body: dict) -> dict:


        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        tags = body.get("tags")
        async with self._pause_lock:
            needs_recovery = self._pause_controller.needs_resume_recovery
            if not self._pause_controller.is_paused and not needs_recovery:
                return {"status": "ok", "message": "Engine already awake"}

            try:

                await self._pause_controller.resume(tags)
                if self.generate_endpoint is not None:
                    await self.generate_endpoint.register_endpoint_instance()
                    logger.info(
                        "[Wake] Re-registered endpoint to discovery - worker added back to routing pool"
                    )
                self._pause_controller.mark_resumed()

                return {
                    "status": "ok",
                    "message": "Engine woke",
                }
            except Exception as e:
                logger.error(f"Failed to wake up engine: {e}")
                return {"status": "error", "message": str(e)}

    async def start_profile(self, body: dict) -> dict:


        profile_prefix = body.get("profile_prefix")
        try:
            await self.engine_client.start_profile(profile_prefix=profile_prefix)
            return {"status": "ok", "message": "Profiling started"}
        except Exception as e:
            logger.error(f"Failed to start profiling: {e}")
            return {"status": "error", "message": str(e)}

    async def stop_profile(self, body: dict) -> dict:


        try:
            await self.engine_client.stop_profile()
            return {"status": "ok", "message": "Profiling stopped"}
        except Exception as e:
            logger.error(f"Failed to stop profiling: {e}")
            return {"status": "error", "message": str(e)}

    async def rl_dispatch(self, request=None):

        async for response in self.rl_route_registry.dispatch_stream(request):
            yield response

    async def liveness_probe(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        try:
            if hasattr(self.engine_client, "check_health"):
                await self.engine_client.check_health()
            else:
                await self.engine_client.collective_rpc("liveness_probe", kwargs={})
            return {"status": "ok", "alive": True}
        except EngineDeadError as e:
            self._shutdown_on_engine_dead(e)
        except Exception as e:
            logger.warning(f"[RL] liveness_probe failed: {e}")
            return {"status": "error", "alive": False, "message": str(e)}

    async def pause_generation(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        mode = body.get("mode", "keep")
        clear_cache = bool(body.get("clear_cache", False))
        if mode not in ("keep", "wait", "abort"):
            return {
                "status": "error",
                "message": f"Invalid mode '{mode}'; expected keep|wait|abort",
            }
        async with self._pause_lock:
            try:
                try:
                    await self.engine_client.pause_generation(
                        mode=mode, clear_cache=clear_cache
                    )
                except TypeError:
                    await self.engine_client.pause_generation()
                    if clear_cache:
                        await self.engine_client.reset_prefix_cache()
                self._paused = True
                logger.info(
                    f"[RL] Engine paused (mode={mode}, clear_cache={clear_cache})"
                )
                return {
                    "status": "ok",
                    "message": "Engine paused",
                    "mode": mode,
                    "clear_cache": clear_cache,
                }
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] Failed to pause: {e}")
                return {"status": "error", "message": str(e)}

    async def resume_generation(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }


        async with self._pause_lock:
            try:
                await self.engine_client.resume_generation()
                self._paused = False
                logger.info("[RL] Engine resumed")
                return {"status": "ok", "message": "Engine resumed"}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] Failed to resume: {e}")
                return {"status": "error", "message": str(e)}

    async def flush_cache(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }


        async with self._pause_lock:
            try:
                await self.engine_client.reset_prefix_cache()
                logger.debug("[RL] Prefix cache flushed")
                return {"status": "ok", "message": "Cache flushed"}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] Failed to flush cache: {e}")
                return {"status": "error", "message": str(e)}

    async def abort_request(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        request_id = body.get("request_id")
        if not request_id:
            return {"status": "error", "message": "Missing 'request_id' in body"}
        try:
            guard = self._deferred_aborts.get(request_id)
            if guard is not None:


                await guard.abort()


                abort_exc = guard._abort_exc
                if abort_exc is not None:
                    if isinstance(abort_exc, EngineDeadError):
                        self._shutdown_on_engine_dead(abort_exc)
                    return {
                        "status": "error",
                        "request_id": request_id,
                        "message": f"abort failed: {abort_exc}",
                    }
            else:
                await self.engine_client.abort(request_id)
            logger.debug(f"[RL] Aborted request {request_id}")
            return {"status": "ok", "request_id": request_id}
        except EngineDeadError as e:
            self._shutdown_on_engine_dead(e)
        except Exception as e:
            logger.error(f"[RL] Failed to abort request {request_id}: {e}")
            return {"status": "error", "message": str(e)}

    async def get_weight_version(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        return {"status": "ok", "version": getattr(self, "_weight_version", "initial")}

    async def update_weights_from_disk(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }


        async with self._pause_lock:
            if not getattr(self, "_paused", False):
                return {
                    "status": "error",
                    "message": (
                        "Worker must be paused via pause_generation() before "
                        "updating weights. Call pause_generation() first, then "
                        "update, then resume_generation()."
                    ),
                }
            path = body.get("model_path")
            if not path:
                return {"status": "error", "message": "Missing 'model_path' in body"}
            version = body.get("weight_version", "unknown")
            rpc = body.get("engine_rpc", "reload_weights")
            kwargs = (
                {"weights_path": path}
                if rpc == "reload_weights"
                else {"weight_path": path}
            )
            try:
                await self.engine_client.collective_rpc(rpc, kwargs=kwargs)


                await self.engine_client.reset_prefix_cache()
                self._weight_version = version
                logger.info(
                    f"[RL] Weights loaded from {path} (version={version}, rpc={rpc})"
                )
                return {"status": "ok", "version": version}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] update_weights_from_disk failed: {e}")
                return {"status": "error", "message": str(e)}

    async def update_weights_from_distributed(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        allow_unpaused = body.get("allow_unpaused", False)
        reset_prefix_cache = body.get("reset_prefix_cache", True)
        if not isinstance(allow_unpaused, bool):
            return {
                "status": "error",
                "message": "'allow_unpaused' must be a boolean",
            }
        if not isinstance(reset_prefix_cache, bool):
            return {
                "status": "error",
                "message": "'reset_prefix_cache' must be a boolean",
            }
        if allow_unpaused and reset_prefix_cache:
            return {
                "status": "error",
                "message": (
                    "Unpaused weight updates cannot reset the prefix cache. "
                    "Set 'reset_prefix_cache' to false or pause generation first."
                ),
            }
        async with self._pause_lock:
            if not self._paused and not allow_unpaused:
                return {
                    "status": "error",
                    "message": (
                        "Worker must be paused via pause_generation() before "
                        "updating weights. Call pause_generation() first, then "
                        "update, then resume_generation()."
                    ),
                }
            version = body.get("weight_version", "unknown")
            rpc = body.get("engine_rpc", "update_weights_from_path")
            rpc_kwargs = {
                k: v
                for k, v in body.items()
                if k not in _DISTRIBUTED_WEIGHT_UPDATE_RESERVED_KEYS
            }
            try:
                await self.engine_client.collective_rpc(rpc, kwargs=rpc_kwargs)
                if reset_prefix_cache:


                    await self.engine_client.reset_prefix_cache()
                self._weight_version = version
                logger.info(
                    f"[RL] Weights received via distributed "
                    f"(version={version}, rpc={rpc})"
                )
                return {"status": "ok", "version": version}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] update_weights_from_distributed failed: {e}")
                return {"status": "error", "message": str(e)}

    async def update_weights_from_tensor(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        return {
            "status": "error",
            "message": "update_weights_from_tensor is not implemented",
        }

    async def init_weights_update_group(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        rpc = body.get("engine_rpc", "init_broadcaster")
        kwargs = {k: v for k, v in body.items() if k != "engine_rpc"}
        async with self._pause_lock:
            try:
                timeout_s = _rl_init_weights_timeout_s()
                rpc_task = asyncio.create_task(
                    self.engine_client.collective_rpc(rpc, kwargs=kwargs)
                )
                try:
                    done, _ = await asyncio.wait({rpc_task}, timeout=timeout_s)
                except asyncio.CancelledError:
                    rpc_task.cancel()
                    await asyncio.gather(rpc_task, return_exceptions=True)
                    raise
                if rpc_task not in done:
                    rpc_task.cancel()
                    await asyncio.gather(rpc_task, return_exceptions=True)
                    logger.error(
                        f"[RL] init_weights_update_group timed out after "
                        f"{timeout_s:.1f} seconds (rpc={rpc}); terminating the "
                        "worker because EngineCore may still be blocked"
                    )
                    self._shutdown_worker()

                await rpc_task
                logger.info(f"[RL] Weight update group initialized (rpc={rpc})")
                return {"status": "ok", "message": "Weight update group initialized"}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] init_weights_update_group failed: {e}")
                return {"status": "error", "message": str(e)}

    async def destroy_weights_update_group(self, body: dict) -> dict:

        if body is None:
            body = {}
        elif not isinstance(body, dict):
            return {
                "status": "error",
                "message": "request body must be a JSON object",
            }
        rpc = body.get("engine_rpc", "destroy_broadcaster")
        kwargs = {k: v for k, v in body.items() if k != "engine_rpc"}
        async with self._pause_lock:
            try:
                await self.engine_client.collective_rpc(rpc, kwargs=kwargs)
                logger.info(f"[RL] Weight update group destroyed (rpc={rpc})")
                return {"status": "ok", "message": "Weight update group destroyed"}
            except EngineDeadError as e:
                self._shutdown_on_engine_dead(e)
            except Exception as e:
                logger.error(f"[RL] destroy_weights_update_group failed: {e}")
                return {"status": "error", "message": str(e)}

    @abstractmethod
    def generate(self, request: RequestT, context: Context) -> AsyncIterator[ResponseT]:
        raise NotImplementedError

    async def _monitor_abort(self, context, request_id, is_prefill, abort_guard=None):


        try:

            wait_for = [context.async_killed_or_stopped()]
            shutdown_task = None

            if self.shutdown_event:

                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                wait_for.append(shutdown_task)


            done, pending = await asyncio.wait(
                wait_for,
                return_when=asyncio.FIRST_COMPLETED,
            )


            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


            logger.debug(
                f"Aborting {'Prefill ' if is_prefill else ''}Request ID: {request_id}"
            )


            if abort_guard is not None:


                await abort_guard.abort()
            else:


                try:
                    await asyncio.shield(self.engine_client.abort(request_id))
                except asyncio.CancelledError:
                    logger.debug(
                        f"Abort shielded from cancellation for request "
                        f"{request_id}, continuing in background"
                    )
                logger.debug(
                    f"Aborted {'Prefill ' if is_prefill else ''}Request ID: {request_id}"
                )


            if shutdown_task and shutdown_task in done:
                raise EngineShutdown("Engine was shut down during generation.")

        except asyncio.CancelledError:

            pass
        except EngineShutdown:
            raise
        except Exception as e:
            logger.error(f"Error in abort monitor for request {request_id}: {e}")

    @asynccontextmanager
    async def _abort_monitor(
        self, context, request_id, is_prefill=False, abort_guard=None
    ):


        task = asyncio.create_task(
            self._monitor_abort(context, request_id, is_prefill, abort_guard)
        )
        try:
            yield task
        finally:

            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:

                task.result()

    async def clear_kv_blocks(self, request=None):
        try:
            reset_successful = await self.engine_client.reset_prefix_cache(
                reset_connector=True
            )
            if reset_successful is False:
                yield {"status": "error", "message": "KV cache reset failed"}
                return
            yield {"status": "success", "message": "KV cache cleared"}
        except Exception as e:
            yield {"status": "error", "message": str(e)}


    def add_temp_dir(self, temp_dir: tempfile.TemporaryDirectory) -> None:

        if temp_dir is not None:
            self.temp_dirs.append(temp_dir)

    def _to_local_dp_rank(self, dp_rank: int | None) -> int | None:

        if dp_rank is None:
            return None
        if dp_rank < self.dp_range[0] or dp_rank >= self.dp_range[0] + self.dp_range[1]:
            logger.warning(
                f"Received DP rank {dp_rank} is out of range [{self.dp_range[0]} - {self.dp_range[0] + self.dp_range[1]}), fallback to vLLM internal DP selection"
            )
            return None
        local_dp_rank = (dp_rank - self.dp_range[0]) % self.dp_range[1]
        logger.debug(
            f"Converted global DP rank {dp_rank} to local DP rank {local_dp_rank}"
        )
        return local_dp_rank

    def _resolve_lora_request(self, model_name: str | None) -> LoRARequest | None:

        if model_name and (lora := self.loaded_loras.get(model_name)):
            return LoRARequest(
                lora_name=model_name,
                lora_int_id=lora.id,
                lora_path=lora.path,
            )
        return None

    def _track_lora_request_activation(self, lora_request: LoRARequest | None) -> None:

        if lora_request is not None:
            self._engine_loaded_loras.add(lora_request.lora_name)

    @staticmethod
    def _is_lora_not_loaded_error(error: Exception) -> bool:

        message = str(error).lower()
        return "not loaded" in message or "not found" in message

    def _get_lora_lock(self, lora_name: str) -> asyncio.Lock:

        return self._lora_load_locks[hash(lora_name) % _LORA_LOCK_STRIPES]

    async def _generate_with_lora_admission_lock(
        self,
        lora_request: LoRARequest | None,
        create_generator: Callable[[LoRARequest | None], AsyncIterator[Any]],
    ) -> AsyncIterator[Any]:


        if lora_request is None or self._preload_lora_into_engine():
            self._track_lora_request_activation(lora_request)
            async for result in create_generator(lora_request):
                yield result
            return

        lock = self._get_lora_lock(lora_request.lora_name)
        async with lock:


            admitted_lora_request = self._resolve_lora_request(lora_request.lora_name)
            if admitted_lora_request is None:
                logger.warning(
                    "LoRA adapter %s was unloaded before vLLM admission; "
                    "rejecting the request",
                    lora_request.lora_name,
                )
                raise ValueError(
                    f"unknown model or LoRA adapter: '{lora_request.lora_name}'"
                )
            generator = create_generator(admitted_lora_request)
            self._track_lora_request_activation(admitted_lora_request)
            try:
                first_result = await anext(generator)
            except StopAsyncIteration:
                return

        yield first_result
        async for result in generator:
            yield result

    def _preload_lora_into_engine(self) -> bool:


        return self.config.disaggregation_mode != DisaggregationMode.PREFILL

    async def load_lora(self, request=None):


        try:
            try:
                lora_name, lora_uri = require_lora_load_request(request)
            except RLAdminValidationError as e:
                yield {"status": "error", "message": str(e)}
                return


            logger.debug(f"load_lora request keys: {list(request.keys())}")
            logger.debug(f"load_lora request: {request}")


            lora_manager = get_lora_manager()
            if lora_manager is None:
                yield {
                    "status": "error",
                    "message": "LoRAManager not initialized. Set DYN_LORA_ENABLED=true to enable URI-based LoRA loading.",
                }
                return


            lock = self._get_lora_lock(lora_name)
            async with lock:
                try:
                    old_info = self.loaded_loras.get(lora_name)
                    hot_swap_enabled = env_bool("DYN_LORA_HOTSWAP_ENABLED")
                    is_hot_swap = old_info is not None and hot_swap_enabled
                    old_engine_loaded = lora_name in self._engine_loaded_loras

                    if old_info is not None and not hot_swap_enabled:
                        logger.info(
                            f"LoRA adapter already loaded: {lora_name} "
                            f"with ID {old_info.id}"
                        )
                        yield {
                            "status": "success",
                            "message": f"LoRA adapter '{lora_name}' already loaded",
                            "lora_name": lora_name,
                            "lora_id": old_info.id,
                            "hot_swap": False,
                        }
                        return

                    logger.info(
                        f"Downloading LoRA adapter: {lora_name} from {lora_uri}"
                    )
                    download_result = await lora_manager.download_lora(lora_uri)

                    if download_result["status"] != "success":
                        yield {
                            "status": "error",
                            "message": f"Failed to download LoRA: {download_result.get('message', 'Unknown error')}",
                        }
                        return

                    lora_path = download_result["local_path"]
                    logger.debug(f"LoRA downloaded to: {lora_path}")


                    lora_id = lora_name_to_id(lora_name)

                    if is_hot_swap and old_info is not None and old_engine_loaded:
                        try:
                            await self.engine_client.remove_lora(old_info.id)
                            self._engine_loaded_loras.discard(lora_name)
                        except Exception as e:
                            logger.error(
                                f"Failed to remove existing LoRA '{lora_name}' "
                                f"before hot-swap: {e}"
                            )
                            yield {
                                "status": "error",
                                "message": (
                                    f"Failed to remove existing LoRA '{lora_name}' "
                                    f"before hot-swap: {e}"
                                ),
                                "lora_name": lora_name,
                            }
                            return


                    preload_into_engine = (
                        self._preload_lora_into_engine() or is_hot_swap
                    )
                    if preload_into_engine:
                        try:
                            await self.engine_client.add_lora(
                                LoRARequest(
                                    lora_name=lora_name,
                                    lora_int_id=lora_id,
                                    lora_path=lora_path,
                                )
                            )
                            self._engine_loaded_loras.add(lora_name)
                        except Exception as e:
                            if (
                                is_hot_swap
                                and old_info is not None
                                and old_engine_loaded
                            ):
                                try:
                                    await self.engine_client.add_lora(
                                        LoRARequest(
                                            lora_name=lora_name,
                                            lora_int_id=old_info.id,
                                            lora_path=old_info.path,
                                        )
                                    )
                                    self._engine_loaded_loras.add(lora_name)
                                except Exception as rollback_error:
                                    self.loaded_loras.pop(lora_name, None)
                                    logger.exception(
                                        f"Rollback failed for LoRA {lora_name}: "
                                        f"{rollback_error}"
                                    )
                            yield {
                                "status": "error",
                                "message": f"Failed to add LoRA '{lora_name}': {e}",
                                "lora_name": lora_name,
                            }
                            return


                    self.loaded_loras[lora_name] = LoRAInfo(id=lora_id, path=lora_path)
                    logger.info(
                        f"Successfully {'hot-swapped' if is_hot_swap else 'loaded'} "
                        f"LoRA adapter: {lora_name} with ID {lora_id}"
                    )

                    if is_hot_swap:
                        try:
                            await self.engine_client.reset_prefix_cache()
                        except Exception as e:


                            rolled_back = "tracking only"
                            if old_info is not None:
                                try:
                                    if preload_into_engine:
                                        await self.engine_client.remove_lora(lora_id)
                                        self._engine_loaded_loras.discard(lora_name)
                                    if old_engine_loaded:
                                        await self.engine_client.add_lora(
                                            LoRARequest(
                                                lora_name=lora_name,
                                                lora_int_id=old_info.id,
                                                lora_path=old_info.path,
                                            )
                                        )
                                        self._engine_loaded_loras.add(lora_name)
                                    self.loaded_loras[lora_name] = old_info
                                    rolled_back = (
                                        "engine+tracking"
                                        if old_engine_loaded
                                        else "tracking only"
                                    )
                                except Exception as rollback_error:


                                    self.loaded_loras.pop(lora_name, None)
                                    logger.exception(
                                        f"LoRA '{lora_name}' hot-swap engine "
                                        f"rollback failed: {rollback_error}"
                                    )
                            else:
                                self.loaded_loras.pop(lora_name, None)
                            logger.error(
                                f"LoRA '{lora_name}' hot-swap rolled back "
                                f"({rolled_back}): prefix cache reset failed: {e}"
                            )
                            yield {
                                "status": "error",
                                "message": (
                                    f"LoRA '{lora_name}' hot-swap aborted; prefix "
                                    f"cache reset failed: {e}"
                                ),
                                "lora_name": lora_name,
                                "lora_id": lora_id,
                            }
                            return


                    if not is_hot_swap and self.generate_endpoint is not None:
                        logger.debug(
                            f"Publishing LoRA '{lora_name}' ModelDeploymentCard to {self.generate_endpoint}"
                        )
                        try:
                            logger.debug(
                                f"Publishing LoRA '{lora_name}' ModelDeploymentCard"
                            )


                            user_data = {
                                "lora_adapter": True,
                                "lora_id": lora_id,
                            }

                            runtime_config = ModelRuntimeConfig()
                            runtime_config.context_length = self.model_max_len
                            runtime_config.tool_call_parser = (
                                self.config.dyn_tool_call_parser
                            )
                            runtime_config.reasoning_parser = (
                                self.config.dyn_reasoning_parser
                            )


                            if (
                                self.config.disaggregation_mode
                                == DisaggregationMode.PREFILL
                            ):
                                lora_model_type = ModelType.Prefill
                                lora_worker_type = WorkerType.Prefill
                                lora_needs_set: list[WorkerType] = [WorkerType.Decode]
                            elif (
                                self.config.disaggregation_mode
                                == DisaggregationMode.DECODE
                            ):
                                lora_model_type = ModelType.Chat | ModelType.Completions
                                lora_worker_type = WorkerType.Decode
                                lora_needs_set = [WorkerType.Prefill]
                            else:
                                lora_model_type = ModelType.Chat | ModelType.Completions
                                lora_worker_type = WorkerType.Aggregated
                                lora_needs_set = []
                            if self.config.route_to_encoder:
                                lora_needs_set.append(WorkerType.Encode)
                            lora_needs: list[list[WorkerType]] = (
                                [lora_needs_set] if lora_needs_set else []
                            )


                            await register_model(
                                model_input=ModelInput.Tokens,
                                model_type=lora_model_type,
                                endpoint=self.generate_endpoint,
                                model_path=self.config.model,
                                kv_cache_block_size=self.config.engine_args.block_size,
                                runtime_config=runtime_config,
                                user_data=user_data,
                                lora_name=lora_name,
                                base_model_path=self.config.model,
                                worker_type=lora_worker_type,
                                needs=lora_needs,


                                max_gpu_lora_count=getattr(
                                    self.config.engine_args, "max_loras", None
                                ),
                            )
                            logger.info(
                                f"Successfully published LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to publish LoRA {lora_name} ModelDeploymentCard: {e}"
                            )


                            try:
                                if preload_into_engine:
                                    logger.debug(
                                        f"Rolling back: removing LoRA '{lora_name}' from engine"
                                    )
                                    await self.engine_client.remove_lora(lora_id)
                                    self._engine_loaded_loras.discard(lora_name)
                                self.loaded_loras.pop(lora_name, None)
                                logger.debug(
                                    f"Successfully rolled back LoRA '{lora_name}'"
                                )
                            except Exception as rollback_error:
                                logger.exception(
                                    f"Failed to rollback LoRA {lora_name}: {rollback_error}"
                                )


                            yield {
                                "status": "error",
                                "message": f"Failed to register LoRA '{lora_name}' in discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return
                    elif not is_hot_swap:
                        logger.debug(
                            f"Cannot publish LoRA '{lora_name}': generate_endpoint={self.generate_endpoint}, config={self.config}"
                        )

                    yield {
                        "status": "success",
                        "message": (
                            f"LoRA adapter '{lora_name}' "
                            f"{'hot-swapped' if is_hot_swap else 'loaded'} successfully"
                        ),
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                        "hot_swap": is_hot_swap,
                    }
                finally:


                    pass
        except Exception as e:
            logger.exception(f"Failed to load LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def unload_lora(self, request=None):


        try:
            try:
                lora_name = require_lora_unload_request(request)
            except RLAdminValidationError as e:
                yield {"status": "error", "message": str(e)}
                return


            lock = self._get_lora_lock(lora_name)
            async with lock:
                try:

                    lora = self.loaded_loras.get(lora_name)
                    if lora is None:
                        yield {
                            "status": "error",
                            "message": f"LoRA adapter '{lora_name}' not found. Available LoRAs: {list(self.loaded_loras.keys())}",
                        }
                        return

                    logger.debug(f"Unloading LoRA adapter: {lora_name}")
                    lora_id = lora.id


                    if self.generate_endpoint is not None:
                        logger.debug(
                            f"Unregistering LoRA '{lora_name}' ModelDeploymentCard"
                        )
                        try:
                            await unregister_model(
                                endpoint=self.generate_endpoint,
                                lora_name=lora_name,
                            )
                            logger.info(
                                f"Successfully unregistered LoRA '{lora_name}' ModelDeploymentCard"
                            )
                        except Exception as e:
                            logger.exception(
                                f"Failed to unregister LoRA {lora_name} ModelDeploymentCard: {e}"
                            )
                            yield {
                                "status": "error",
                                "message": f"Failed to unregister LoRA '{lora_name}' from discovery registry: {str(e)}",
                                "lora_name": lora_name,
                            }
                            return
                    else:
                        logger.debug(
                            f"Cannot unregister LoRA '{lora_name}': generate_endpoint={self.generate_endpoint}"
                        )


                    if lora_name in self._engine_loaded_loras:
                        try:
                            await self.engine_client.remove_lora(lora_id)
                        except Exception as e:
                            if not self._is_lora_not_loaded_error(e):
                                raise
                        self._engine_loaded_loras.discard(lora_name)
                    del self.loaded_loras[lora_name]

                    logger.info(
                        f"Successfully unloaded LoRA adapter: {lora_name} with ID {lora_id}"
                    )
                    yield {
                        "status": "success",
                        "message": f"LoRA adapter '{lora_name}' unloaded successfully",
                        "lora_name": lora_name,
                        "lora_id": lora_id,
                    }
                finally:


                    pass
        except Exception as e:
            logger.exception(f"Failed to unload LoRA adapter: {e}")
            yield {"status": "error", "message": str(e)}

    async def list_loras(self, request=None):


        try:
            loras = {name: lora.id for name, lora in self.loaded_loras.items()}
            yield {
                "status": "success",
                "loras": loras,
                "count": len(loras),
            }
        except Exception as e:
            logger.error(f"Failed to list LoRA adapters: {e}")
            yield {"status": "error", "message": str(e)}

    def cleanup(self):

        for temp_dir in self.temp_dirs:
            try:
                temp_dir.cleanup()
            except Exception as e:
                logger.warning(f"Failed to clean up temp directory: {e}")

    def _decode_prompt_embeds(self, prompt_embeds_base64: str):


        if not isinstance(prompt_embeds_base64, str):
            raise ValueError(
                f"Prompt embeds must be base64 encoded string. Got {type(prompt_embeds_base64)}."
            )

        if self.model_config is None:
            raise ValueError("ModelConfig is unavailable for prompt_embeds validation.")

        try:
            return safe_load_prompt_embeds(
                self.model_config, prompt_embeds_base64.encode()
            )
        except Exception as e:
            logger.error(f"Failed to decode prompt_embeds: {e}")
            raise ValueError(f"Failed to decode prompt_embeds as PyTorch tensor: {e}")

    def _create_prompt_from_embeddings(
        self, prompt_embeds_base64: str
    ) -> tuple[EmbedsPrompt, int, torch.Tensor]:


        embeddings_tensor = self._decode_prompt_embeds(prompt_embeds_base64)
        if embeddings_tensor.dim() != 2:
            raise ValueError(
                f"prompt embeds should have dim 2 after vllm processing, but found dim {embeddings_tensor.dim()}"
            )


        sequence_length = embeddings_tensor.shape[0]


        prompt = EmbedsPrompt(prompt_embeds=embeddings_tensor)

        return prompt, sequence_length, embeddings_tensor

    async def _try_receive_mm_kwargs(
        self, request: Dict[str, Any]
    ) -> Dict[str, Any] | None:


        extra_args = request.get("extra_args") or {}
        logger.debug(
            "[mm-routing] _try_receive_mm_kwargs: extra_args keys=%s",
            list(extra_args.keys()),
        )


        shm_meta_raw = extra_args.get("mm_kwargs_shm")
        if shm_meta_raw:
            shm_meta = MmKwargsShmTransferMetadata.model_validate(shm_meta_raw)
            return await self._receive_mm_kwargs(
                extra_args, "shm", MmKwargsShmReceiver(), shm_meta
            )

        nixl_meta_raw = extra_args.get("mm_kwargs_nixl")
        if nixl_meta_raw:
            nixl_meta = MmKwargsTransferMetadata.model_validate(nixl_meta_raw)
            if self._mm_kwargs_receiver is None:
                self._mm_kwargs_receiver = MmKwargsNixlReceiver()
            return await self._receive_mm_kwargs(
                extra_args, "nixl", self._mm_kwargs_receiver, nixl_meta
            )

        logger.debug("[mm-routing] No mm_kwargs transfer metadata in extra_args")
        return None

    async def _receive_mm_kwargs(
        self,
        extra_args: Dict[str, Any],
        transport: str,
        receiver: MmKwargsReceiver,
        metadata: Any,
    ) -> Dict[str, Any] | None:


        color = "magenta" if transport == "nixl" else "cyan"
        rng = _nvtx.start_range(f"mm_backend:{transport}_receive", color=color)
        try:
            mm_hashes = extra_args.get("mm_hashes")
            mm_placeholders = extra_args.get("mm_placeholders")
            if not mm_hashes or not mm_placeholders:
                logger.warning(
                    "[mm-routing] %s present but mm_hashes/mm_placeholders missing",
                    transport,
                )
                return None
            mm_hashes = _pad_mm_hashes_to_64(mm_hashes)


            results = await receiver.receive(metadata)

            pickled_items = results.get("__pickled_kwargs_item__")
            if not pickled_items:
                logger.warning(
                    "[mm-routing] %s: no pickled kwargs items received", transport
                )
                return None


            kwargs_items: list[MultiModalKwargsItem] = []
            with _nvtx.annotate(f"mm_backend:{transport}_pickle_loads", color=color):
                for pi in pickled_items:
                    item = pickle.loads(pi)
                    if not isinstance(item, MultiModalKwargsItem):
                        logger.warning(
                            "[mm-routing] %s: deserialized object is %s, expected "
                            "MultiModalKwargsItem; falling back to normal path",
                            transport,
                            type(item).__name__,
                        )
                        return None
                    kwargs_items.append(item)


            expanded_token_ids = extra_args.get("expanded_token_ids")
            if not expanded_token_ids:
                logger.warning(
                    "[mm-routing] %s: no expanded_token_ids in extra_args, "
                    "cannot use pre-rendered mm_kwargs; falling back",
                    transport,
                )
                return None

            mm_hashes_dict = {metadata.modality: mm_hashes}
            mm_kwargs_dict = {metadata.modality: kwargs_items}
            with _nvtx.annotate(
                f"mm_backend:{transport}_build_engine_input", color=color
            ):
                engine_input = {
                    "type": "multimodal",
                    "prompt_token_ids": expanded_token_ids,
                    "mm_kwargs": mm_kwargs_dict,
                    "mm_hashes": mm_hashes_dict,
                    "mm_placeholders": {
                        metadata.modality: [
                            PlaceholderRange(offset=off, length=length)
                            for off, length in mm_placeholders
                        ],
                    },
                }


            try:
                self.engine_client.input_processor.inject_into_mm_cache(
                    mm_hashes_dict, mm_kwargs_dict
                )
            except Exception:
                logger.debug(
                    "[mm-routing] Failed to inject into mm_cache", exc_info=True
                )

            logger.debug(
                "[mm-routing] %s: constructed pre-rendered MultiModalInput from "
                "%d kwargs_items, %d hashes, %d placeholders",
                transport,
                len(kwargs_items),
                len(mm_hashes),
                len(mm_placeholders),
            )
            return engine_input
        except Exception:
            logger.exception("[mm-routing] %s receive failed, falling back", transport)
            return None
        finally:
            _nvtx.end_range(rng)

    @staticmethod
    def _get_mm_processor_kwargs(
        request: Dict[str, Any],
    ) -> Dict[str, Any] | None:


        mm_processor_kwargs = request.get("mm_processor_kwargs")
        if mm_processor_kwargs is None:
            req_extra_args = request.get("extra_args")
            if isinstance(req_extra_args, dict):
                mm_processor_kwargs = req_extra_args.get("mm_processor_kwargs")
        return mm_processor_kwargs

    async def _extract_multimodal_data(
        self,
        request: Dict[str, Any],
        request_id: str,
        context,
        mm_processor_kwargs: Dict[str, Any] | None = None,
    ) -> Dict[str, Any] | None:


        rng = _nvtx.start_range("mm_backend:extract_multimodal_data", color="orange")
        if "multi_modal_data" not in request or request["multi_modal_data"] is None:
            _nvtx.end_range(rng)
            return None


        if not self.enable_multimodal:
            raise ValueError(
                "Received multimodal data but multimodal processing is not enabled. "
                "Use --enable-multimodal flag to enable multimodal processing."
            )

        mm_map = request["multi_modal_data"]

        vllm_mm_data = {}


        if self.embedding_loader is not None:


            image_urls = []
            supported = True
            for item in mm_map.get(IMAGE_URL_KEY, []):
                if isinstance(item, dict) and "Url" in item:
                    image_urls.append(item["Url"])
                elif isinstance(item, dict) and "Decoded" in item:
                    supported = False
            if supported:
                vllm_mm_data = await self.embedding_loader.load_multimodal_embeddings(
                    image_urls, request_id, model=self.config.model, context=context
                )
                logger.debug(
                    f"Fetched multimodal embeddings for {len(vllm_mm_data)} items"
                )

        image_mm_items = mm_map.get(IMAGE_URL_KEY, [])


        image_modality_key = (
            "vision_chunk" if self._use_unified_vision_chunk else "image"
        )
        if image_modality_key not in vllm_mm_data and image_mm_items:
            with _nvtx.annotate("mm_backend:image_download", color="green"):
                images = await self.image_loader.load_image_batch(
                    image_mm_items,
                )

            if images:
                if self._use_unified_vision_chunk:


                    chunks = [
                        {"type": "image", "image": img, "uuid": None} for img in images
                    ]
                    vllm_mm_data["vision_chunk"] = (
                        chunks[0] if len(chunks) == 1 else chunks
                    )
                else:

                    vllm_mm_data["image"] = images[0] if len(images) == 1 else images
                logger.debug(
                    f"Extracted {len(images)} image(s) for multimodal "
                    f"processing under modality={image_modality_key!r}"
                )

        video_mm_items = mm_map.get(VIDEO_URL_KEY, [])
        if video_mm_items:
            videos = await self.video_loader.load_video_batch(video_mm_items)

            if videos:

                vllm_mm_data["video"] = videos[0] if len(videos) == 1 else videos
                logger.debug(
                    f"Extracted {len(videos)} video(s) for multimodal processing"
                )


        audio_mm_items = mm_map.get(AUDIO_URL_KEY, [])
        if audio_mm_items:
            audios = await self.audio_loader.load_audio_batch(audio_mm_items)
            if audios:
                vllm_mm_data["audio"] = audios[0] if len(audios) == 1 else audios
                logger.debug(
                    f"Extracted {len(audios)} audio item(s) for multimodal processing"
                )


        if (
            video_mm_items
            and mm_processor_kwargs
            and mm_processor_kwargs.get("use_audio_in_video", False)
        ):
            video_audios: list = []
            for item in video_mm_items:
                url = item.get(URL_VARIANT_KEY) if isinstance(item, dict) else None
                if not url:
                    raise ValueError(
                        "use_audio_in_video requires all video items to be "
                        "URL-based. Got a non-URL video item (e.g. frontend-"
                        "decoded). Audio extraction from decoded video data "
                        "is not yet supported."
                    )
                try:
                    audio = await self.audio_loader.load_audio(url)
                    video_audios.append(audio)
                except Exception:
                    logger.error(
                        "Failed to extract audio from video %s. "
                        "use_audio_in_video requires every video to "
                        "contain an audio stream.",
                        url[:80],
                    )
                    raise
            if video_audios:
                existing = vllm_mm_data.get("audio")
                if existing is not None:
                    all_audios = (
                        existing if isinstance(existing, list) else [existing]
                    ) + video_audios
                else:
                    all_audios = video_audios
                vllm_mm_data["audio"] = (
                    all_audios[0] if len(all_audios) == 1 else all_audios
                )
                logger.debug(
                    "Extracted %d audio track(s) from video URL(s) "
                    "(use_audio_in_video=True)",
                    len(video_audios),
                )

        _nvtx.end_range(rng)
        return vllm_mm_data if vllm_mm_data else None

    def _build_prompt_from_request(
        self,
        request: Dict[str, Any],
        request_id: str,
        multi_modal_data: Dict[str, Any] | None,
        log_prefix: str = "",
        mm_processor_kwargs: Dict[str, Any] | None = None,
    ) -> tuple[TokensPrompt | EmbedsPrompt | None, int | None, Dict[str, Any] | None]:


        embedding_sequence_length = None

        if "prompt_embeds" in request and request["prompt_embeds"]:
            if not self.config.engine_args.enable_prompt_embeds:
                msg = (
                    "Set `--enable-prompt-embeds` to allow `prompt_embeds` in request."
                )
                logger.error(
                    f"Rejected prompt_embeds for {log_prefix.lower().strip() or 'request'} "
                    f"{request_id}: {msg}"
                )
                return (
                    None,
                    None,
                    {
                        "finish_reason": f"error: Invalid prompt_embeds: {msg}",
                        "token_ids": [],
                    },
                )
            try:
                (
                    prompt,
                    embedding_sequence_length,
                    tensor,
                ) = self._create_prompt_from_embeddings(request["prompt_embeds"])
                logger.info(
                    f"{log_prefix}Using prompt embeddings: shape={tensor.shape}, "
                    f"dtype={tensor.dtype}, sequence_length={embedding_sequence_length}, "
                    f"request_id={request_id}"
                )
                return prompt, embedding_sequence_length, None
            except Exception as e:
                logger.error(
                    f"Failed to process prompt_embeds for {log_prefix.lower().strip() or 'request'} "
                    f"{request_id}: {e}"
                )
                return (
                    None,
                    None,
                    {
                        "finish_reason": f"error: Invalid prompt_embeds: {e}",
                        "token_ids": [],
                    },
                )


        extra_args = request.get("extra_args") or {}
        forwarded_hashes = extra_args.get("mm_hashes")
        mm_uuids: dict[str, Any] | None = None
        if forwarded_hashes:
            forwarded_hashes = _pad_mm_hashes_to_64(forwarded_hashes)


            mm_modality_key = (
                "vision_chunk" if self._use_unified_vision_chunk else "image"
            )
            mm_uuids = {mm_modality_key: forwarded_hashes}
        elif self.embedding_loader is None:
            mm_uuids = _compute_mm_uuids(multi_modal_data)
            if mm_uuids and multi_modal_data:
                logger.warning(
                    "[mm-routing] No forwarded mm_hashes from frontend; "
                    "recomputed from image data. KV-cache-aware MM routing "
                    "may not match the frontend's routing decisions."
                )
        prompt_kwargs = dict[str, Any](
            prompt_token_ids=request["token_ids"],
            multi_modal_data=multi_modal_data,
        )
        if mm_uuids is not None:
            prompt_kwargs["multi_modal_uuids"] = mm_uuids
        if mm_processor_kwargs is not None:
            prompt_kwargs["mm_processor_kwargs"] = mm_processor_kwargs

        prompt = TokensPrompt(**prompt_kwargs)
        return prompt, embedding_sequence_length, None

    @staticmethod
    def _build_completion_usage(
        request_output: RequestOutput,
        embedding_sequence_length: int | None = None,
        completion_token_counts: dict[int, int] | None = None,
    ) -> Dict[str, Any]:


        if embedding_sequence_length is not None:
            prompt_tokens = embedding_sequence_length
        elif request_output.prompt_token_ids:
            prompt_tokens = len(request_output.prompt_token_ids)
        else:
            prompt_tokens = None

        if completion_token_counts is not None:
            completion_tokens = sum(completion_token_counts.values())
        else:
            completion_tokens = sum(
                len(output.token_ids) for output in request_output.outputs
            )

        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": (
                prompt_tokens + completion_tokens if prompt_tokens is not None else None
            ),
            "prompt_tokens_details": (
                {"cached_tokens": num_cached}
                if (num_cached := getattr(request_output, "num_cached_tokens", None))
                else None
            ),
        }

    @staticmethod
    def _extract_logprobs(
        output, num_output_tokens_so_far: int, tokenizer=None
    ) -> tuple[list[float] | None, list[list[dict]] | None]:

        return _shared_logprobs.extract_from_completion_output(
            output,
            num_output_tokens_so_far,
            tokenizer=tokenizer,
            fallback_to_first_on_missing=True,
            include_bytes=True,
        )

    @staticmethod
    def _log_with_lora_context(
        message: str,
        request_id: str,
        lora_request=None,
        level: str = "debug",
        **kwargs,
    ) -> None:


        if lora_request:
            lora_info = f" with LoRA {lora_request.lora_name}"
        else:
            lora_info = ""

        formatted_message = message.format(
            request_id=request_id,
            lora_info=lora_info,
            **kwargs,
        )

        if level == "info":
            logger.info(formatted_message)
        else:
            logger.debug(formatted_message)

    async def generate_tokens(
        self,
        prompt,
        sampling_params,
        request_id,
        data_parallel_rank=None,
        lora_request=None,
        embedding_sequence_length=None,
        trace_headers=None,
        priority=0,
        reasoning_ended=None,
        reasoning_parser_kwargs=None,
    ):
        try:

            self._log_with_lora_context(
                "Starting token generation for request {request_id}{lora_info}",
                request_id,
                lora_request,
            )
            gen = self._generate_with_lora_admission_lock(
                lora_request,
                lambda admitted_lora_request: self.engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    lora_request=admitted_lora_request,
                    data_parallel_rank=data_parallel_rank,
                    trace_headers=trace_headers,
                    priority=priority,
                    **_engine_generate_reasoning_kwargs(
                        self.engine_client,
                        reasoning_ended,
                        reasoning_parser_kwargs,
                    ),
                ),
            )

            total_output_tokens_by_index: dict[int, int] = {}
            raw_routed_experts_by_output: dict[int, Any] = {}


            prompt_logprobs_payload: Optional[list] = None
            async for res in gen:
                if _ttft_diag is not None:
                    _ttft_diag.request_queue_event("worker_request_dequeued", request_id, res)

                if (
                    prompt_logprobs_payload is None
                    and getattr(res, "prompt_logprobs", None) is not None
                ):
                    prompt_logprobs_payload = _serialize_prompt_logprobs(
                        res.prompt_logprobs
                    )

                if not res.outputs:
                    self._log_with_lora_context(
                        "Request {request_id}{lora_info} returned no outputs",
                        request_id,
                        lora_request,
                    )


                    yield {
                        "finish_reason": "error: No outputs from vLLM engine",
                        "index": 0,
                        "token_ids": [],
                    }
                    break

                prepared_outputs = []
                for output in res.outputs:
                    output_idx = getattr(output, "index", 0) or 0
                    token_ids = list(output.token_ids or [])
                    total_output_tokens_by_index[
                        output_idx
                    ] = total_output_tokens_by_index.get(output_idx, 0) + len(token_ids)
                    finish_reason = getattr(output, "finish_reason", None)
                    stop_reason = getattr(output, "stop_reason", None)
                    if not token_ids and not finish_reason and not stop_reason:
                        continue
                    prepared_outputs.append(
                        (output, output_idx, token_ids, finish_reason, stop_reason)
                    )

                for (
                    output,
                    output_idx,
                    token_ids,
                    finish_reason,
                    stop_reason,
                ) in prepared_outputs:
                    out = {
                        "index": output_idx,
                        "token_ids": token_ids,
                    }


                    raw_routed_experts = getattr(output, "routed_experts", None)
                    if raw_routed_experts is not None:
                        raw_routed_experts_by_output[output_idx] = raw_routed_experts


                    tokenizer = getattr(self.engine_client, "tokenizer", None)
                    log_probs, top_logprobs = self._extract_logprobs(
                        output, 0, tokenizer=tokenizer
                    )
                    if log_probs is not None:
                        out["log_probs"] = log_probs
                    if top_logprobs is not None:
                        out["top_logprobs"] = top_logprobs

                    if finish_reason:
                        out["finish_reason"] = normalize_finish_reason(finish_reason)
                        out[
                            "completion_usage"
                        ] = BaseWorkerHandler._build_completion_usage(
                            request_output=res,
                            embedding_sequence_length=embedding_sequence_length,
                            completion_token_counts=total_output_tokens_by_index,
                        )
                        if prompt_logprobs_payload is not None:
                            _attach_prompt_logprobs_engine_data(
                                out, prompt_logprobs_payload
                            )


                        raw_start = int(
                            getattr(sampling_params, "routed_experts_prompt_start", 0)
                            or 0
                        )
                        prompt_len = len(getattr(res, "prompt_token_ids", None) or [])
                        effective_start = min(raw_start, prompt_len)
                        routed_experts = _serialize_routed_experts(
                            raw_routed_experts_by_output.get(output_idx),
                            start=effective_start,
                        )
                        if routed_experts is not None:
                            _attach_routed_experts_engine_data(out, routed_experts)

                        self._log_with_lora_context(
                            "Completed token generation for request {request_id}{lora_info}: "
                            "{output_tokens} output tokens, finish_reason={finish_reason}",
                            request_id,
                            lora_request,
                            output_tokens=total_output_tokens_by_index.get(
                                output_idx, 0
                            ),
                            finish_reason=finish_reason,
                        )
                    if stop_reason:
                        out["stop_reason"] = stop_reason
                    yield out

        except EngineDeadError as e:
            logger.error(f"vLLM EngineDeadError: {e}")
            logger.warning("Initiating Dynamo Runtime shutdown.")
            self.runtime.shutdown()
            os._exit(1)


class DecodeWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        model_config: ModelConfig | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Client | None = None,
    ):
        super().__init__(
            runtime,
            config,
            engine,
            default_sampling_params,
            model_max_len=model_max_len,
            model_config=model_config,
            enable_multimodal=enable_multimodal,
            generate_endpoint=generate_endpoint,
            use_vllm_tokenizer=use_vllm_tokenizer,
            shutdown_event=shutdown_event,
            enable_frontend_decoding=enable_frontend_decoding,
            encode_worker_client=encode_worker_client,
        )

    async def generate(self, request, context):

        request_id = context.id()
        chord_metadata = request_metadata(request) if self._chord_enabled else None
        if chord_metadata is not None:
            request_id = str(chord_metadata[BACKEND_REQUEST_ID_KEY])
        elif self._llumnix_enabled:
            running_transfer = migration_envelope(request)
            if running_transfer is not None:
                request_id = running_transfer.request_id
        if _ttft_diag is not None:
            _ttft_diag.worker_received(request_id)
        logger.debug(f"Decode Request ID: {request_id}")
        first_token = True
        with time_and_log_code_section(
            f"[DECODE] request: {request_id} generate"
        ) as decode_timer:
            if self.use_vllm_tokenizer:

                generator = self._generate_text_mode(request, context, request_id)
            else:

                generator = self._generate_token_mode(request, context, request_id)

            stream = generator
            if self._llumnix_enabled:
                stream = self._llumnix_stream(request, context, request_id, generator)
            if self._chord_enabled:
                stream = self._chord_stream(request, context, request_id, stream)
            async for chunk in stream:
                if first_token:
                    decode_timer.stop_interval()
                    first_token = False
                yield chunk

    async def _chord_stream(self, request, context, request_id: str, source_stream):
        del request_id
        async for chunk in chord_worker_stream(self, request, context, source_stream):
            yield chunk

    async def _llumnix_stream(self, request, context, request_id: str, source_stream):
        registry = getattr(self, "_llumnix_handoffs", None)
        if not self._llumnix_enabled or registry is None:
            async for chunk in source_stream:
                yield chunk
            return

        entry = registry.register(request_id, dict(request), context)
        source_output_tokens = 0
        source_finished = False
        migration_abort = False
        try:
            async for chunk in source_stream:
                finish_reason = str(chunk.get("finish_reason") or "").lower()
                if (
                    registry.active(entry)
                    and finish_reason in {"abort", "cancelled"}
                    and not context.is_stopped()
                ):
                    migration_abort = True


                    token_ids = chunk.get("token_ids")
                    if isinstance(token_ids, list) and token_ids:
                        source_output_tokens += len(token_ids)
                        pending = dict(chunk)
                        for key in ("finish_reason", "stop_reason", "completion_usage"):
                            pending.pop(key, None)
                        yield pending
                    continue
                token_ids = chunk.get("token_ids")
                if isinstance(token_ids, list):
                    source_output_tokens += len(token_ids)
                source_finished = bool(finish_reason)
                yield chunk

            if registry.active(entry) and (migration_abort or not source_finished):
                try:
                    async for chunk in registry.drain(entry, source_output_tokens):
                        yield chunk
                except Exception as error:
                    logger.exception(
                        "Llumnix migration stream failed: request_id=%s",
                        request_id,
                    )
                    yield {
                        "finish_reason": {
                            "error": f"Llumnix migration failed: {error}"
                        },
                        "index": 0,
                        "token_ids": [],
                    }
        finally:
            await registry.close(entry)

    async def _generate_token_mode(self, request, context, request_id):


        prefill_result = request.get("prefill_result")
        if prefill_result and isinstance(prefill_result, dict):


            disaggregated_params = prefill_result.get("disaggregated_params") or {}
            kv_params = disaggregated_params.get("kv_transfer_params")
            embedding_params = disaggregated_params.get("embedding_params")

            if not embedding_params:
                embedding_params = None
        else:
            kv_params = None
            embedding_params = None

        is_decode_only = self.config.disaggregation_mode == DisaggregationMode.DECODE
        has_mm_data = (
            "multi_modal_data" in request and request["multi_modal_data"] is not None
        )

        mm_processor_kwargs = self._get_mm_processor_kwargs(request)

        multi_modal_data: Dict[str, Any] | None = None
        pre_rendered: Dict[str, Any] | None = None
        if is_decode_only:

            if resolve_model_family(self.config.model) is ModelFamily.QWEN_VL:

                if embedding_params is not None:
                    multi_modal_data = construct_qwen_decode_mm_data(
                        embedding_params["image_grid_thw"],
                        embedding_params["embeddings_shape"],
                        request_id,
                    )
                elif has_mm_data and request["multi_modal_data"].get(IMAGE_URL_KEY):


                    msg = (
                        "Decode worker received multimodal request without "
                        "prefill result"
                        if prefill_result is None
                        else "Prefill did not produce required multimodal "
                        "embedding metadata (image_grid_thw) for Qwen VL "
                        "decode. Use --route-to-encoder or the P/D launcher "
                        "with grid_thw computation support"
                    )
                    logger.error("Request %s: %s", request_id, msg)
                    yield {"status": "error", "message": msg}
                    return
            else:


                if embedding_params and "expanded_prompt_token_ids" in embedding_params:
                    request["token_ids"] = embedding_params["expanded_prompt_token_ids"]
                    has_mm_data = False


            if multi_modal_data is None and has_mm_data:
                mm = request["multi_modal_data"]
                if mm.get(VIDEO_URL_KEY) or mm.get(AUDIO_URL_KEY):
                    multi_modal_data = await self._extract_multimodal_data(
                        request,
                        request_id,
                        context,
                        mm_processor_kwargs=mm_processor_kwargs,
                    )
        else:


            with _nvtx.annotate("mm_backend:receive_mm_kwargs", color="magenta"):
                pre_rendered = await self._try_receive_mm_kwargs(request)
            if pre_rendered is not None:
                logger.debug(
                    "[mm-routing] Request %s: received pre-rendered mm_kwargs via NIXL/SHM",
                    request_id,
                )
            else:

                multi_modal_data = await self._extract_multimodal_data(
                    request,
                    request_id,
                    context,
                    mm_processor_kwargs=mm_processor_kwargs,
                )


        prompt: Any
        with _nvtx.annotate("mm_backend:build_prompt", color="yellow"):
            if pre_rendered is not None:


                prompt = pre_rendered
                embedding_sequence_length = None
                error = None
                logger.debug(
                    "[mm-routing] Request %s: using pre-rendered MultiModalInput",
                    request_id,
                )
            else:
                (
                    prompt,
                    embedding_sequence_length,
                    error,
                ) = self._build_prompt_from_request(
                    request,
                    request_id,
                    multi_modal_data,
                    mm_processor_kwargs=mm_processor_kwargs,
                )
        if error is not None:
            yield error
            return

        _apply_nvext_cache_salt(request, prompt)


        sampling_params = build_sampling_params(
            request,
            self.default_sampling_params,
            self.model_max_len,
            enable_rl=self.config.enable_rl,
        )

        if kv_params is not None:
            if sampling_params.extra_args is None:
                sampling_params.extra_args = {}
            sampling_params.extra_args["kv_transfer_params"] = kv_params
            logger.debug(
                f"Using disaggregated params from prefill for request {request_id}"
            )
        if self._llumnix_enabled:
            apply_resume_metadata(request, sampling_params)
        prefill_prompt_tokens_details = (
            prefill_result.get("prompt_tokens_details") if prefill_result else None
        )


        model_name = request.get("model")
        lora_request = self._resolve_lora_request(model_name)
        if lora_request:
            logger.info(
                f"Decode request {request_id} will use LoRA adapter: {model_name} (ID: {lora_request.lora_int_id})"
            )
        else:
            logger.debug(
                f"Decode request {request_id} has no LoRA specified (model: {model_name})"
            )
        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))

        trace_headers = context.trace_headers()
        reasoning_ended, reasoning_parser_kwargs = _request_reasoning_metadata(request)


        async with _deferred_abort_guard(
            self.engine_client,
            request_id,
            is_decode_only,
            self._deferred_aborts,
            self._shutdown_on_engine_dead,
        ) as abort_guard:
            async with self._abort_monitor(
                context, request_id, abort_guard=abort_guard
            ):


                want_engine_data = _nvext_extra_field_requested(request, "engine_data")


                request_prompt_token_ids = (
                    _prompt_token_ids_for_engine_data(request, prompt)
                    if want_engine_data
                    else None
                )
                accumulated_token_ids: dict[int, list[int]] = {}
                accumulated_log_probs: dict[int, list[float]] = {}
                engine_prompt = prompt
                chord_request = await build_chord_engine_request(
                    self.engine_client,
                    dict(request),
                    prompt,
                    sampling_params,
                    request_id,
                    lora_request=lora_request,
                    trace_headers=trace_headers,
                    priority=priority,
                    data_parallel_rank=dp_rank,
                )
                if chord_request is not None:
                    engine_prompt = chord_request
                try:
                    async for tok in self.generate_tokens(
                        engine_prompt,
                        sampling_params,
                        request_id,
                        data_parallel_rank=dp_rank,
                        lora_request=lora_request,
                        embedding_sequence_length=embedding_sequence_length,
                        trace_headers=trace_headers,
                        priority=priority,
                        reasoning_ended=reasoning_ended,
                        reasoning_parser_kwargs=reasoning_parser_kwargs,
                    ):
                        if abort_guard is not None:
                            abort_guard.signal_first_token()
                        if prefill_result is not None and "completion_usage" in tok:
                            tok["completion_usage"][
                                "prompt_tokens_details"
                            ] = prefill_prompt_tokens_details

                        if want_engine_data:
                            _accumulate_engine_data(
                                tok,
                                request_prompt_token_ids,
                                accumulated_token_ids,
                                accumulated_log_probs,
                            )
                        if _ttft_diag is not None:
                            _ttft_diag.worker_stream(request_id, len(tok.get("token_ids") or []))
                        yield tok
                except EngineDeadError as e:
                    logger.error(f"vLLM EngineDeadError: {e}")
                    logger.warning("Initiating Dynamo Runtime shutdown.")
                    self.runtime.shutdown()
                    os._exit(1)

    async def _generate_text_mode(self, request, context, request_id):

        if self._chord_enabled:
            raise ValueError(
                "Chord experiments require the token-input Dynamo protocol"
            )

        input_data = self.input_param_manager.get_input_param(
            request, use_tokenizer=True
        )


        if isinstance(input_data, list):
            prompt = TokensPrompt(prompt_token_ids=input_data)
        else:
            prompt = TextPrompt(prompt=input_data)


        sampling_params = build_sampling_params_openai(
            request, self.default_sampling_params
        )
        if self._llumnix_enabled:
            apply_resume_metadata(request, sampling_params)

        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))
        openai_request_id = request.get("id") or request.get("request_id", request_id)
        previous_text_per_choice: dict[int, str] = {}

        trace_headers = context.trace_headers()


        is_decode_only = self.config.disaggregation_mode == DisaggregationMode.DECODE
        async with _deferred_abort_guard(
            self.engine_client,
            request_id,
            is_decode_only,
            self._deferred_aborts,
            self._shutdown_on_engine_dead,
        ) as abort_guard, self._abort_monitor(
            context, request_id, abort_guard=abort_guard
        ):
            try:
                gen = self.engine_client.generate(
                    prompt,
                    sampling_params,
                    request_id,
                    data_parallel_rank=dp_rank,
                    trace_headers=trace_headers,
                    priority=priority,
                )

                async for res in gen:
                    if not res.outputs:
                        yield {
                            "id": openai_request_id,
                            "created": int(time.time()),
                            "object": "chat.completion.chunk",
                            "model": "unknown",
                            "choices": [
                                {
                                    "index": 0,
                                    "delta": {"role": "assistant", "content": ""},
                                    "finish_reason": "error",
                                }
                            ],
                        }
                        break

                    for output in res.outputs:
                        if abort_guard is not None:
                            abort_guard.signal_first_token()
                        output_idx = getattr(output, "index", 0) or 0
                        previous_text = previous_text_per_choice.get(output_idx, "")

                        delta_text = output.text[len(previous_text) :]

                        choice_data = {
                            "index": output_idx,
                            "delta": {
                                "role": "assistant",
                                "content": delta_text,
                            },
                            "finish_reason": normalize_finish_reason(
                                output.finish_reason
                            ),
                        }

                        chunk = {
                            "id": openai_request_id,
                            "created": int(time.time()),
                            "object": "chat.completion.chunk",
                            "model": "unknown",
                            "choices": [choice_data],
                        }

                        if output.finish_reason:
                            chunk["usage"] = BaseWorkerHandler._build_completion_usage(
                                request_output=res,
                            )

                        yield chunk
                        previous_text_per_choice[output_idx] = output.text

            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)


class PrefillWorkerHandler(BaseWorkerHandler):
    def __init__(
        self,
        runtime,
        config: Config,
        engine,
        default_sampling_params,
        model_max_len: int | None = None,
        model_config: ModelConfig | None = None,
        enable_multimodal: bool = False,
        generate_endpoint=None,
        use_vllm_tokenizer: bool = False,
        shutdown_event: asyncio.Event | None = None,
        enable_frontend_decoding: bool = False,
        encode_worker_client: Client | None = None,
    ):
        super().__init__(
            runtime,
            config,
            engine,
            default_sampling_params,
            model_max_len=model_max_len,
            model_config=model_config,
            enable_multimodal=enable_multimodal,
            generate_endpoint=generate_endpoint,
            use_vllm_tokenizer=use_vllm_tokenizer,
            shutdown_event=shutdown_event,
            enable_frontend_decoding=enable_frontend_decoding,
            encode_worker_client=encode_worker_client,
        )


        if resolve_model_family(config.model) is ModelFamily.QWEN_VL:
            self._qwen_grid_params = load_qwen_grid_params(
                config.model,
                trust_remote_code=config.engine_args.trust_remote_code,
            )
            if self._qwen_grid_params is None and self.embedding_loader is None:
                logger.error(
                    "Qwen VL grid params failed to load and no encode worker "
                    "is configured. P/D multimodal requests will fail because "
                    "prefill cannot produce embedding_params for decode. "
                    "Use --route-to-encoder or ensure the model is cached."
                )
        else:
            self._qwen_grid_params = None

    async def generate(self, request, context):

        request_id = context.id()
        logger.debug(f"Prefill Request ID: {request_id}")


        with time_and_log_code_section(f"[PREFILL] request: {request_id} generate"):
            async for chunk in self._generate_token_mode(request, context, request_id):
                yield chunk

    async def _generate_token_mode(self, request, context, request_id):


        mm_processor_kwargs = self._get_mm_processor_kwargs(request)


        multi_modal_data = await self._extract_multimodal_data(
            request,
            request_id,
            context,
            mm_processor_kwargs=mm_processor_kwargs,
        )


        prompt, embedding_sequence_length, error = self._build_prompt_from_request(
            request,
            request_id,
            multi_modal_data,
            log_prefix="Prefill ",
            mm_processor_kwargs=mm_processor_kwargs,
        )
        if error is not None:

            error["disaggregated_params"] = None
            yield error
            return

        _apply_nvext_cache_salt(request, prompt)


        sampling_params = build_sampling_params(
            request,
            self.default_sampling_params,
            self.model_max_len,
            enable_rl=self.config.enable_rl,
        )


        kv_protocol: KvConnectorProtocol = make_kv_connector_protocol(
            self.engine_client.vllm_config
        )
        if sampling_params.extra_args is None:
            sampling_params.extra_args = {}
        sampling_params.extra_args[
            "kv_transfer_params"
        ] = kv_protocol.prefill_request_kv_transfer_params()

        sampling_params.max_tokens = 1
        sampling_params.min_tokens = 1


        model_name = request.get("model")
        lora_request = self._resolve_lora_request(model_name)
        if lora_request:
            logger.info(
                f"Prefill request {request_id} will use LoRA adapter: {model_name} "
                f"(ID: {lora_request.lora_int_id}), path: {lora_request.lora_path}"
            )
        else:
            logger.debug(
                f"Prefill request {request_id} has no LoRA specified (model: {model_name})"
            )

        routing = request.get("routing") or {}
        dp_rank = self._to_local_dp_rank(routing.get("dp_rank"))
        priority = -int(routing.get("priority", 0))

        trace_headers = context.trace_headers()
        reasoning_ended, reasoning_parser_kwargs = _request_reasoning_metadata(request)

        async with self._abort_monitor(context, request_id, is_prefill=True):
            try:
                gen = self._generate_with_lora_admission_lock(
                    lora_request,
                    lambda admitted_lora_request: self.engine_client.generate(
                        prompt,
                        sampling_params,
                        request_id,
                        data_parallel_rank=dp_rank,
                        lora_request=admitted_lora_request,
                        trace_headers=trace_headers,
                        priority=priority,
                        **_engine_generate_reasoning_kwargs(
                            self.engine_client,
                            reasoning_ended,
                            reasoning_parser_kwargs,
                        ),
                    ),
                )
            except EngineDeadError as e:
                logger.error(f"vLLM EngineDeadError: {e}")
                logger.warning("Initiating Dynamo Runtime shutdown.")
                self.runtime.shutdown()
                os._exit(1)

            async for res in gen:
                logger.debug(f"kv transfer params: {res.kv_transfer_params}")

                token_ids = res.outputs[0].token_ids if res.outputs else []


                embedding_params = self._build_embedding_params(
                    multi_modal_data or {}, res.prompt_token_ids
                )

                output: Dict[str, Any] = {
                    "token_ids": list(token_ids),
                    "disaggregated_params": self._build_disaggregated_params(
                        kv_protocol.decode_request_kv_transfer_params(res),
                        embedding_params,
                    ),
                    "completion_usage": BaseWorkerHandler._build_completion_usage(
                        request_output=res,
                        embedding_sequence_length=embedding_sequence_length,
                    ),
                }


                self._log_with_lora_context(
                    "Prefill completed for request {request_id}{lora_info}: "
                    "generated {token_count} token(s), has_kv_params={has_kv_params}",
                    request_id,
                    lora_request,
                    level="info" if lora_request else "debug",
                    token_count=len(token_ids),
                    has_kv_params=res.kv_transfer_params is not None,
                )

                yield output

    def _build_disaggregated_params(
        self, kv_transfer_params, embedding_params=None, expanded_prompt_token_ids=None
    ):
        disaggregated_params = {}
        if kv_transfer_params is not None:
            disaggregated_params["kv_transfer_params"] = kv_transfer_params
        if embedding_params is not None:
            disaggregated_params["embedding_params"] = embedding_params
        if expanded_prompt_token_ids is not None:
            disaggregated_params[
                "expanded_prompt_token_ids"
            ] = expanded_prompt_token_ids

        return disaggregated_params if disaggregated_params else None

    def _build_embedding_params(
        self, multi_modal_data: dict[str, Any], prompt_token_ids: list[int]
    ) -> Dict[str, Any] | None:


        if resolve_model_family(self.config.model) is not ModelFamily.QWEN_VL:


            if multi_modal_data:
                return {"expanded_prompt_token_ids": prompt_token_ids}
        else:


            return build_qwen_embedding_params(multi_modal_data, self._qwen_grid_params)
        return None


class EmbeddingWorkerHandler:


    def __init__(
        self,
        runtime,
        engine: Any,
        config: Config,
        shutdown_event: Optional[asyncio.Event] = None,
    ) -> None:
        self.runtime = runtime
        self.engine_client = engine
        self.config = config
        self.shutdown_event = shutdown_event


        self.engine_monitor = VllmEngineMonitor(runtime, engine, shutdown_event)
        logger.info("Embedding worker handler initialized")

    def cleanup(self) -> None:


        return None

    async def _monitor_abort(self, context: Context, request_id: str) -> None:


        shutdown_task: Optional[asyncio.Task] = None
        try:


            wait_for: list[Any] = [context.async_killed_or_stopped()]
            if self.shutdown_event is not None:
                shutdown_task = asyncio.create_task(self.shutdown_event.wait())
                wait_for.append(shutdown_task)

            done, pending = await asyncio.wait(
                wait_for, return_when=asyncio.FIRST_COMPLETED
            )

            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            logger.debug(f"Aborting embedding request ID: {request_id}")
            try:
                await asyncio.shield(self.engine_client.abort(request_id))
            except asyncio.CancelledError:
                logger.debug(
                    f"Abort shielded from cancellation for embedding request "
                    f"{request_id}, continuing in background"
                )

            if shutdown_task is not None and shutdown_task in done:
                raise EngineShutdown("Engine was shut down during embedding.")
        except asyncio.CancelledError:
            pass
        except EngineShutdown:
            raise
        except Exception as e:


            logger.error(
                f"Error in embedding abort monitor for request {request_id}: {e}"
            )
            raise
        finally:


            if shutdown_task is not None and not shutdown_task.done():
                shutdown_task.cancel()
                try:
                    await shutdown_task
                except asyncio.CancelledError:
                    pass

    @asynccontextmanager
    async def _abort_monitor(self, context: Context, request_id: str):


        task = asyncio.create_task(self._monitor_abort(context, request_id))
        try:
            yield task
        finally:
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            else:

                task.result()

    async def generate(
        self, request: dict, context: Context
    ) -> AsyncIterator[Dict[str, Any]]:


        model_name = request.get("model") or self.config.served_model_name or ""
        input_field = request.get("input")
        if input_field is None:
            raise ValueError("Embedding request missing required 'input' field")


        prompts: list[Any] = _classify_embedding_input(input_field)

        dimensions = request.get("dimensions")
        if dimensions is not None and not isinstance(dimensions, int):
            raise TypeError(
                f"Invalid 'dimensions' type {type(dimensions).__name__}; expected int"
            )
        if dimensions is not None and dimensions < 1:
            raise ValueError(f"dimensions must be >= 1, got {dimensions}")

        encoding_format = request.get("encoding_format", "float")
        if encoding_format not in ("float", "base64"):
            raise ValueError(
                f"Invalid 'encoding_format' value {encoding_format!r}; "
                "expected 'float' or 'base64'"
            )

        truncate_prompt_tokens = request.get("truncate_prompt_tokens")
        tokenization_kwargs: dict[str, Any] | None = None
        if truncate_prompt_tokens is not None:
            if not isinstance(truncate_prompt_tokens, int) or isinstance(
                truncate_prompt_tokens, bool
            ):
                raise TypeError(
                    "Invalid 'truncate_prompt_tokens' type "
                    f"{type(truncate_prompt_tokens).__name__}; expected int"
                )
            if truncate_prompt_tokens < -1:
                raise ValueError(
                    "truncate_prompt_tokens must be >= -1, "
                    f"got {truncate_prompt_tokens}"
                )
            tokenization_kwargs = {
                "truncate_prompt_tokens": truncate_prompt_tokens,
            }


        pooling_kwargs: dict[str, Any] = {"task": "embed"}
        if dimensions is not None:
            pooling_kwargs["dimensions"] = dimensions
        pooling_params = PoolingParams(**pooling_kwargs)


        base_request_id = context.id()

        async def _encode_one(idx: int, prompt: Any):
            request_id = f"{base_request_id}-{idx}"
            encode_arg: Any = (
                prompt
                if isinstance(prompt, str)
                else TokensPrompt(prompt_token_ids=prompt)
            )
            final_output = None
            async with self._abort_monitor(context, request_id):
                encode_kwargs: dict[str, Any] = {
                    "prompt": encode_arg,
                    "pooling_params": pooling_params,
                    "request_id": request_id,
                }
                if tokenization_kwargs is not None and isinstance(encode_arg, str):
                    encode_kwargs["tokenization_kwargs"] = tokenization_kwargs

                async for out in self.engine_client.encode(**encode_kwargs):
                    final_output = out
            if final_output is None:
                raise RuntimeError(
                    f"vLLM engine.encode produced no output for input index {idx}"
                )
            return final_output


        tasks = [asyncio.create_task(_encode_one(i, p)) for i, p in enumerate(prompts)]
        try:
            outputs = await asyncio.gather(*tasks)
        finally:
            pending = [t for t in tasks if not t.done()]
            for t in pending:
                t.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)

        embedding_objects: list[Dict[str, Any]] = []
        prompt_tokens = 0
        for idx, final_output in enumerate(outputs):


            embedding = _pooling_output_to_list(final_output.outputs.data)


            if dimensions is not None and len(embedding) < dimensions:
                raise ValueError(
                    f"dimensions={dimensions} exceeds model embedding "
                    f"dimension {len(embedding)}"
                )


            embedding_objects.append(
                {
                    "object": "embedding",
                    "embedding": _encode_floats_to_base64(embedding),
                    "index": idx,
                }
            )
            token_ids = getattr(final_output, "prompt_token_ids", None) or []
            prompt_tokens += len(token_ids)

        yield {
            "object": "list",
            "data": embedding_objects,
            "model": model_name,
            "usage": {
                "prompt_tokens": prompt_tokens,
                "total_tokens": prompt_tokens,
            },
        }


def _is_token_id(x: Any) -> bool:


    return isinstance(x, int) and not isinstance(x, bool)


def _classify_embedding_input(input_field: Any) -> list[Any]:


    if isinstance(input_field, str):
        return [input_field]
    if not isinstance(input_field, list):
        raise TypeError(
            f"Invalid 'input' type {type(input_field).__name__}; "
            "expected str, list[str], list[int], or list[list[int]]"
        )
    if not input_field:
        raise ValueError("Embedding request 'input' must be non-empty")

    first = input_field[0]
    if isinstance(first, str):
        texts: list[str] = []
        for item in input_field:
            if not isinstance(item, str):
                raise TypeError(
                    "'input' list mixes str and non-str entries; pass either "
                    "all strings or all token-id arrays"
                )
            texts.append(item)
        return texts
    if _is_token_id(first):
        token_ids: list[int] = []
        for item in input_field:
            if not _is_token_id(item):
                raise TypeError(
                    "'input' list mixes int and non-int entries; for tokenized "
                    "input pass all integers (single prompt) or list[list[int]]"
                )
            token_ids.append(item)

        return [token_ids]
    if isinstance(first, list):
        prompts: list[list[int]] = []
        for i, item in enumerate(input_field):
            if not isinstance(item, list):
                raise TypeError(
                    f"'input' list element at index {i} must be a list of "
                    "ints (token IDs); mixed batches are not supported"
                )
            inner: list[int] = []
            for x in item:
                if not _is_token_id(x):
                    raise TypeError(
                        f"'input' list element at index {i} must be a list of "
                        "ints (token IDs); mixed batches are not supported"
                    )
                inner.append(x)
            prompts.append(inner)
        return prompts
    raise TypeError(
        f"Unsupported 'input' element type {type(first).__name__}; "
        "expected str, int, or list[int]"
    )


def _pooling_output_to_list(data: Any) -> list[float]:


    if isinstance(data, torch.Tensor):
        return data.detach().cpu().flatten().tolist()
    if isinstance(data, (list, tuple)):

        if data and isinstance(data[0], (list, tuple)):
            return [float(x) for row in data for x in row]
        return [float(x) for x in data]
    raise TypeError(
        f"Unsupported PoolingOutput.data type {type(data).__name__}; "
        "expected torch.Tensor or list"
    )


def _encode_floats_to_base64(floats: list[float]) -> str:


    packed = struct.pack(f"<{len(floats)}f", *floats)
    return base64.b64encode(packed).decode("ascii")
