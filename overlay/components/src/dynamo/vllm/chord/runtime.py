from __future__ import annotations

from typing import Any


def install_engine_core_utilities() -> None:


    from vllm.v1.engine.core import EngineCore

    if getattr(EngineCore, "_dynamo_chord_utilities_installed", False):
        return

    def adapter(core: Any) -> Any:
        value = getattr(core.scheduler, "_chord_adapter", None)
        if value is None:
            raise RuntimeError("Chord scheduler adapter is unavailable")
        return value

    def chord_status(self, body: dict[str, Any]):
        return adapter(self).collect(body)

    def chord_drop_epoch(self, body: dict[str, Any]):
        return adapter(self).drop_epoch(body)

    EngineCore.chord_status = chord_status
    EngineCore.chord_drop_epoch = chord_drop_epoch
    EngineCore._dynamo_chord_utilities_installed = True
