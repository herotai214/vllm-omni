# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any

from vllm.engine.protocol import EngineClient

ENDPOINT_LOAD_METRICS_FORMAT_HEADER_LABEL = "endpoint-load-metrics-format"


async def _get_vllm_config(engine_client: EngineClient) -> Any:
    if hasattr(engine_client, "get_vllm_config"):
        return await engine_client.get_vllm_config()
    return getattr(engine_client, "vllm_config", None)
