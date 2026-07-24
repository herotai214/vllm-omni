# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from http import HTTPStatus
from typing import Any

from fastapi import Request
from fastapi.responses import JSONResponse
from vllm.entrypoints.launcher import terminate_if_errored
from vllm.entrypoints.openai.engine.protocol import ErrorResponse
from vllm.entrypoints.serve.instrumentator.basic import base
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

logger = init_logger(__name__)


def _get_request_id_from_request(req: Request) -> str | None:
    return req.state.request_metadata.request_id if hasattr(req.state, "request_metadata") else None


def _build_engine_error_payload(
    exc: EngineDeadError | EngineGenerateError,
    *,
    request_id: str | None,
) -> tuple[dict[str, Any], int]:
    err = create_error_response(exc)
    payload = err.model_dump()
    error_body = payload.get("error", {})

    error_body["request_id"] = request_id
    error_body["error_stage_id"] = getattr(exc, "error_stage_id", None)

    return payload, err.error.code


def _create_engine_error_json_response(
    req: Request,
    exc: EngineDeadError | EngineGenerateError,
) -> JSONResponse:
    request_id = _get_request_id_from_request(req)
    error_stage_id = getattr(exc, "error_stage_id", None)
    engine = req.app.state.engine_client

    if isinstance(exc, EngineDeadError):
        # Log Omni-specific diagnostic information for dead engines.
        orchestrator_alive = engine.engine.is_alive() if hasattr(engine, "engine") else "N/A"
        logger.error(
            "EngineDeadError: orchestrator_alive=%s, errored=%s, request_id=%s, error_stage_id=%s",
            orchestrator_alive,
            engine.errored,
            request_id,
            error_stage_id,
        )

    terminate_if_errored(
        server=req.app.state.server,
        engine=engine,
    )

    payload, status_code = _build_engine_error_payload(exc, request_id=request_id)
    return JSONResponse(content=payload, status_code=status_code)


def _error_response_to_json_response(
    err: ErrorResponse,
    *,
    status_code: HTTPStatus | int | None = None,
    default_status_code: HTTPStatus | int = HTTPStatus.BAD_REQUEST,
) -> JSONResponse:
    resolved_status = int(
        status_code
        if status_code is not None
        else (err.error.code if err.error and err.error.code is not None else default_status_code)
    )
    payload = err.model_dump()
    if err.error:
        payload["error"]["code"] = resolved_status
    return JSONResponse(content=payload, status_code=resolved_status)


def _create_speech_error_json_response(
    raw_request: Request,
    message: str,
    *,
    err_type: str = "BadRequestError",
    status_code: HTTPStatus = HTTPStatus.BAD_REQUEST,
) -> JSONResponse:
    err = base(raw_request).create_error_response(
        message=message,
        err_type=err_type,
        status_code=status_code,
    )
    return _error_response_to_json_response(err, status_code=status_code)


def _register_omni_exception_handlers(app) -> None:
    """Override upstream vLLM exception handlers with Omni-aware versions."""

    async def omni_engine_error_handler(
        req: Request,
        exc: EngineDeadError | EngineGenerateError,
    ):
        request_id = _get_request_id_from_request(req)

        if req.app.state.args.log_error_stack:
            logger.exception("Engine Exception caught. Request id: %s", request_id)

        return _create_engine_error_json_response(req, exc)

    app.exception_handler(EngineGenerateError)(omni_engine_error_handler)
    app.exception_handler(EngineDeadError)(omni_engine_error_handler)
