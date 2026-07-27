# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import asyncio
import json
import os
import time
from http import HTTPStatus
from typing import Any, Literal, cast

from fastapi import File, Form, HTTPException, Request, UploadFile
from PIL import Image
from vllm.entrypoints.launcher import terminate_if_errored
from vllm.entrypoints.serve.utils.error_response import create_error_response
from vllm.logger import init_logger
from vllm.v1.engine.exceptions import EngineDeadError, EngineGenerateError

from vllm_omni.diffusion.models.interface import ReferenceVideoDecodeSpec
from vllm_omni.entrypoints.openai.app_state import Omnivideo
from vllm_omni.entrypoints.openai.errors import InvalidInputReferenceError
from vllm_omni.entrypoints.openai.protocol.videos import (
    SecondStr,
    SizeStr,
    VideoError,
    VideoGenerationRequest,
    VideoGenerationStatus,
    VideoResponse,
)
from vllm_omni.entrypoints.openai.serving_video import (
    OmniOpenAIServingVideo,
    ReferenceAudio,
    ReferenceImage,
    ReferenceVideo,
)
from vllm_omni.entrypoints.openai.storage import STORAGE_MANAGER
from vllm_omni.entrypoints.openai.stores import VIDEO_STORE
from vllm_omni.entrypoints.openai.utils import get_stage_type
from vllm_omni.entrypoints.openai.video_api_utils import decode_audio_url, decode_input_reference
from vllm_omni.errors import OmniClientError

logger = init_logger(__name__)

VIDEO_SYNC_TIMEOUT_S = float(os.environ.get("VLLM_OMNI_VIDEO_SYNC_TIMEOUT", 600.0))


def _resolve_video_runtime_context(raw_request: Request) -> tuple[str | None, list[Any] | None]:
    app_model_name = None
    serving_models = getattr(raw_request.app.state, "openai_serving_models", None)
    if serving_models and getattr(serving_models, "base_model_paths", None):
        base_paths = serving_models.base_model_paths
        if base_paths:
            app_model_name = base_paths[0].name

    app_stage_configs = getattr(raw_request.app.state, "stage_configs", None)
    return app_model_name, app_stage_configs


def _parse_form_json(value: str | None, expected_type: type | None = None) -> Any:
    if value is None or value == "":
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Invalid JSON in form field.",
        ) from exc
    if expected_type is not None and not isinstance(parsed, expected_type):
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail=f"Invalid JSON in form field: expected {expected_type.__name__}, got {type(parsed).__name__}.",
        )
    return parsed


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, dict):
        return config.get(key, default)
    if hasattr(config, "get"):
        try:
            return config.get(key, default)
        except Exception:
            pass
    return getattr(config, key, default)


def _stage_engine_args(stage_cfg: Any) -> Any:
    return _config_get(stage_cfg, "engine_args", {}) or {}


def _diffusion_model_classes(stage_configs: list[Any] | None) -> list[type]:
    if not stage_configs:
        return []

    from vllm_omni.diffusion.registry import DiffusionModelRegistry

    model_classes: list[type] = []
    for stage_cfg in stage_configs:
        if get_stage_type(stage_cfg) != "diffusion":
            continue
        model_class_name = _config_get(_stage_engine_args(stage_cfg), "model_class_name")
        if not model_class_name:
            continue
        model_cls = DiffusionModelRegistry._try_load_model_cls(model_class_name)
        if model_cls is not None:
            model_classes.append(model_cls)
    return model_classes


def _normalize_reference_video_decode_spec(spec: ReferenceVideoDecodeSpec) -> ReferenceVideoDecodeSpec:
    max_frames = spec.max_frames
    if max_frames is not None:
        try:
            max_frames = int(max_frames)
        except (TypeError, ValueError) as exc:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid reference video decode spec: max_frames must be an integer.",
            ) from exc
        if max_frames <= 0:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail="Invalid reference video decode spec: max_frames must be positive.",
            )

    keep = str(spec.keep or "first").strip().lower()
    if keep not in {"first", "last"}:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Invalid reference video decode spec: keep must be either 'first' or 'last'.",
        )
    return ReferenceVideoDecodeSpec(max_frames=max_frames, keep=cast(Literal["first", "last"], keep))


def _reference_video_decode_spec(
    req: VideoGenerationRequest,
    stage_configs: list[Any] | None,
) -> ReferenceVideoDecodeSpec:
    video_params = req.resolve_video_params()
    extra_params = req.extra_params if isinstance(req.extra_params, dict) else {}
    for model_cls in _diffusion_model_classes(stage_configs):
        resolver = getattr(model_cls, "reference_video_decode_spec", None)
        if resolver is None:
            continue
        try:
            spec = resolver(num_frames=video_params.num_frames, extra_args=extra_params)
        except (TypeError, ValueError) as exc:
            raise HTTPException(status_code=HTTPStatus.BAD_REQUEST.value, detail=str(exc)) from exc
        if spec is not None:
            return _normalize_reference_video_decode_spec(spec)
    return ReferenceVideoDecodeSpec(max_frames=video_params.num_frames, keep="first")


def video_response_from_request(model_name: str, req: VideoGenerationRequest) -> VideoResponse:
    resp = VideoResponse(
        model=model_name,
        status=VideoGenerationStatus.QUEUED,
        size=req.size,
        prompt=req.prompt,
    )
    resp.seconds = str(req.seconds or resp.seconds)
    return resp


def _status_code_for_video_failure(error: VideoError | None) -> int:
    if error is None:
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if isinstance(error.code, int):
        if 400 <= error.code < 600:
            return error.code
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if error.code == "HTTPException":
        status_text, _, _ = error.message.partition(":")
        try:
            status_code = int(status_text)
        except ValueError:
            return HTTPStatus.INTERNAL_SERVER_ERROR.value
        if 400 <= status_code < 600:
            return status_code
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    if error.code == "EngineDeadError":
        return HTTPStatus.INTERNAL_SERVER_ERROR.value
    if error.code == "EngineGenerateError":
        return HTTPStatus.INTERNAL_SERVER_ERROR.value

    return HTTPStatus.INTERNAL_SERVER_ERROR.value


def _video_error_from_exception(exc: Exception) -> VideoError:
    if isinstance(exc, HTTPException):
        message = str(exc.detail) if exc.detail else str(exc)
        return VideoError(code=exc.status_code, message=message)

    if isinstance(exc, OmniClientError):
        return VideoError(code=exc.status_code, message=exc.message)

    if isinstance(exc, (EngineGenerateError, EngineDeadError)):
        err = create_error_response(exc)
        return VideoError(code=err.error.code, message=err.error.message)

    return VideoError(
        code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
        message=str(exc),
    )


async def _cleanup_video(video_id: str):
    try:
        await STORAGE_MANAGER.delete(video_id)
    except Exception:
        logger.warning("Failed to cleanup partial video file '%s'", video_id)


async def _run_video_generation_job(
    handler: OmniOpenAIServingVideo,
    request: VideoGenerationRequest,
    video_id: str,
    reference_image: ReferenceImage | None = None,
    reference_video: ReferenceVideo | None = None,
    reference_audio: ReferenceAudio | None = None,
    app_state: Any | None = None,
) -> None:
    job = await VIDEO_STORE.get(video_id)
    if job is None:
        logger.warning("Video job %s missing before generation task started; skipping", video_id)
        return

    await VIDEO_STORE.update_fields(video_id, {"status": VideoGenerationStatus.IN_PROGRESS})
    started_at = time.perf_counter()
    try:
        video_bytes, stage_durations, peak_memory_mb, action = await handler.generate_video_bytes(
            request,
            video_id,
            reference_image=reference_image,
            reference_video=reference_video,
            reference_audio=reference_audio,
        )

        save_context = await STORAGE_MANAGER.save(video_bytes, video_id)
        logger.info("Video request %s persisted %s output file.", video_id, save_context.key)

        updated_fields = {
            "status": VideoGenerationStatus.COMPLETED,
            "progress": 100,
            "file_name": f"{video_id}.{job.file_extension}",
            "completed_at": save_context.created_at,
            "inference_time_s": time.perf_counter() - started_at,
            "stage_durations": stage_durations,
            "peak_memory_mb": peak_memory_mb,
            "action": action,
        }
        if save_context.expires_at is not None:
            updated_fields["expires_at"] = save_context.expires_at

        await VIDEO_STORE.update_fields(video_id, updated_fields)
    except (EngineGenerateError, EngineDeadError) as exc:
        logger.exception("Video generation failed (engine error) for id=%s", video_id)

        await _cleanup_video(video_id)
        await VIDEO_STORE.update_fields(
            video_id,
            {
                "status": VideoGenerationStatus.FAILED,
                "completed_at": int(time.time()),
                "error": _video_error_from_exception(exc),
                "inference_time_s": time.perf_counter() - started_at,
            },
        )
        # Background tasks can't propagate exceptions to FastAPI handlers.
        # Actively signal shutdown when the engine is dead.
        if app_state is not None and isinstance(exc, EngineDeadError):
            terminate_if_errored(
                server=app_state.server,
                engine=app_state.engine_client,
            )
    except Exception as exc:
        logger.exception("Video generation failed for id=%s", video_id)

        await _cleanup_video(video_id)
        await VIDEO_STORE.update_fields(
            video_id,
            {
                "status": VideoGenerationStatus.FAILED,
                "completed_at": int(time.time()),
                "error": _video_error_from_exception(exc),
                "inference_time_s": time.perf_counter() - started_at,
            },
        )
    except asyncio.CancelledError:
        await _cleanup_video(video_id)
        await VIDEO_STORE.pop(video_id)
        raise
    finally:
        if reference_audio is not None and os.path.exists(reference_audio.path):
            os.unlink(reference_audio.path)


async def _parse_video_form(
    raw_request: Request,
    prompt: str = Form(...),
    input_reference: UploadFile | None = File(default=None),
    image_reference: str | None = Form(default=None),
    video_reference: str | None = Form(default=None),
    audio_reference: str | None = Form(default=None),
    model: str | None = Form(default=None),
    seconds: SecondStr | None = Form(default=None),
    size: SizeStr | None = Form(default=None),
    user: str | None = Form(default=None),
    width: int | None = Form(default=None),
    height: int | None = Form(default=None),
    num_frames: int | None = Form(default=None),
    fps: int | None = Form(default=None),
    num_inference_steps: int | None = Form(default=None),
    guidance_scale: float | None = Form(default=None),
    guidance_scale_2: float | None = Form(default=None),
    boundary_ratio: float | None = Form(default=None),
    flow_shift: float | None = Form(default=None),
    true_cfg_scale: float | None = Form(default=None),
    seed: int | None = Form(default=None),
    generate_sound: bool | None = Form(default=None),
    sound_duration: float | None = Form(default=None, gt=0.0),
    negative_prompt: str | None = Form(default=None),
    enable_frame_interpolation: bool | None = Form(default=None),
    frame_interpolation_exp: int | None = Form(default=None, ge=1),
    frame_interpolation_scale: float | None = Form(default=None, gt=0.0),
    frame_interpolation_model_path: str | None = Form(default=None),
    lora: str | None = Form(default=None),
    extra_params: str | None = Form(default=None),
) -> tuple[
    VideoGenerationRequest,
    "OmniOpenAIServingVideo",
    str,
    ReferenceImage | None,
    ReferenceVideo | None,
    ReferenceAudio | None,
]:
    """FastAPI dependency that parses video form data, validates inputs,
    resolves the handler, and decodes any reference image.

    Used by both ``POST /v1/videos`` (async) and ``POST /v1/videos/sync``.
    """
    input_reference_bytes = await input_reference.read() if input_reference is not None else None
    parsed_image_reference = _parse_form_json(image_reference)
    parsed_video_reference = _parse_form_json(video_reference)
    parsed_audio_reference = _parse_form_json(audio_reference)

    provided_references = sum(
        item is not None for item in (parsed_image_reference, parsed_video_reference, input_reference_bytes)
    )
    if provided_references > 1:
        raise HTTPException(
            status_code=HTTPStatus.BAD_REQUEST.value,
            detail="Provide only one of input_reference, image_reference, or video_reference.",
        )

    request_data: dict[str, Any] = {
        "prompt": prompt,
        "model": model,
        "seconds": seconds,
        "size": size,
        "image_reference": parsed_image_reference,
        "video_reference": parsed_video_reference,
        "audio_reference": parsed_audio_reference,
        "user": user,
        "width": width,
        "height": height,
        "num_frames": num_frames,
        "fps": fps,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "guidance_scale_2": guidance_scale_2,
        "boundary_ratio": boundary_ratio,
        "flow_shift": flow_shift,
        "true_cfg_scale": true_cfg_scale,
        "seed": seed,
        "generate_sound": generate_sound,
        "sound_duration": sound_duration,
        "negative_prompt": negative_prompt,
        "enable_frame_interpolation": enable_frame_interpolation,
        "frame_interpolation_exp": frame_interpolation_exp,
        "frame_interpolation_scale": frame_interpolation_scale,
        "frame_interpolation_model_path": frame_interpolation_model_path,
        "lora": _parse_form_json(lora, expected_type=dict),
        "extra_params": _parse_form_json(extra_params, expected_type=dict),
    }
    request_data = {k: v for k, v in request_data.items() if v is not None}
    request = VideoGenerationRequest(**request_data)

    handler = Omnivideo(raw_request)
    if handler is None:
        raise HTTPException(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE.value,
            detail="Video generation handler not initialized.",
        )
    logger.info("Video generation handler: %s", type(handler).__name__)
    try:
        app_model_name, app_stage_configs = _resolve_video_runtime_context(raw_request)
        effective_model_name = handler.model_name or app_model_name or request.model or "unknown"
        if request.model is not None and effective_model_name is not None and request.model != effective_model_name:
            raise HTTPException(
                status_code=HTTPStatus.BAD_REQUEST.value,
                detail=(
                    f"Model mismatch: request specifies '{request.model}' but server is running "
                    f"'{effective_model_name}'."
                ),
            )
        handler.set_stage_configs_if_missing(app_stage_configs)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Video generation setup failed: %s", e)
        raise HTTPException(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR.value,
            detail=f"Video generation setup failed: {str(e)}",
        )

    decode_spec = ReferenceVideoDecodeSpec()
    if parsed_video_reference is not None or input_reference_bytes is not None:
        stage_configs = (
            handler.stage_configs
            or app_stage_configs
            or getattr(getattr(handler, "_engine_client", None), "stage_configs", None)
        )
        decode_spec = _reference_video_decode_spec(request, stage_configs)
    try:
        media_data = await decode_input_reference(
            request.image_reference,
            request.video_reference,
            input_reference_bytes,
            max_video_frames=decode_spec.max_frames,
            video_keep=decode_spec.keep,
        )
    except InvalidInputReferenceError as exc:
        raise HTTPException(400, detail=str(exc) or "Invalid input reference.") from exc

    reference_image = ReferenceImage(data=media_data) if isinstance(media_data, Image.Image) else None
    reference_video = ReferenceVideo(data=media_data) if isinstance(media_data, list) else None

    reference_audio: ReferenceAudio | None = None
    if request.audio_reference is not None:
        try:
            audio_path = await decode_audio_url(request.audio_reference.audio_url)
        except InvalidInputReferenceError as exc:
            raise HTTPException(400, detail=str(exc)) from exc
        reference_audio = ReferenceAudio(path=audio_path)

    return request, handler, effective_model_name, reference_image, reference_video, reference_audio
