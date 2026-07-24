# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from fastapi import Request

from vllm_omni.entrypoints.openai.serving_audio_generate import OmniOpenAIServingAudioGenerate
from vllm_omni.entrypoints.openai.serving_chat import OmniOpenAIServingChat
from vllm_omni.entrypoints.openai.serving_speech import OmniOpenAIServingSpeech
from vllm_omni.entrypoints.openai.serving_video import OmniOpenAIServingVideo


def Omnivideo(request: Request) -> OmniOpenAIServingVideo | None:
    return request.app.state.openai_serving_video


def Omnichat(request: Request) -> OmniOpenAIServingChat | None:
    return request.app.state.openai_serving_chat


def Omnispeech(request: Request) -> OmniOpenAIServingSpeech | None:
    return request.app.state.openai_serving_speech


def OmniAudioGenerate(request: Request) -> OmniOpenAIServingAudioGenerate | None:
    return getattr(request.app.state, "openai_serving_audio_generate", None)
