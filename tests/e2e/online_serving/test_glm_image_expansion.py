"""
Comprehensive tests of diffusion features that are available in online serving mode
and are supported by the following models:
- zai-org/GLM-Image: (supports t2i & i2i)
Coverage:
    For both t2i & i2i cases:
    - Baseline
    - Cache-Dit

assert_diffusion_response validates successful generation and the expected
1024x1024 resolution.
"""

import pytest

from tests.helpers.mark import hardware_marks
from tests.helpers.media import generate_synthetic_image
from tests.helpers.runtime import (
    OmniServer,
    OmniServerParams,
    OpenAIClientHandler,
    dummy_messages_from_mix_data,
)

EDIT_PROMPT = "Transform this modern, geometric image into a Vincent van Gogh style impressionist painting."
NEGATIVE_PROMPT = "low quality, blurry, distorted, unnatural colors"

# By default, stage 0 and 1 of GLM-Image run at device 0 and 1 respecitively -> 2 cards in total
BASELINE_FEATURE_MARKS = hardware_marks(res={"cuda": "H100"}, num_cards=2)


def _get_diffusion_feature_cases(model: str):
    return [
        # Baseline (2 GPUs)
        pytest.param(
            OmniServerParams(model=model, server_args=[]),
            id="baseline",
            marks=BASELINE_FEATURE_MARKS,
        ),
        # Cache-Dit (2 GPUs)
        pytest.param(
            OmniServerParams(
                model=model,
                server_args=["--cache-backend", "cache_dit"],
            ),
            id="cachedit",
            marks=BASELINE_FEATURE_MARKS,
        ),
    ]


# Loop through both modes
MODES = ["t2i", "i2i"]


@pytest.mark.full_model
@pytest.mark.diffusion
@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize(
    "omni_server",
    _get_diffusion_feature_cases("/mnt/nvme3n1/models/GLM-Image"),
    indirect=True,
)
def test_glm_image(
    omni_server: OmniServer,
    mode: str,
    openai_client: OpenAIClientHandler,
):
    """Test GLM-Image in both T2I and I2I modes across all configurations."""

    if mode == "t2i":
        # Text‑only input
        messages = dummy_messages_from_mix_data(content_text=EDIT_PROMPT)
    else:  # i2i
        # Image + text input
        image_size = 1024
        image_data_url = f"data:image/jpeg;base64,{generate_synthetic_image(image_size, image_size)['base64']}"
        messages = dummy_messages_from_mix_data(
            image_data_url=image_data_url,
            content_text=EDIT_PROMPT,
        )

    request_config = {
        "model": omni_server.model,
        "messages": messages,
        "extra_body": {
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 2,
            "guidance_scale": 1.5,
            "true_cfg_scale": 4.0,
            "negative_prompt": NEGATIVE_PROMPT,
            "seed": 42,
        },
    }

    openai_client.send_diffusion_request(request_config)
