# OpenAI Entrypoint Helpers

This package contains OpenAI-compatible serving code for vLLM-Omni.

## What Belongs Here

- OpenAI protocol or response-shape helpers shared by multiple OpenAI endpoint
  families.
- App-state accessors for OpenAI serving objects, for example helpers that read
  `request.app.state.openai_serving_*`.
- Bootstrap helpers that are specific to OpenAI-compatible serving and are not
  route bodies.

## What Does Not Belong Here

- FastAPI app/server mechanics that are not OpenAI-specific. Put those under
  `vllm_omni.entrypoints.serve.utils`.
- Modality-specific request parsing, validation, or response formatting. Put
  those under the owning endpoint package, such as `images/` or
  `video/generation/`.
- Broad cleanup of existing utility files such as `image_api_utils.py`,
  `video_api_utils.py`, `audio_utils_mixin.py`, and `utils.py`.

Those older utility files remain acceptable temporary homes in this PR. They
should be revisited and likely split or renamed during the later
endpoint-family stages tracked by #5227.
