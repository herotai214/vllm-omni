# Video Endpoints

This package owns OpenAI-compatible video endpoint families.

## Put Here

- Video generation/resource endpoint code under `generation/`.
- Generated video streaming-output code under a dedicated subpackage when that
  route moves.
- Video-specific request parsing, response formatting, and storage/job helpers.

## Do Not Put Here

- Live streaming-input chat sessions such as `/v1/video/chat/stream`; those
  should move under `openai/streaming_input/video_chat`.
- Generic server utilities.
- Image or audio endpoint behavior.

`video_api_utils.py` still lives at the OpenAI package root and remains an
acceptable temporary home during this helper-only PR. Its helpers should be
revisited and likely split during the video refactor tracked by #5227.
