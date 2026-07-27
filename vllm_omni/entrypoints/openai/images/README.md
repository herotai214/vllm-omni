# Image Endpoints

This package owns OpenAI-compatible image generation and image edit endpoint
logic.

## Put Here

- Request parsing, validation, and response helpers used only by image
  endpoints.
- Image endpoint route bodies after route extraction.
- Image-specific bridge/adapters introduced during the image refactor.

## Do Not Put Here

- Generic server utilities.
- App-state accessors shared by multiple endpoint families.
- Video or audio endpoint behavior.

`image_api_utils.py` still lives at the OpenAI package root and remains an
acceptable temporary home during this helper-only PR. Its helpers should be
revisited and likely split or moved here during the image refactor tracked by
#5227.
