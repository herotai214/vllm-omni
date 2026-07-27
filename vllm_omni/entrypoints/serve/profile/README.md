# Profile Routes

This package owns profiling-control schemas and, after route extraction, should
own the profiling route bodies.

## Put Here

- Request/response models for `/start_profile` and `/stop_profile`.
- Profile-specific helper functions.
- The profile `api_router.py` once route bodies move out of `api_server.py`.

## Do Not Put Here

- General server error handling.
- Engine profiling implementation internals.
- OpenAI modality endpoint logic.

The route bodies still live in `openai/api_server.py` in this helper-only PR.
They should move here during the endpoint ownership work tracked by #5227.
