# Video Generation

This package owns the `/v1/videos` resource family.

## Put Here

- Multipart form parsing for video generation requests.
- Async/sync video generation job helpers.
- Video resource status, delete, retrieve, and download helpers.
- The `api_router.py` for `/v1/videos*` once route bodies move out of
  `api_server.py`.

## Do Not Put Here

- Live video-chat streaming input.
- Generated video streaming-output sessions.
- Generic video encoding helpers used outside the generation resource family.

The helper migration places non-route video generation code here first. Full
route extraction and `video_api_utils.py` cleanup are tracked by #5227.
