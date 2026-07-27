# Serve Utilities

This directory is for server/app utilities shared by Omni entrypoints.

## Put Here

- FastAPI app/router mutation helpers.
- Server exception-handler registration.
- Engine-failure helpers that are not owned by one endpoint family.

## Do Not Put Here

- OpenAI response-shape adapters. Those belong under
  `vllm_omni.entrypoints.openai`.
- Modality validation or payload parsing.
- Generic helpers without a clear server/app responsibility.

Keep this directory aligned with the spirit of upstream
`vllm.entrypoints.serve.utils`. Broader utility cleanup is tracked by #5227.
