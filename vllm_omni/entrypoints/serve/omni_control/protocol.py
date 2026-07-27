# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from pydantic import BaseModel


class OmniSleepRequest(BaseModel):
    stage_ids: list[int]
    level: int = 2


class OmniWakeupRequest(BaseModel):
    stage_ids: list[int]
