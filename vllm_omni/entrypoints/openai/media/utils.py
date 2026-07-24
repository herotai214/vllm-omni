# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import Any


def _update_if_not_none(object: Any, key: str, val: Any) -> None:
    if val is not None:
        setattr(object, key, val)
