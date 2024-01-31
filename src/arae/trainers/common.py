from dataclasses import dataclass
from enum import Enum
from typing import Optional

import torch
from jaxtyping import Int64


@dataclass
class TokenIds:
    input_ids: Int64[torch.Tensor, "batch input_sequence"]
    attention_mask: Int64[torch.Tensor, "batch input_sequence"]
    labels: Optional[Int64[torch.Tensor, "batch target_sequence"]] = None


class Mode(str, Enum):
    SIM = "SIM"
    ALT = "ALT"


def sum_default(values: list, default: float = 0.0):
    return sum([value if value is not None else default for value in values])


def exclude_none(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}
