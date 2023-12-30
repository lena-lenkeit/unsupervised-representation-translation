from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Int64


# Custom loss functions
def clm_loss_fn(
    logits: Float[torch.Tensor, "batch sequence features"],
    labels: Int64[torch.Tensor, "batch sequence"],
    mask: Optional[Bool[torch.Tensor, "batch sequence"]] = None,
) -> Float[torch.Tensor, ""]:
    # Shift so that tokens < n predict n
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = rearrange(logits, "b s f -> (b s) f")
    labels = rearrange(labels, "b s -> (b s)")

    # Calculate next-token loss
    if mask is None:
        loss = F.cross_entropy(logits, labels, reduction="mean")
    else:
        loss = F.cross_entropy(logits, labels, reduction="none")
        mask = mask[..., 1:].contiguous()
        mask = rearrange(mask, "b s -> (b s)")
        loss = torch.mean(loss * mask)

    return loss
