import torch
from jaxtyping import Float, Int64


def gather_from_tokens(
    key: Int64[torch.Tensor, "batch sequence"],
    values: Float[torch.Tensor, "batch sequence features"],
    query: int,
) -> Float[torch.Tensor, "batch features"]:
    batch, sequence, features = values.shape

    _, index = torch.where(key == query)
    assert index.shape[0] == values.shape[0], "Matched multiple tokens"

    index = index.reshape(batch, 1, 1)
    index = index.expand(batch, 1, features)
    out = torch.gather(values, 1, index)
    out = out.reshape(batch, features)

    return out


def scatter_to_tokens(
    key: Int64[torch.Tensor, "batch sequence"],
    source: Float[torch.Tensor, "batch features"],
    values: Float[torch.Tensor, "batch sequence features"],
    query: int,
) -> Float[torch.Tensor, "batch sequence features"]:
    batch, sequence, features = values.shape

    _, index = torch.where(key == query)
    assert index.shape[0] == values.shape[0], "Matched multiple tokens"

    index = index.reshape(batch, 1, 1)
    index = index.expand(batch, 1, features)
    source = source.reshape(batch, 1, features)
    out = torch.scatter(values, 1, index, source)

    return out
