from typing import List, Union

import torch
from jaxtyping import Float, Int64
from transformers import PreTrainedModel, PreTrainedTokenizer, PreTrainedTokenizerFast

from arae.tokens import ARAETokens, LabelTokens, PlaceholderTokens, TaskTokens, Token


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
    allow_multiple: bool = False,
) -> Float[torch.Tensor, "batch sequence features"]:
    batch, sequence, features = values.shape

    if not allow_multiple:
        _, index = torch.where(key == query)
        assert index.shape[0] == values.shape[0], "Matched multiple tokens"

        index = index.reshape(batch, 1, 1)
        index = index.expand(batch, 1, features)
        source = source.reshape(batch, 1, features)
        out = torch.scatter(values, 1, index, source)
    else:
        key = key.reshape(batch, sequence, 1)
        source = source.reshape(batch, 1, features)
        out = torch.where(key == query, source, values)

    return out


def add_tokens_to_model(
    model: PreTrainedModel,
    tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
) -> ARAETokens:
    # Define token strings
    TASK_CLM_TOKEN = "<|CLM-TASK|>"
    TASK_ENCODING_TOKEN = "<|ENC-TASK|>"
    TASK_DECODING_TOKEN = "<|DEC-TASK|>"
    TASK_CLASSIFICATION_TOKEN = "<|CLS-TASK|>"

    CLASS_A_TOKEN = "<|CLS-A|>"
    CLASS_B_TOKEN = "<|CLS-B|>"

    EMBEDDING_PLACEHOLDER_TOKEN = "<|EMB-PLACEHOLDER|>"
    SCORE_PLACEHOLDER_TOKEN = "<|SCORE-PLACEHOLDER|>"

    # Add to tokenizer
    tokenizer.add_tokens(
        [
            TASK_CLM_TOKEN,
            TASK_ENCODING_TOKEN,
            TASK_DECODING_TOKEN,
            TASK_CLASSIFICATION_TOKEN,
            CLASS_A_TOKEN,
            CLASS_B_TOKEN,
            EMBEDDING_PLACEHOLDER_TOKEN,
            SCORE_PLACEHOLDER_TOKEN,
        ],
        special_tokens=True,
    )

    # Expand embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Save new token information
    def assert_single(token_id: int | List[int]) -> int:
        if isinstance(token_id, list):
            raise TypeError(token_id)

        return token_id

    tokens = ARAETokens(
        task=TaskTokens(
            modeling=Token(
                id=assert_single(tokenizer.convert_tokens_to_ids(TASK_CLM_TOKEN)),
                text=TASK_CLM_TOKEN,
            ),
            encoding=Token(
                id=assert_single(tokenizer.convert_tokens_to_ids(TASK_ENCODING_TOKEN)),
                text=TASK_ENCODING_TOKEN,
            ),
            decoding=Token(
                id=assert_single(tokenizer.convert_tokens_to_ids(TASK_DECODING_TOKEN)),
                text=TASK_DECODING_TOKEN,
            ),
            classification=Token(
                id=assert_single(
                    tokenizer.convert_tokens_to_ids(TASK_CLASSIFICATION_TOKEN)
                ),
                text=TASK_CLASSIFICATION_TOKEN,
            ),
        ),
        placeholder=PlaceholderTokens(
            embedding=Token(
                id=assert_single(
                    tokenizer.convert_tokens_to_ids(EMBEDDING_PLACEHOLDER_TOKEN)
                ),
                text=EMBEDDING_PLACEHOLDER_TOKEN,
            ),
            label=Token(
                id=assert_single(
                    tokenizer.convert_tokens_to_ids(SCORE_PLACEHOLDER_TOKEN)
                ),
                text=SCORE_PLACEHOLDER_TOKEN,
            ),
        ),
        label=LabelTokens(
            a=Token(
                id=assert_single(tokenizer.convert_tokens_to_ids(CLASS_A_TOKEN)),
                text=CLASS_A_TOKEN,
            ),
            b=Token(
                id=assert_single(tokenizer.convert_tokens_to_ids(CLASS_B_TOKEN)),
                text=CLASS_B_TOKEN,
            ),
        ),
    )

    return tokens
