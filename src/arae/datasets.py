from dataclasses import asdict, dataclass
from typing import List, Optional, Union

import numpy as np
from jaxtyping import Int64
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, PreTrainedTokenizerFast

from arae.tokens import ARAETokens


# Helper function to truncate token IDs and create attention mask
def pad_tokens(token_ids: List[int], max_length: int, pad_token_id: int):
    attention_mask = [1] * len(token_ids)

    # Pad token_ids and attention mask to max_length
    padding_length = max_length - len(token_ids)
    token_ids = token_ids + ([pad_token_id] * padding_length)
    attention_mask = attention_mask + ([0] * padding_length)

    return token_ids, attention_mask


# Function to prepare model inputs from token ids directly
def prepare_model_inputs(
    prefix_token_ids: List[int],
    text_token_ids: List[int],
    postfix_token_ids: List[int],
    max_length: int,
    pad_token_id: int,
    pad: bool = True,
):
    special_length = len(prefix_token_ids) + len(postfix_token_ids)
    text_max_length = max_length - special_length

    text_token_ids = text_token_ids[:text_max_length]
    sequence_token_ids = prefix_token_ids + text_token_ids + postfix_token_ids
    assert len(sequence_token_ids) <= max_length, "Token sequence too long!"

    if pad:
        token_ids, attention_mask = pad_tokens(
            sequence_token_ids, max_length, pad_token_id
        )

        assert len(token_ids) == max_length, "Length mismatch"
        assert len(attention_mask) == max_length, "Length mismatch"

        token_ids = np.asarray(token_ids, dtype=np.int64)
        attention_mask = np.asarray(attention_mask, dtype=np.int64)
    else:
        token_ids = np.asarray(sequence_token_ids, dtype=np.int64)
        attention_mask = np.ones_like(token_ids)

    return TokenIds(input_ids=token_ids, attention_mask=attention_mask)


@dataclass
class TokenIds:
    input_ids: Int64[np.ndarray, "input_sequence"]
    attention_mask: Int64[np.ndarray, "input_sequence"]
    labels: Optional[Int64[np.ndarray, "target_sequence"]] = None


@dataclass
class ARAEInputs:
    clm: TokenIds
    enc: TokenIds
    dec: TokenIds
    cls: TokenIds
    cls_id: Int64[np.ndarray, ""]
    cls_token_id: Int64[np.ndarray, ""]


@dataclass
class EncDecInputs:
    enc: TokenIds
    dec: TokenIds
    cls: TokenIds
    adv: TokenIds


# Define a custom dataset
class ARAEDataset(Dataset):
    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        file_A: str,
        file_B: str,
        max_length: int,
        tokens: ARAETokens,
        num_cls_emb_tokens: int = 1,
    ):
        with open(file_A, "r") as f:
            self.sentences_A = [
                line.split(maxsplit=1)[-1].strip() for line in f.readlines()
            ]

        with open(file_B, "r") as f:
            self.sentences_B = [
                line.split(maxsplit=1)[-1].strip() for line in f.readlines()
            ]

        self.dataset = self.sentences_A + self.sentences_B
        self.labels = [0] * len(self.sentences_A) + [1] * len(self.sentences_B)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokens = tokens
        self.num_cls_emb_tokens = num_cls_emb_tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        text = self.dataset[idx]
        label = self.labels[idx]
        cls_token_id = self.tokens.label.a.id if label == 0 else self.tokens.label.b.id

        # Token IDs for special tokens, placeholders, and padding
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        # Tokenize the text into token IDs
        text_token_ids = self.tokenizer(
            text, add_special_tokens=False
        ).input_ids  # Don't add special tokens

        # Tokenize CLM task
        clm_inputs = prepare_model_inputs(
            [self.tokens.task.modeling.id],
            text_token_ids,
            [],
            self.max_length,
            pad_token_id,
        )

        # Tokenize encoding task
        enc_inputs = prepare_model_inputs(
            [self.tokens.task.encoding.id, cls_token_id],
            text_token_ids,
            [self.tokens.placeholder.embedding.id],
            self.max_length,
            pad_token_id,
        )

        # Tokenize decoding task
        dec_inputs = prepare_model_inputs(
            [
                self.tokens.task.decoding.id,
                cls_token_id,
                self.tokens.placeholder.embedding.id,
            ],
            text_token_ids,
            [],
            self.max_length,
            pad_token_id,
        )

        # Tokenize classification task (no need to handle text, just tokens)
        num_cls_emb_tokens = (
            self.max_length if self.num_cls_emb_tokens < 0 else self.num_cls_emb_tokens
        )
        cls_inputs = prepare_model_inputs(
            [self.tokens.task.classification.id],
            [self.tokens.placeholder.embedding.id] * num_cls_emb_tokens,
            [self.tokens.placeholder.label.id],
            self.max_length,
            pad_token_id,
        )

        inputs = ARAEInputs(
            clm=clm_inputs,
            enc=enc_inputs,
            dec=dec_inputs,
            cls=cls_inputs,
            cls_id=np.asarray(label, dtype=np.int64),
            cls_token_id=np.asarray(cls_token_id, dtype=np.int64),
        )

        return asdict(inputs)


class EncDecDataset(Dataset):
    """Dataset for encoder-decoder language models"""

    def __init__(
        self,
        tokenizer: Union[PreTrainedTokenizer, PreTrainedTokenizerFast],
        file_A: str,
        file_B: str,
        max_length: int,
        tokens: ARAETokens,
    ):
        with open(file_A, "r") as f:
            self.sentences_A = [
                line.split(maxsplit=1)[-1].strip() for line in f.readlines()
            ]

        with open(file_B, "r") as f:
            self.sentences_B = [
                line.split(maxsplit=1)[-1].strip() for line in f.readlines()
            ]

        self.dataset = self.sentences_A + self.sentences_B
        self.labels = [0] * len(self.sentences_A) + [1] * len(self.sentences_B)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokens = tokens

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        text = self.dataset[idx]
        label = self.labels[idx]
        cls_token = self.tokens.label.a if label == 0 else self.tokens.label.b
        adv_token = self.tokens.label.b if label == 0 else self.tokens.label.a

        # Get padding token id
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        # Tokenize the text into token IDs
        text_token_ids = self.tokenizer(
            text, add_special_tokens=False
        ).input_ids  # Don't add special tokens

        # Notation:
        # * [Token] - A token (string of text with associated id) and an associated
        #   embedding.
        # * {Vector} - A vector of size equal to a token embedding. These can't be
        #   tokenized directly, instead requiring replacement of some placeholder token
        #   with the intended values at a later stage.

        # Tokenize encoding task:
        # * To Encoder: [Encoding Task Token] [Class Token] [Data Token 1] ... [Data
        #   Token N]
        # * To Decoder: None
        # * Note: The output here are the last hidden values of the encoder at the data
        #   token positions, hence the decoder isn't used at all.
        enc_inputs = prepare_model_inputs(
            [self.tokens.task.encoding.id, cls_token.id],
            text_token_ids,
            [],
            self.max_length,
            pad_token_id,
        )

        # Tokenize decoding task:
        # * To Encoder: [Decoding Task Token] [Class Token] {Encoder Outputs 1} ...
        #   {Encoder Outputs N}
        # * To Decoder: [Data Token 1] ... [Data Token N]
        # * Note: Padding token as placeholder for the encoder outputs. Since the
        #   padding token is explicitly included in the inputs, up to the length of the
        #   encoded data, the attention mask will also include these tokens.
        dec_inputs = prepare_model_inputs(
            [self.tokens.task.decoding.id, cls_token.id],
            [pad_token_id] * len(text_token_ids),
            [],
            self.max_length,
            pad_token_id,
        )
        dec_inputs.labels = prepare_model_inputs(
            [],
            text_token_ids,
            [],
            self.max_length,
            pad_token_id,
        ).input_ids

        # Tokenize classification task:
        # * To Encoder: [Classification Task Token] {Encoder Outputs 1} ... {Encoder
        #   Outputs N}
        # * To Decoder: [Class Token]
        cls_inputs = prepare_model_inputs(
            [self.tokens.task.classification.id],
            [pad_token_id] * len(text_token_ids),
            [],
            self.max_length,
            pad_token_id,
        )
        cls_inputs.labels = prepare_model_inputs(
            [],
            [cls_token.id],
            [],
            self.max_length,
            pad_token_id,
        ).input_ids

        # Tokenize adversarial classification task:
        # * To Encoder: [Classification Task Token] {Encoder Outputs 1} ... {Encoder
        #   Outputs N}
        # * To Decoder: [Adversarial Class Token]
        adv_inputs = prepare_model_inputs(
            [self.tokens.task.classification.id],
            [pad_token_id] * len(text_token_ids),
            [],
            self.max_length,
            pad_token_id,
        )
        adv_inputs.labels = prepare_model_inputs(
            [],
            [adv_token.id],
            [],
            self.max_length,
            pad_token_id,
        ).input_ids

        inputs = EncDecInputs(
            enc=enc_inputs, dec=dec_inputs, cls=cls_inputs, adv=adv_inputs
        )

        return asdict(inputs)
