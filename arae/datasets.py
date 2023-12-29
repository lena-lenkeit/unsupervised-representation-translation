from typing import List, NamedTuple

import numpy as np
from jaxtyping import Int64
from tokens import ARAETokens
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer


class SingleTaskData(NamedTuple):
    input_ids: Int64[np.ndarray, "sequence"]
    attention_mask: Int64[np.ndarray, "sequence"]


class SingleInputs(NamedTuple):
    clm_inputs: SingleTaskData
    enc_inputs: SingleTaskData
    dec_inputs: SingleTaskData
    cls_inputs: SingleTaskData
    cls_id: Int64[np.ndarray, ""]
    cls_token_id: Int64[np.ndarray, ""]


# Define a custom dataset
class ARAEDataset(Dataset):
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        file_A: str,
        file_B: str,
        max_length: int,
        tokens: ARAETokens,
    ):
        with open(file_A, "r") as f:
            self.sentences_A = [line.strip() for line in f.readlines()]

        with open(file_B, "r") as f:
            self.sentences_B = [line.strip() for line in f.readlines()]

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
        cls_token_id = self.tokens.label.a.id if label == 0 else self.tokens.label.b.id

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
        ):
            special_length = len(prefix_token_ids) + len(postfix_token_ids)
            text_max_length = max_length - special_length

            text_token_ids = text_token_ids[:text_max_length]
            sequence_token_ids = prefix_token_ids + text_token_ids + postfix_token_ids
            assert len(sequence_token_ids) <= max_length, "Token sequence too long!"

            token_ids, attention_mask = pad_tokens(
                sequence_token_ids, max_length, pad_token_id
            )

            assert len(token_ids) == max_length, "Length mismatch"
            assert len(attention_mask) == max_length, "Length mismatch"

            token_ids = np.asarray(token_ids, dtype=np.int64)
            attention_mask = np.asarray(attention_mask, dtype=np.int64)

            return SingleTaskData(input_ids=token_ids, attention_mask=attention_mask)

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
        cls_inputs = prepare_model_inputs(
            [
                self.tokens.task.classification.id,
                self.tokens.placeholder.embedding.id,
                self.tokens.placeholder.label.id,
            ],
            [],
            [],
            self.max_length,
            pad_token_id,
        )

        return SingleInputs(
            clm_inputs=clm_inputs,
            enc_inputs=enc_inputs,
            dec_inputs=dec_inputs,
            cls_inputs=cls_inputs,
            cls_id=np.asarray(label, dtype=np.int64),
            cls_token_id=np.asarray(cls_token_id, dtype=np.int64),
        )
