from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Mapping, NamedTuple, Optional, TypedDict, Union

import torch
import torch.nn.functional as F
from dacite import from_dict
from jaxtyping import Float, Int64
from transformers import PreTrainedModel, Trainer
from transformers.modeling_outputs import Seq2SeqLMOutput

import arae.losses as L
from arae.tokens import ARAETokens
from arae.trainers.common import Mode, TokenIds, exclude_none, sum_default


@dataclass
class EncDecInputs:
    enc: TokenIds
    dec: TokenIds
    cls: TokenIds
    adv: TokenIds


def encode(
    model: PreTrainedModel,
    input_ids: Int64[torch.Tensor, "batch input_sequence"],
    attention_mask: Int64[torch.Tensor, "batch input_sequence"],
    num_prefix_tokens: int,
):
    output = model(input_ids=input_ids, attention_mask=attention_mask)

    embeddings = output.encoder_last_hidden_state[:, num_prefix_tokens:]
    return embeddings


def decode(
    model: PreTrainedModel,
    input_ids: Int64[torch.Tensor, "batch input_sequence"],
    attention_mask: Int64[torch.Tensor, "batch input_sequence"],
    labels: Optional[Int64[torch.Tensor, "batch target_sequence"]],
    embeddings: Float[torch.Tensor, "batch {input_sequence - num_prefix_tokens}"],
    num_prefix_tokens: int,
) -> Seq2SeqLMOutput:
    input_embeddings = model.get_input_embeddings()(input_ids)
    input_embeddings[:, num_prefix_tokens:] = embeddings

    output = model(
        inputs_embeds=input_embeddings, attention_mask=attention_mask, labels=labels
    )
    return output


class EncDecTrainer(Trainer):
    """Trainer for encoder-decoder models to perform unsupervised translation"""

    def __init__(self, *, tokens: ARAETokens, mode: Mode = Mode.SIM, **kwargs):
        super().__init__(**kwargs)

        self.tokens = tokens
        self.mode = mode

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ):
        inputs = from_dict(data_class=EncDecInputs, data=inputs)  # type: ignore
        assert isinstance(inputs, EncDecInputs)

        # Autoencoder

        ## Encode
        encoding_embeddings = encode(
            model=model,
            input_ids=inputs.enc.input_ids,
            attention_mask=inputs.enc.attention_mask,
            num_prefix_tokens=2,
        )

        ## Decode
        decoding_output = decode(
            model=model,
            input_ids=inputs.dec.input_ids,
            attention_mask=inputs.dec.attention_mask,
            labels=inputs.dec.labels,
            embeddings=encoding_embeddings,
            num_prefix_tokens=2,
        )

        ae_loss = decoding_output.loss

        # Adversarial classification

        do_adv_step = None
        do_cls_step = None

        if self.mode == Mode.SIM:
            do_adv_step = True
            do_cls_step = True
        elif self.mode == Mode.ALT:
            do_adv_step = self.state.global_step % 2 == 0
            do_cls_step = self.state.global_step % 2 == 1

        assert do_adv_step is not None
        assert do_cls_step is not None

        adv_loss = None
        if do_adv_step:
            # Adversarial loss
            model.requires_grad_(False)
            adv_classification_output = decode(
                model=model,
                input_ids=inputs.adv.input_ids,
                attention_mask=inputs.adv.attention_mask,
                labels=inputs.cls.labels,
                embeddings=encoding_embeddings,
                num_prefix_tokens=1,
            )
            adv_loss = adv_classification_output.loss
            model.requires_grad_(True)

        cls_loss = None
        if do_cls_step:
            # Classification loss
            classification_output = decode(
                model=model,
                input_ids=inputs.cls.input_ids,
                attention_mask=inputs.cls.attention_mask,
                labels=inputs.cls.labels,
                embeddings=encoding_embeddings.detach(),
                num_prefix_tokens=1,
            )
            cls_loss = classification_output.loss

        do_log = self.state.global_step % self.state.logging_steps == 0
        if do_log:
            self.log_metrics(
                split="train",
                metrics=exclude_none(
                    ae_loss=ae_loss,
                    adv_loss=adv_loss,
                    cls_loss=cls_loss,
                ),
            )

        loss = sum_default([ae_loss, adv_loss, cls_loss])
        return loss
