from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Mapping, NamedTuple, Optional, TypedDict, Union

import torch
import torch.nn.functional as F
from dacite import from_dict
from jaxtyping import Float, Int64
from transformers import PreTrainedModel, Trainer
from transformers.modeling_outputs import CausalLMOutput, CausalLMOutputWithPast

import arae.losses as L
from arae.tokens import ARAETokens
from arae.utils import gather_from_tokens, scatter_to_tokens


def cls_loss_fn(
    model: PreTrainedModel,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    cls_ids: torch.Tensor,
    embeddings: Float[torch.Tensor, "batch features"],
    tokens: ARAETokens,
) -> Float[torch.Tensor, ""]:
    input_embeddings = model.get_input_embeddings()(input_ids)
    input_embeddings = scatter_to_tokens(
        key=input_ids,
        source=embeddings,
        values=input_embeddings,
        query=tokens.placeholder.embedding.id,
        allow_multiple=True,
    )

    outputs = model(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
    )

    assert isinstance(outputs, (CausalLMOutput, CausalLMOutputWithPast))

    logits = gather_from_tokens(
        key=input_ids,
        values=outputs.logits,
        query=tokens.placeholder.label.id,
    )
    logits = logits[:, [tokens.label.a.id, tokens.label.b.id]]
    loss = F.cross_entropy(logits, cls_ids)

    return loss


@dataclass
class TokenIds:
    input_ids: Int64[torch.Tensor, "batch input_sequence"]
    attention_mask: Int64[torch.Tensor, "batch input_sequence"]
    labels: Optional[Int64[torch.Tensor, "batch target_sequence"]] = None


@dataclass
class ARAEInputs:
    clm: TokenIds
    enc: TokenIds
    dec: TokenIds
    cls: TokenIds
    cls_id: Int64[torch.Tensor, "batch"]
    cls_token_id: Int64[torch.Tensor, "batch"]


class Mode(str, Enum):
    SIM = "SIM"
    ALT = "ALT"


def sum_default(values: list, default: float = 0.0):
    return sum([value if value is not None else default for value in values])


def exclude_none(**kwargs):
    return {k: v for k, v in kwargs.items() if v is not None}


class ARAETrainer(Trainer):
    def __init__(
        self,
        *,
        tokens: ARAETokens,
        mode: Mode = Mode.SIM,
        enable_clm: bool = True,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.tokens = tokens
        self.mode = mode
        self.enable_clm = enable_clm

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ):
        inputs = from_dict(data_class=ARAEInputs, data=inputs)  # type: ignore
        assert isinstance(inputs, ARAEInputs)

        clm_loss = None
        if self.enable_clm:
            # Causal language modelling loss
            clm_outputs = model(
                input_ids=inputs.clm.input_ids,
                attention_mask=inputs.clm.attention_mask,
            )
            assert isinstance(clm_outputs, (CausalLMOutput, CausalLMOutputWithPast))

            clm_loss = L.clm_loss_fn(
                logits=clm_outputs.logits,
                labels=inputs.clm.input_ids,
                mask=inputs.clm.attention_mask,
            )

        # Autoencoder loss

        ## Encode
        enc_outputs = model(
            input_ids=inputs.enc.input_ids,
            attention_mask=inputs.enc.attention_mask,
            output_hidden_states=True,
        )
        assert isinstance(enc_outputs, (CausalLMOutput, CausalLMOutputWithPast))
        assert isinstance(enc_outputs.hidden_states, tuple)

        enc_embeddings = gather_from_tokens(
            key=inputs.enc.input_ids,
            values=enc_outputs.hidden_states[-1],
            query=self.tokens.placeholder.embedding.id,
        )

        ## Decode

        ### Construct decoding input
        dec_input_embeddings = model.get_input_embeddings()(inputs.dec.input_ids)
        dec_input_embeddings = scatter_to_tokens(
            key=inputs.dec.input_ids,
            source=enc_embeddings,
            values=dec_input_embeddings,
            query=self.tokens.placeholder.embedding.id,
        )

        ## Loss
        dec_outputs = model(
            inputs_embeds=dec_input_embeddings,
            attention_mask=inputs.dec.attention_mask,
        )
        assert isinstance(dec_outputs, (CausalLMOutput, CausalLMOutputWithPast))

        ae_loss = L.clm_loss_fn(
            logits=dec_outputs.logits,
            labels=inputs.dec.input_ids,
            mask=inputs.dec.attention_mask,
        )

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
            adv_loss = cls_loss_fn(
                model=model,
                input_ids=inputs.cls.input_ids,
                attention_mask=inputs.cls.attention_mask,
                cls_ids=1 - inputs.cls_id,
                embeddings=enc_embeddings,
                tokens=self.tokens,
            )
            model.requires_grad_(True)

        cls_loss = None
        if do_cls_step:
            # Classification loss
            cls_loss = cls_loss_fn(
                model=model,
                input_ids=inputs.cls.input_ids,
                attention_mask=inputs.cls.attention_mask,
                cls_ids=inputs.cls_id,
                embeddings=enc_embeddings.detach(),
                tokens=self.tokens,
            )

        do_log = self.state.global_step % self.state.logging_steps == 0
        if do_log:
            self.log_metrics(
                split="train",
                metrics=exclude_none(
                    clm_loss=clm_loss,
                    ae_loss=ae_loss,
                    adv_loss=adv_loss,
                    cls_loss=cls_loss,
                ),
            )

        loss = sum_default([clm_loss, ae_loss, adv_loss, cls_loss])
        return loss
