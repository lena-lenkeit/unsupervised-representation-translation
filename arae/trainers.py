from typing import NamedTuple, TypedDict

import torch
import torch.nn.functional as F
from jaxtyping import Float, Int64
from tokens import ARAETokens
from transformers import PreTrainedModel, Trainer
from transformers.modeling_outputs import CausalLMOutputWithPast
from utils import gather_from_tokens, scatter_to_tokens

import arae.losses as L


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
    )

    outputs = model(
        inputs_embeds=input_embeddings,
        attention_mask=attention_mask,
    )

    logits = gather_from_tokens(
        key=input_ids,
        values=outputs["logits"],
        query=tokens.placeholder.label.id,
    )
    logits = logits[:, [tokens.label.a.id, tokens.label.b.id]]
    loss = F.cross_entropy(logits, cls_ids)

    return loss


class LMInputs(NamedTuple):
    input_ids: Int64[torch.Tensor, "batch sequence"]
    attention_mask: Int64[torch.Tensor, "batch sequence"]


class BatchInputs(NamedTuple):
    clm: LMInputs
    enc: LMInputs
    dec: LMInputs
    cls: LMInputs
    cls_id: Int64[torch.Tensor, "batch"]
    cls_token_id: Int64[torch.Tensor, "batch"]


class ARAETrainer(Trainer):
    def __init__(self, *, tokens: ARAETokens, **kwargs):
        self.tokens = tokens
        super().__init__(**kwargs)

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: BatchInputs,
        return_outputs: bool = False,
    ):
        # Causal language modelling loss
        clm_outputs = model(
            input_ids=inputs.clm.input_ids,
            attention_mask=inputs.clm.attention_mask,
        )
        assert isinstance(clm_outputs, CausalLMOutputWithPast)

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
        assert isinstance(enc_outputs, CausalLMOutputWithPast)
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
        assert isinstance(dec_outputs, CausalLMOutputWithPast)

        ae_loss = L.clm_loss_fn(
            logits=dec_outputs.logits,
            labels=inputs.dec.input_ids,
            mask=inputs.dec.attention_mask,
        )

        # Matching loss
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

        # Classification loss
        cls_loss = cls_loss_fn(
            model=model,
            input_ids=inputs.cls.input_ids,
            attention_mask=inputs.cls.attention_mask,
            cls_ids=inputs.cls_id,
            embeddings=enc_embeddings.detach(),
            tokens=self.tokens,
        )

        self.log_metrics(
            split="train",
            metrics={
                "clm_loss": clm_loss,
                "ae_loss": ae_loss,
                "adv_loss": adv_loss,
                "cls_loss": cls_loss,
            },
        )

        return clm_loss + ae_loss + adv_loss + cls_loss
