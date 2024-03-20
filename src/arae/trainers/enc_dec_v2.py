from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Dict, Mapping, NamedTuple, Optional, TypedDict, Union

import torch
import torch.nn.functional as F
from dacite import from_dict
from jaxtyping import Float, Int64
from transformers import (
    PreTrainedModel,
    Seq2SeqTrainer,
    T5ForConditionalGeneration,
    Trainer,
)
from transformers.modeling_outputs import Seq2SeqLMOutput

import arae.losses as L
from arae.tokens import ARAETokens
from arae.trainers.common import Mode, TokenIds, exclude_none, sum_default


class EncoderDecoderLMForUnsupervisedTranslationTrainer(Trainer):
    """Trainer for encoder-decoder language models to perform unsupervised
    translation"""

    def __init__(self, *, label0_token_id: int, label1_token_id: int, **kwargs):
        super().__init__(**kwargs)

        self.label0_token_id = label0_token_id
        self.label1_token_id = label1_token_id

    def compute_loss(
        self,
        model: T5ForConditionalGeneration | PreTrainedModel,
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ):
        # Retrieve base inputs from batch
        # * (encoder_)input_ids should be [LABEL_TOKEN] [TEXT_TOKEN_1] ...
        # * decoder_input_ids should be [LABEL_TOKEN] [TEXT_TOKEN_1] ...
        # * labels should be the label ids
        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]
        # decoder_input_ids: torch.Tensor = inputs["decoder_input_ids"]
        # decoder_attention_mask: torch.Tensor = inputs["decoder_attention_mask"]
        labels: torch.Tensor = inputs["labels"]

        # Autoencoding (for representation learning)
        autoencoding_outputs: Seq2SeqLMOutput = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            # decoder_input_ids=input_ids,
            # decoder_attention_mask=attention_mask,
            labels=input_ids,
        )

        autoencoding_loss = autoencoding_outputs.loss
        assert autoencoding_loss is not None

        # Adversarial training (for matching of representations across languages)
        # * This implementes an NS-GAN loss, i.e. D is trained as normal to accurately
        # predict the binary label of the data, by minimizing the cross entropy, while G
        # is trained to fool D into predicting the other label (-> NS-GAN), instead of
        # maximizing the loss of D (-> Standard-GAN)

        batch_size = input_ids.shape[0]

        # For T5, the decoder start token is the pad token
        assert self.tokenizer is not None
        decoder_start_token_id = self.tokenizer.pad_token_id

        # Construct inputs for the decoder needed to perform classification. Here, this
        # is the decoder start token as input, and the loss is computed from the logits
        # of the next predicted token
        classifier_input_ids = [[decoder_start_token_id]] * batch_size
        classifier_input_ids = torch.LongTensor(classifier_input_ids).to(
            input_ids.device
        )

        classifier_attention_mask = [[1]] * batch_size
        classifier_attention_mask = torch.LongTensor(classifier_attention_mask).to(
            input_ids.device
        )

        encoder_last_hidden_state = autoencoding_outputs.encoder_last_hidden_state
        assert encoder_last_hidden_state is not None

        do_encoder_step = self.state.global_step % 2 == 0
        if do_encoder_step:
            # Invert the label, so that G is trained to fool D into predicting the wrong
            # label of the latents
            classifier_labels = 1 - labels

            # Turn off decoder / classifier gradients, to not update D into helping G
            model.get_decoder().requires_grad_(False)
        else:
            # Keep the same label, so D is trained to predict the correct label of the
            # latents
            classifier_labels = labels

            # Turn off encoder gradients (by preventing backpropagation into the
            # encoder), to not update G into helping D
            # * Setting model.get_encoder().requires_grad_(False), as above for the
            # decoder, would not work, since encoder_last_hidden_state was computed with
            # the gradients enabled, and here I reuse it for efficiency, instead of
            # recomputing it with gradients disabled
            encoder_last_hidden_state = encoder_last_hidden_state.detach()

        classifier_outputs: Seq2SeqLMOutput = model(
            encoder_outputs=(encoder_last_hidden_state,),
            decoder_input_ids=classifier_input_ids,
            decoder_attention_mask=classifier_attention_mask,
        )

        classifier_logits = classifier_outputs.logits[
            :, 0, [self.label0_token_id, self.label1_token_id]
        ]  # Logits of the label tokens, at the first decoder token position

        classifier_loss = F.cross_entropy(classifier_logits, classifier_labels)

        if do_encoder_step:
            # Turn gradients back on for next training step
            model.get_decoder().requires_grad_(True)

        # Back-translation (for representation consistency across encoding-decoding
        # cycles)
        # TODO: Implement this

        do_log_step = self.state.global_step % self.state.logging_steps == 0
        if do_log_step:
            self.log_metrics(
                split="train",
                metrics=dict(
                    autoencoding_loss=autoencoding_loss, classifier_loss=classifier_loss
                ),
            )

        loss = autoencoding_loss + classifier_loss
        return loss
