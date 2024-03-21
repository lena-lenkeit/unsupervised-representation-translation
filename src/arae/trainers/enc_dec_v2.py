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
        assert self.tokenizer is not None
        assert self.tokenizer.pad_token_id is not None
        assert self.tokenizer.unk_token_id is not None

        # Enable dropout only in encoder, forming a denoising autoencoder
        model.get_encoder().train()
        model.get_decoder().eval()

        # Retrieve base inputs from batch
        # * (encoder_)input_ids should be [TEXT_TOKEN_1] ...
        # * labels should be the language label ids {0, 1}
        # * everything else is derived automatically
        input_ids: torch.Tensor = inputs["input_ids"]
        attention_mask: torch.Tensor = inputs["attention_mask"]
        labels: torch.Tensor = inputs["labels"]

        # Autoencoding (for representation learning)

        # Construct inputs
        # * Encoder: [LABELi_TOKEN] [TEXT_TOKEN_1] ...
        # * Decoder: [LABELi_TOKEN] [TEXT_TOKEN_1] ...
        # * Labels: [TEXT_TOKEN_1] ..., masked at encoder [PAD] tokens

        noise_mask = torch.rand_like(input_ids, dtype=torch.float32) < 0.5
        pad_mask = input_ids != self.tokenizer.pad_token_id

        masked_input_ids = torch.where(
            torch.logical_and(noise_mask, pad_mask),
            self.tokenizer.unk_token_id,
            input_ids,
        )

        # Shift right to make space for the label token, and add it
        autoencoding_input_ids = torch.roll(masked_input_ids.clone(), 1, dims=1)

        autoencoding_input_ids[:, 0] = torch.where(
            labels == 0, self.label0_token_id, self.label1_token_id
        )

        # Shift right to make space for the label token, and add it
        autoencoding_decoder_input_ids = torch.roll(input_ids.clone(), 1, dims=1)

        autoencoding_decoder_input_ids[:, 0] = torch.where(
            labels == 0, self.label0_token_id, self.label1_token_id
        )

        # Shift right and set active to include label token in mask
        autoencoding_attention_mask = torch.roll(attention_mask.clone(), 1, dims=1)
        autoencoding_attention_mask[:, 0] = 1

        # Mask labels at encoder [PAD] tokens
        autoencoding_labels = torch.where(
            autoencoding_input_ids == self.tokenizer.pad_token_id,
            -100,
            input_ids.clone(),
        )

        autoencoding_outputs: Seq2SeqLMOutput = model(
            input_ids=autoencoding_input_ids,
            attention_mask=autoencoding_attention_mask,
            decoder_input_ids=autoencoding_decoder_input_ids,
            decoder_attention_mask=autoencoding_attention_mask,
            labels=autoencoding_labels,
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

        num_classifier_steps = 1
        do_encoder_step = self.state.global_step % (num_classifier_steps + 1) == 0
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
            encoder_last_hidden_state.requires_grad_(True)

        # Instance Noise
        # noise_interp = min(1, self.state.global_step / 500)
        # encoder_last_hidden_state = noise_interp * encoder_last_hidden_state + (
        #    1 - noise_interp
        # ) * torch.randn_like(encoder_last_hidden_state)

        classifier_outputs: Seq2SeqLMOutput = model(
            encoder_outputs=(encoder_last_hidden_state,),
            decoder_input_ids=classifier_input_ids,
            decoder_attention_mask=classifier_attention_mask,
        )

        classifier_logits = classifier_outputs.logits[
            :, 0, [self.label0_token_id, self.label1_token_id]
        ]  # Logits of the label tokens, at the first decoder token position

        classifier_loss = F.cross_entropy(classifier_logits, classifier_labels)

        # Gradient penality
        gradient_penalty = 0.0
        if not do_encoder_step:
            classifier_binary_prob = F.softmax(classifier_logits, dim=1)
            # classifier_binary_prob = classifier_logits

            classifier_binary_prob = torch.where(
                classifier_labels == 1,
                classifier_binary_prob[:, 1],
                classifier_binary_prob[:, 0],
            )

            # classifier_binary_prob = classifier_binary_prob[::2]
            # classifier_labels = classifier_labels[::2]

            classifier_grads = torch.autograd.grad(
                outputs=classifier_binary_prob,
                inputs=encoder_last_hidden_state,
                grad_outputs=torch.ones_like(classifier_binary_prob),
                create_graph=True,
                retain_graph=True,
                only_inputs=True,
            )[0]

            print(classifier_labels)

            gradient_penalty = torch.mean(
                torch.square(classifier_grads.norm(p=2, dim=(1, 2)))
            )

        if do_encoder_step:
            # Turn gradients back on for next training step
            model.get_decoder().requires_grad_(True)

        # Back-translation (for representation consistency across encoding-decoding
        # cycles)
        # TODO: Implement this

        do_log_step = self.state.global_step % self.state.logging_steps == 0
        if do_log_step:
            classifier_loss_name = "Encoder" if do_encoder_step else "Classifier"
            classifier_loss_name = classifier_loss_name + " Loss"

            self.log_metrics(
                split="train",
                metrics={
                    "Autoencoder Loss": autoencoding_loss,
                    classifier_loss_name: classifier_loss,
                    # "Instance Noise": 1 - noise_interp,
                    "Gradient Penalty": gradient_penalty,
                },
            )

        loss = autoencoding_loss + classifier_loss * 5 + gradient_penalty * 50
        return loss
