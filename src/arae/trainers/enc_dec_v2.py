import itertools
from dataclasses import dataclass
from enum import Enum, auto
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Tuple,
    TypedDict,
    Union,
)

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F
from dacite import from_dict
from jaxtyping import Float, Int64
from torch.utils.data import DataLoader
from tqdm.auto import tqdm as tq
from tqdm.auto import trange
from transformers import (
    PreTrainedModel,
    Seq2SeqTrainer,
    T5ForConditionalGeneration,
    Trainer,
)
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    Seq2SeqLMOutput,
    SequenceClassifierOutput,
)

import arae.losses as L
from arae.models import T5ForUnsupervisedTranslation
from arae.tokens import ARAETokens
from arae.trainers.common import Mode, TokenIds, exclude_none, sum_default


class EncoderDecoderLMForUnsupervisedTranslationTrainer(Trainer):
    """Trainer for encoder-decoder language models to perform unsupervised
    translation"""

    def __init__(
        self,
        *,
        label0_token_id: int,
        label1_token_id: int,
        cls_token_id: int,
        use_decoder_as_classifier: bool | None = True,
        model_has_cls_module: bool = False,
        model_has_cls_head: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.label0_token_id = label0_token_id
        self.label1_token_id = label1_token_id
        self.cls_token_id = cls_token_id
        self.use_decoder_as_classifier = use_decoder_as_classifier
        self.model_has_cls_module = model_has_cls_module
        self.model_has_cls_head = model_has_cls_head

    def compute_loss(
        self,
        model: (
            T5ForUnsupervisedTranslation | T5ForConditionalGeneration | PreTrainedModel
        ),
        inputs: Dict[str, Any],
        return_outputs: bool = False,
    ):
        assert self.tokenizer is not None
        assert self.tokenizer.pad_token_id is not None
        assert self.tokenizer.unk_token_id is not None

        # Enable dropout only in encoder, forming a denoising autoencoder
        # model.get_encoder().train()
        # model.get_decoder().eval()

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

        # print(autoencoding_input_ids)
        # print(autoencoding_decoder_input_ids)
        # print(input_ids)

        autoencoding_loss = autoencoding_outputs.loss
        assert autoencoding_loss is not None

        # Adversarial training (for matching of representations across languages)
        # * This implementes an NS-GAN loss, i.e. D is trained as normal to accurately
        # predict the binary label of the data, by minimizing the cross entropy, while G
        # is trained to fool D into predicting the other label (-> NS-GAN), instead of
        # maximizing the loss of D (-> Standard-GAN)

        batch_size = input_ids.shape[0]

        if self.use_decoder_as_classifier:
            # For T5, the decoder start token is the pad token
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
                attention_mask=autoencoding_attention_mask,
                decoder_input_ids=classifier_input_ids,
                decoder_attention_mask=classifier_attention_mask,
            )

            classifier_logits = classifier_outputs.logits[
                :, 0, [self.label0_token_id, self.label1_token_id]
            ]  # Logits of the label tokens, at the first decoder token position

            classifier_loss = F.cross_entropy(classifier_logits, classifier_labels)

            # Gradient penality
            gradient_penalty = torch.tensor(0.0, dtype=torch.float32)
            # if not do_encoder_step:
            if False:
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

                # print(classifier_labels)

                gradient_penalty = torch.mean(
                    torch.square(classifier_grads.norm(p=2, dim=(1, 2)))
                )

            if do_encoder_step:
                # Turn gradients back on for next training step
                model.get_decoder().requires_grad_(True)
        else:
            embedding_layer = model.shared
            cls_token_id_tensor = torch.LongTensor([[self.cls_token_id]] * batch_size)
            cls_token_id_tensor = cls_token_id_tensor.to(model.device)
            cls_token_embedding_vector = embedding_layer(cls_token_id_tensor)

            encoder_last_hidden_state = autoencoding_outputs.encoder_last_hidden_state
            assert encoder_last_hidden_state is not None

            if self.model_has_cls_module:
                classifier_module = model.classifier
            else:
                classifier_module = model.get_encoder()

            num_classifier_steps = 1
            do_encoder_step = self.state.global_step % (num_classifier_steps + 1) == 0
            if do_encoder_step:
                # Invert the label, so that G is trained to fool D into predicting the wrong
                # label of the latents
                classifier_labels = 1 - labels

                # Turn off classifier gradients, to not update D into helping G
                classifier_module.requires_grad_(False)
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
                # encoder_last_hidden_state.requires_grad_(True)

            # Instance Noise
            # noise_interp = min(1, self.state.global_step / 500)
            # encoder_last_hidden_state = noise_interp * encoder_last_hidden_state + (
            #    1 - noise_interp
            # ) * torch.randn_like(encoder_last_hidden_state)

            autoencoder_output_embeddings = encoder_last_hidden_state[:, 1:]
            classifier_attention_mask = autoencoding_attention_mask

            classifier_input_embeddings = torch.cat(
                [cls_token_embedding_vector, autoencoder_output_embeddings], dim=1
            )

            classifier_outputs: BaseModelOutputWithPastAndCrossAttentions = (
                classifier_module(
                    inputs_embeds=classifier_input_embeddings,
                    attention_mask=autoencoding_attention_mask,
                )
            )

            if self.model_has_cls_head:
                classifier_logits = model.cls_head(
                    classifier_outputs.last_hidden_state[:, 0]
                )[:, 0]
            else:
                classifier_logits = model.lm_head(
                    classifier_outputs.last_hidden_state[:, 0]
                )

                classifier_logits = classifier_logits[:, self.cls_token_id]

                """
                classifier_logits = classifier_logits[
                    :, [self.label0_token_id, self.label1_token_id]
                ]  # Logits of the label tokens

                classifier_loss = F.cross_entropy(classifier_logits, classifier_labels)
                """

            classifier_loss = F.binary_cross_entropy_with_logits(
                classifier_logits, classifier_labels.float()
            )

            # Gradient penality
            gradient_penalty = torch.tensor(0.0, dtype=torch.float32)

            if do_encoder_step:
                # Turn gradients back on for next training step
                classifier_module.requires_grad_(True)

        # Back-translation (for representation consistency across encoding-decoding
        # cycles)
        # TODO: Implement this

        # do_log_step = self.state.global_step % self.state.logging_steps == 0
        do_log_step = True
        if do_log_step:
            classifier_loss_name = "Encoder" if do_encoder_step else "Classifier"
            classifier_loss_name = classifier_loss_name + " Loss"

            self.log_metrics(
                split="train",
                metrics={
                    "Autoencoder Loss": f"{autoencoding_loss.item():.2e}",
                    classifier_loss_name: f"{classifier_loss.item():.2e}",
                    # "Instance Noise": 1 - noise_interp,
                    "Gradient Penalty": f"{gradient_penalty.item():.2e}",
                },
            )

        loss = autoencoding_loss + classifier_loss * 1  # + gradient_penalty * 10
        return loss


def filter_weight_decay(named_params: Iterable[Tuple[str, nn.Parameter]]):
    with_weight_decay: List[Tuple[str, nn.Parameter]] = []
    without_weight_decay: List[Tuple[str, nn.Parameter]] = []

    for n, p in named_params:
        apply_weight_decay = True

        # No weight decay for all bias terms
        if n.endswith(".bias"):
            apply_weight_decay = False

        # No weight decay for all layer norms
        if "layer_norm" in n:
            apply_weight_decay = False

        if apply_weight_decay:
            with_weight_decay.append(p)
        else:
            without_weight_decay.append(p)

    return with_weight_decay, without_weight_decay


def make_param_groups(
    named_params: Iterable[Tuple[str, nn.Parameter]],
    learning_rate: float,
    weight_decay: float,
):
    with_weight_decay, without_weight_decay = filter_weight_decay(named_params)

    base_group = {
        "params": with_weight_decay,
        "lr": learning_rate,
        "weight_decay": weight_decay,
    }

    no_wd_group = {
        "params": without_weight_decay,
        "lr": learning_rate,
        "weight_decay": 0.0,
    }

    return [base_group, no_wd_group]


def utl_autoencode(
    model: T5ForUnsupervisedTranslation,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    pad_token_id: int,
    label0_token_id: int,
    label1_token_id: int,
    mask_token_id: int,
):
    # Generate input id mask
    noise_mask = torch.rand_like(input_ids, dtype=torch.float32) < -0.5
    pad_mask = input_ids != pad_token_id

    # Mask input ids
    masked_input_ids = torch.where(
        torch.logical_and(noise_mask, pad_mask),
        mask_token_id,
        input_ids,
    )

    # Create autoencoder inputs
    autoencoder_input_ids = torch.roll(masked_input_ids.clone(), 1, dims=1)
    autoencoder_input_ids[:, 0] = torch.where(
        labels == 0, label0_token_id, label1_token_id
    )

    autoencoder_decoder_input_ids = torch.roll(input_ids.clone(), 1, dims=1)
    autoencoder_decoder_input_ids[:, 0] = torch.where(
        labels == 0, label0_token_id, label1_token_id
    )

    autoencoder_attention_mask = torch.roll(attention_mask.clone(), 1, dims=1)
    autoencoder_attention_mask[:, 0] = 1

    # Ignore decoder labels at padding tokens
    autoencoder_labels = torch.where(
        autoencoder_input_ids == pad_token_id,
        -100,
        input_ids.clone(),
    )

    autoencoder_output: Seq2SeqLMOutput = model(
        input_ids=autoencoder_input_ids,
        attention_mask=autoencoder_attention_mask,
        decoder_input_ids=autoencoder_decoder_input_ids,
        decoder_attention_mask=autoencoder_attention_mask,
        labels=autoencoder_labels,
    )

    return autoencoder_output, autoencoder_attention_mask


def utl_classifier(
    model: T5ForUnsupervisedTranslation,
    encoder_embeds: torch.Tensor,
    encoder_attention_mask: torch.Tensor,
    labels: torch.Tensor,
):
    classifier_outputs = model.classifier(
        inputs_embeds=encoder_embeds, attention_mask=encoder_attention_mask
    )
    classifier_pooled = classifier_outputs.last_hidden_state
    classifier_pooled = torch.mean(classifier_pooled, dim=1)
    classifier_logits = model.cls_head(classifier_pooled)[:, 0]

    classifier_loss = F.binary_cross_entropy_with_logits(
        classifier_logits, labels.float()
    )

    return SequenceClassifierOutput(loss=classifier_loss, logits=classifier_logits)  # type: ignore


def utl_model_training_loop(
    model: T5ForUnsupervisedTranslation,
    tokenizer,
    train_dataloader: DataLoader,
    num_steps: int,
    learning_rate: float,
    weight_decay: float,
    pad_token_id: int,
    label0_token_id: int,
    label1_token_id: int,
    mask_token_id: int,
    cls_token_id: int,
    save_dir: str,
):
    OptimizerClass = bnb.optim.adamw.AdamW8bit
    device = model.device

    autoencoder_params = itertools.chain(
        model.encoder.named_parameters(),
        model.decoder.named_parameters(),
        model.lm_head.named_parameters(),
        model.shared.named_parameters(),
    )

    encoder_params = itertools.chain(
        model.encoder.named_parameters(),
        model.shared.named_parameters(),
    )

    classifier_params = itertools.chain(
        model.classifier.named_parameters(),
        model.shared.named_parameters(),
    )

    autoencoder_optimizer = OptimizerClass(
        make_param_groups(autoencoder_params, learning_rate, weight_decay)
    )

    encoder_optimizer = OptimizerClass(
        make_param_groups(encoder_params, learning_rate, weight_decay)
    )

    classifier_optimizer = OptimizerClass(
        make_param_groups(classifier_params, learning_rate, weight_decay)
    )

    model.train()

    pbar = tq(enumerate(train_dataloader))
    for step, batch in pbar:
        input_ids: torch.Tensor = batch["input_ids"].to(device)
        attention_mask: torch.Tensor = batch["attention_mask"].to(device)
        labels: torch.Tensor = batch["labels"].to(device)

        # print(input_ids.shape)

        # Autoencoder Step
        autoencoder_output, _ = utl_autoencode(
            model,
            input_ids,
            attention_mask,
            labels,
            pad_token_id,
            label0_token_id,
            label1_token_id,
            mask_token_id,
        )

        autoencoder1_loss = autoencoder_output.loss
        assert autoencoder1_loss is not None

        autoencoder_optimizer.zero_grad(set_to_none=True)
        autoencoder1_loss.backward()
        autoencoder_optimizer.step()

        # Classifier Step
        # with torch.no_grad():
        autoencoder_output, encoder_attention_mask = utl_autoencode(
            model,
            input_ids,
            attention_mask,
            labels,
            pad_token_id,
            label0_token_id,
            label1_token_id,
            mask_token_id,
        )

        assert autoencoder_output.encoder_last_hidden_state is not None

        classifier_output = utl_classifier(
            model,
            autoencoder_output.encoder_last_hidden_state,
            encoder_attention_mask,
            torch.zeros_like(labels),
        )

        classifier_loss = classifier_output.loss
        assert classifier_loss is not None

        classifier_optimizer.zero_grad(set_to_none=True)
        classifier_loss.backward()
        print(
            torch.nn.utils.clip_grad.clip_grad_norm_(
                classifier_optimizer.param_groups[0]["params"], max_norm=10000
            )
        )
        classifier_optimizer.step()

        # Autoencoder Step
        autoencoder_output, _ = utl_autoencode(
            model,
            input_ids,
            attention_mask,
            labels,
            pad_token_id,
            label0_token_id,
            label1_token_id,
            mask_token_id,
        )

        autoencoder2_loss = autoencoder_output.loss
        assert autoencoder2_loss is not None

        autoencoder_optimizer.zero_grad(set_to_none=True)
        autoencoder2_loss.backward()
        autoencoder_optimizer.step()

        # Adversarial Encoder Step
        """
        autoencoder_output, encoder_attention_mask = utl_autoencode(
            model,
            input_ids,
            attention_mask,
            labels,
            pad_token_id,
            label0_token_id,
            label1_token_id,
            mask_token_id,
        )

        assert autoencoder_output.encoder_last_hidden_state is not None

        model.classifier.requires_grad_(False)
        model.cls_head.requires_grad_(False)

        classifier_output = utl_classifier(
            model,
            autoencoder_output.encoder_last_hidden_state,
            encoder_attention_mask,
            1 - labels,
        )

        model.classifier.requires_grad_(True)
        model.cls_head.requires_grad_(True)

        adversarial_loss = classifier_output.loss
        assert adversarial_loss is not None

        encoder_optimizer.zero_grad(set_to_none=True)
        adversarial_loss.backward()
        encoder_optimizer.step()
        """
        adversarial_loss = 0.0

        if (step + 1) % 100 == 0:
            model.save_pretrained(save_dir)
            tokenizer.save_pretrained(save_dir)

        pbar.set_postfix_str(
            f"AE1: {autoencoder1_loss:.2e} AE2: {autoencoder2_loss:.2e} CLS: {classifier_loss:.2e} ADV: {adversarial_loss:.2e}"
        )
