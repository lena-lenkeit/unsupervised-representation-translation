import copy
import itertools
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import bitsandbytes as bnb
import datasets
import numpy as np
import safetensors
import safetensors.torch
import tokenizers
import tokenizers.decoders
import tokenizers.models
import tokenizers.normalizers
import tokenizers.pre_tokenizers
import tokenizers.trainers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import x_transformers as xt
from einops import rearrange
from torch.utils.data import DataLoader, default_collate
from tqdm import tqdm as tq
from tqdm import trange
from transformers import GenerationConfig
from transformers.generation import GenerateEncoderDecoderOutput
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5ForConditionalGeneration,
    T5PreTrainedModel,
    T5Stack,
)

from arae.datasets import make_wikisentence_dataset


class UTT5Config(T5Config):
    def __init__(
        self,
        is_vae: bool = True,
        has_classifier: bool = True,
        num_classifier_layers: int | None = None,
        do_backtranslation: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.is_vae = is_vae
        self.has_classifier = has_classifier
        self.num_classifier_layers = (
            num_classifier_layers
            if num_classifier_layers is not None
            else self.num_layers
        )
        self.do_backtranslation = do_backtranslation


class T5ForUnsupervisedTranslation(T5ForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = [
        "decoder.block.0.layer.1.EncDecAttention.relative_attention_bias.weight",
    ]
    _tied_weights_keys = [
        "encoder.embed_tokens.weight",
        "decoder.embed_tokens.weight",
        "classifier.embed_tokens.weight",
        "lm_head.weight",
    ]

    def __init__(self, config: UTT5Config):
        super(T5PreTrainedModel, self).__init__(config)
        self.model_dim = config.d_model

        # self.shared = bnb.nn.StableEmbedding(config.vocab_size, config.d_model)
        self.shared = bnb.nn.Embedding(config.vocab_size, config.d_model)

        encoder_config = copy.deepcopy(config)
        encoder_config.is_decoder = False
        encoder_config.is_encoder_decoder = False
        encoder_config.use_cache = False
        self.encoder = T5Stack(encoder_config, self.shared)

        decoder_config = copy.deepcopy(config)
        decoder_config.is_decoder = True
        decoder_config.is_encoder_decoder = False
        decoder_config.num_layers = config.num_decoder_layers
        self.decoder = T5Stack(decoder_config, self.shared)

        if config.has_classifier:
            classifier_config = copy.deepcopy(config)
            classifier_config.is_decoder = False
            classifier_config.is_encoder_decoder = False
            classifier_config.use_cache = False
            classifier_config.num_layers = config.num_classifier_layers
            self.classifier = T5Stack(classifier_config, self.shared)

        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.is_vae:
            self.mean_head = nn.Linear(config.d_model, config.d_model, bias=False)
            self.log_var_head = nn.Linear(config.d_model, config.d_model, bias=False)
        if config.has_classifier:
            self.cls_head = nn.Linear(config.d_model, 1, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

        # Model parallel
        self.model_parallel = False
        self.device_map = None

    def set_input_embeddings(self, new_embeddings):
        self.shared = new_embeddings
        self.encoder.set_input_embeddings(new_embeddings)
        self.decoder.set_input_embeddings(new_embeddings)
        if self.config.has_classifier:
            self.classifier.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)
            if self.config.has_classifier:
                self._tie_or_clone_weights(self.classifier.embed_tokens, self.shared)

    def _init_weights(self, module):
        factor = self.config.initializer_factor
        if isinstance(module, T5ForUnsupervisedTranslation):
            print("Initializing UTT5")
            module.encoder.embed_tokens.weight.data.normal_(mean=0.0, std=1e-4)
            module.decoder.embed_tokens.weight.data.normal_(mean=0.0, std=1e-4)
            if module.config.has_classifier:
                module.classifier.embed_tokens.weight.data.normal_(mean=0.0, std=1e-4)
            module.lm_head.weight.data.normal_(mean=0.0, std=1e-4)
            module.shared.weight.data.normal_(mean=0.0, std=1e-4)
            # module.shared.norm.reset_parameters()
            if module.config.is_vae:
                print("Initializing VAE heads")
                module.mean_head.weight.data.normal_(
                    mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
                )
                module.log_var_head.weight.data.normal_(
                    mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
                )
            if module.config.has_classifier:
                print("Initializing CLS head")
                module.cls_head.weight.data.normal_(
                    mean=0.0, std=factor * ((self.config.d_model) ** -0.5)
                )
        else:
            print("Initializing T5")
            super()._init_weights(module)


class XTransformerForUnsupervisedTranslation(nn.Module):
    def __init__(
        self,
        **kwargs,
    ):
        super().__init__()

        self.x_transformer = xt.XTransformer(**kwargs)
        self.mean_head = nn.Linear(kwargs["dim"], kwargs["dim"], bias=False)
        self.log_var_head = nn.Linear(kwargs["dim"], kwargs["dim"], bias=False)

        self.x_transformer.encoder.token_emb.emb = bnb.nn.Embedding(
            kwargs["enc_num_tokens"], kwargs["dim"]
        )

        self.x_transformer.decoder.net.token_emb.emb = bnb.nn.Embedding(
            kwargs["dec_num_tokens"], kwargs["dim"]
        )

    def encoder(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> BaseModelOutputWithPastAndCrossAttentions:
        last_hidden_state = self.x_transformer.encoder(
            input_ids,
            mask=attention_mask.bool(),
            return_embeddings=True,
        )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state
        )

    def decoder(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> BaseModelOutputWithPastAndCrossAttentions:

        last_hidden_state = self.x_transformer.decoder.net(
            input_ids,
            mask=attention_mask.bool(),
            context=encoder_hidden_states,
            context_mask=encoder_attention_mask.bool(),
            return_embeddings=True,
        )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=last_hidden_state
        )

    def lm_head(self, x: torch.Tensor) -> torch.Tensor:
        return self.x_transformer.decoder.net.to_logits(x)


def exists(val):
    return val is not None


def max_neg_value(tensor):
    return -torch.finfo(tensor.dtype).max


def dropout_seq(seq, mask, dropout):
    b, n, *_, device = *seq.shape, seq.device
    logits = torch.randn(b, n, device=device)

    if exists(mask):
        mask_value = max_neg_value(logits)
        logits = logits.masked_fill(~mask, mask_value)

    keep_prob = 1.0 - dropout
    num_keep = max(1, int(keep_prob * n))
    keep_indices = logits.topk(num_keep, dim=1).indices

    batch_indices = torch.arange(b, device=device)
    batch_indices = rearrange(batch_indices, "b -> b 1")

    seq = seq[batch_indices, keep_indices]

    if exists(mask):
        seq_counts = mask.sum(dim=-1)
        seq_keep_counts = torch.ceil(seq_counts * keep_prob).int()
        keep_mask = torch.arange(num_keep, device=device) < rearrange(
            seq_keep_counts, "b -> b 1"
        )

        mask = mask[batch_indices, keep_indices] & keep_mask

    return seq, mask


def dropout_seq_simple(seq, mask, dropout):
    mask = torch.where(
        torch.rand(mask.shape, device=mask.device) < dropout,
        torch.zeros_like(mask),
        mask,
    )
    mask = mask.bool()

    return seq, mask


def standard_normal_log_kl_div(mean: torch.Tensor, log_var: torch.Tensor):
    return 0.5 * (torch.exp(log_var) + torch.square(mean) - 1 - log_var)


def filter_weight_decay(named_params: Iterable[Tuple[str, nn.Parameter]]):
    with_weight_decay: List[nn.Parameter] = []
    without_weight_decay: List[nn.Parameter] = []

    for n, p in named_params:
        apply_weight_decay = True

        # No weight decay for all bias terms
        # if "bias" in n:
        #    apply_weight_decay = False

        # No weight decay for all layer norms
        if "norm" in n:
            apply_weight_decay = False

        if apply_weight_decay:
            with_weight_decay.append(p)
        else:
            without_weight_decay.append(p)

    return with_weight_decay, without_weight_decay


def filter_duplicate_parameters(named_params: Iterable[Tuple[str, nn.Parameter]]):
    previous = set()
    for n, p in named_params:
        if p not in previous:
            previous.add(p)
            yield n, p


def make_param_groups(
    named_params: Iterable[Tuple[str, nn.Parameter]],
    learning_rate: float,
    weight_decay: float,
):
    named_params = filter_duplicate_parameters(named_params)
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


def tensor_dict_to_device(
    tensor_dict: Dict[str, torch.Tensor], device: str, dtype: torch.dtype
):
    return {
        key: (
            tensor.to(device=device, dtype=dtype)
            if tensor.dtype.is_floating_point
            else tensor.to(device=device)
        )
        for key, tensor in tensor_dict.items()
    }


def with_prefixes(d: dict, prefixes: List[str]):
    pd = {}

    for prefix in prefixes:
        pd.update({prefix + k: v for k, v in d.items()})

    return pd


def train_tokenizer(dataset: datasets.Dataset, vocab_size: int):
    # Build tokenizer
    tokenizer = tokenizers.Tokenizer(tokenizers.models.BPE())
    tokenizer.normalizer = tokenizers.normalizers.NFD()  # type: ignore
    tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Sequence(
        [
            tokenizers.pre_tokenizers.Digits(individual_digits=True),
            tokenizers.pre_tokenizers.ByteLevel(),
        ]
    )  # type: ignore
    tokenizer.decoder = tokenizers.decoders.ByteLevel()  # type: ignore

    # Build trainer
    trainer = tokenizers.trainers.BpeTrainer(
        vocab_size=vocab_size,
        show_progress=True,
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[MASK]", "[LABEL_0]", "[LABEL_1]"],
    )  # type: ignore

    # Make iterator
    def batch_iterator(batch_size: int = 2**16):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    # Train tokenizer
    tokenizer.train_from_iterator(
        batch_iterator(), trainer=trainer, length=len(dataset)
    )

    return tokenizer


def train_model(
    model_path: str,
    config: UTT5Config,
    tokenizer: tokenizers.Tokenizer,
    language0_dataset: datasets.Dataset,
    language1_dataset: datasets.Dataset,
):
    # Parameters
    learning_rate = 1e-4
    weight_decay = 1e-2
    batch_size = 32
    max_length = 32
    num_train_steps = 100000
    device = "cuda"
    dtype = config.torch_dtype
    ae_pretrain_steps = 25000

    save_interval = 100

    # Create dataset
    pad_token_id = tokenizer.token_to_id("[PAD]")
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    language0_token_id = tokenizer.token_to_id("[LABEL_0]")
    language1_token_id = tokenizer.token_to_id("[LABEL_1]")

    def add_label_fn(row: Dict[str, Any], label: int):
        return {"label": label}

    def tokenize_fn(row: Dict[str, Any]):
        text = row["text"]
        tokens: tokenizers.Encoding = tokenizer.encode(text)

        return {"token_ids": tokens.ids}

    def make_inputs_fn(row: Dict[str, Any]):
        token_ids = row["token_ids"]
        label = row["label"]

        label_token_id = language0_token_id if label == 0 else language1_token_id
        label_ignore_id = -100

        # Make encoder inputs
        encoder_input_ids = token_ids

        ## Truncate
        encoder_input_ids = encoder_input_ids[: max_length - 3]

        ## Add special tokens
        encoder_input_ids = (
            [bos_token_id, label_token_id] + encoder_input_ids + [eos_token_id]
        )

        ## Pad
        sequence_length = len(encoder_input_ids)
        pad_length = max_length - sequence_length

        encoder_input_ids = encoder_input_ids + [pad_token_id] * pad_length

        ## Attention mask
        encoder_attention_mask = [1] * sequence_length + [0] * pad_length

        # Make decoder inputs
        decoder_input_ids = token_ids

        ## Truncate
        decoder_input_ids = decoder_input_ids[: max_length - 3]

        ## Add special tokens
        decoder_input_ids = (
            [bos_token_id, label_token_id] + decoder_input_ids + [eos_token_id]
        )

        ## Pad
        sequence_length = len(decoder_input_ids)
        pad_length = max_length - sequence_length

        decoder_input_ids = decoder_input_ids + [pad_token_id] * pad_length

        ## Attention mask
        decoder_attention_mask = [1] * sequence_length + [0] * pad_length

        ## Labels
        decoder_labels = (
            [label_ignore_id]
            + decoder_input_ids[2:sequence_length]
            + [label_ignore_id] * (pad_length + 1)
        )

        return {
            "encoder_input_ids": encoder_input_ids,
            "encoder_attention_mask": encoder_attention_mask,
            "decoder_input_ids": decoder_input_ids,
            "decoder_attention_mask": decoder_attention_mask,
            "decoder_labels": decoder_labels,
            "class_label": label,
        }

    def corrupt_tokens_fn(
        row: Dict[str, Any],
        mask_token_id: int,
        mask_prob: float,
        randomize_prob: float,
        shuffle_prob: float,
        shuffle_max_range: int,
    ):
        token_ids = row["token_ids"]
        token_ids = np.asarray(token_ids, dtype=np.int64)
        corrupted_token_ids = token_ids.copy()

        # Shuffle tokens by selecting among nearby tokens
        random_offsets = np.random.randint(
            -shuffle_max_range, shuffle_max_range + 1, size=len(token_ids)
        )
        random_idx = (np.arange(len(token_ids)) + random_offsets) % len(token_ids)
        random_token_ids = token_ids[random_idx]
        random_mask = np.random.rand(len(token_ids)) < shuffle_prob
        corrupted_token_ids = np.where(
            random_mask, random_token_ids, corrupted_token_ids
        )

        # Randomize tokens by selecting among random token ids across the entire context
        random_idx = np.random.randint(len(token_ids), size=len(token_ids))
        random_token_ids = token_ids[random_idx]
        random_mask = np.random.rand(len(token_ids)) < randomize_prob
        corrupted_token_ids = np.where(
            random_mask, random_token_ids, corrupted_token_ids
        )

        # Randomly mask tokens
        random_mask = np.random.rand(len(token_ids)) < mask_prob
        corrupted_token_ids = np.where(random_mask, mask_token_id, corrupted_token_ids)

        return {"corrupted_token_ids": corrupted_token_ids.tolist()}

    language0_dataset = language0_dataset.shuffle().flatten_indices(keep_in_memory=True)
    language1_dataset = language1_dataset.shuffle().flatten_indices(keep_in_memory=True)

    language0_iterable_dataset = language0_dataset.to_iterable_dataset()
    language1_iterable_dataset = language1_dataset.to_iterable_dataset()

    language0_iterable_dataset = language0_iterable_dataset.map(
        add_label_fn, fn_kwargs={"label": 0}
    )
    language1_iterable_dataset = language1_iterable_dataset.map(
        add_label_fn, fn_kwargs={"label": 1}
    )

    train_dataset = datasets.combine.interleave_datasets(
        [language0_iterable_dataset, language1_iterable_dataset],  # type: ignore
        split=datasets.Split.TRAIN,
    )

    train_dataset = train_dataset.map(tokenize_fn)
    train_dataset = train_dataset.map(make_inputs_fn)
    train_dataset = train_dataset.select_columns(
        [
            "encoder_input_ids",
            "encoder_attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "decoder_labels",
            "class_label",
        ]
    )
    train_dataset = train_dataset.with_format(type="torch")

    train_dataloader = DataLoader(
        train_dataset,  # type: ignore
        batch_size,
        pin_memory=True,
        pin_memory_device=device,
        drop_last=True,
    )
    train_dataloader_iter = iter(train_dataloader)

    # Update config
    config.vocab_size = tokenizer.get_vocab_size()
    config.pad_token_id = pad_token_id
    config.bos_token_id = bos_token_id
    config.eos_token_id = eos_token_id
    config.use_cache = True

    # Create model
    # model = T5ForUnsupervisedTranslation(config)
    # model: T5ForUnsupervisedTranslation = model.to(device=device, dtype=dtype)  # type: ignore
    # model.train()
    # model.gradient_checkpointing_enable()
    # model.config.use_cache = False

    kwargs_dict = dict(
        num_tokens=config.vocab_size,
        max_seq_len=32,
        depth=4,
        heads=8,
        l2norm_embed=True,
        ff_glu=True,
        use_simple_rmsnorm=True,
        ff_no_bias=True,
        rotary_pos_emb=True,
        # rel_pos_bias=True,
    )

    model = XTransformerForUnsupervisedTranslation(
        dim=512, **with_prefixes(kwargs_dict, ["enc_", "dec_"])
    ).to(device=device, dtype=dtype)
    model.train()

    # safetensors.torch.load_model(model, f"{model_path}/model.safetensors")
    # model.to(device=device, dtype=dtype)
    # model.train()

    # Create optimizers
    """
    autoencoder_parameters = [
        model.shared.named_parameters(prefix="shared"),
        model.encoder.named_parameters(prefix="encoder"),
        model.decoder.named_parameters(prefix="decoder"),
        model.lm_head.named_parameters(prefix="lm_head"),
    ]

    if config.is_vae:
        autoencoder_parameters.extend(
            [
                model.mean_head.named_parameters(prefix="mean_head"),
                model.log_var_head.named_parameters(prefix="log_var_head"),
            ]
        )
    """

    # autoencoder_parameters = itertools.chain(*autoencoder_parameters)
    autoencoder_parameters = model.named_parameters()

    autoencoder_optimizer = bnb.optim.AdamW8bit(
        make_param_groups(autoencoder_parameters, learning_rate, weight_decay)
    )

    if config.has_classifier:
        classifier_parameters = itertools.chain(
            model.shared.named_parameters(prefix="shared"),
            model.classifier.named_parameters(prefix="classifier"),
            model.cls_head.named_parameters(prefix="cls_head"),
        )

        classifier_optimizer = bnb.optim.AdamW8bit(
            make_param_groups(classifier_parameters, learning_rate, weight_decay)
        )

    def encode(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        include_loss: bool = False,
    ):
        # Encode
        outputs: BaseModelOutputWithPastAndCrossAttentions = model.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )

        # Latents
        latents = outputs.last_hidden_state
        latent_loss = 0.0

        if config.is_vae:
            # Sample latents
            latents_mean = model.mean_head(latents)
            latents_log_var = model.log_var_head(latents)

            latents_noise = torch.randn_like(latents_mean)
            latents = latents_mean + latents_noise * torch.exp(0.5 * latents_log_var)

            # Latent Loss
            if include_loss:
                latent_loss = standard_normal_log_kl_div(latents_mean, latents_log_var)

                # Mask loss
                latent_loss = latent_loss * attention_mask[..., None]
                # Sum over latent features
                latent_loss = torch.sum(latent_loss, dim=2)
                # Manual mean over tokens
                num_tokens = torch.sum(attention_mask, dim=1)
                latent_loss = torch.sum(latent_loss, dim=1) / num_tokens
                # Mean over batch
                latent_loss = torch.mean(latent_loss, dim=0)

        return outputs, latents, latent_loss

    def decode(
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        encoder_latents: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        include_loss: bool = False,
    ):
        encoder_latents, encoder_attention_mask = dropout_seq_simple(
            encoder_latents, encoder_attention_mask.bool(), 0.5
        )

        # Outputs
        outputs: BaseModelOutputWithPastAndCrossAttentions = model.decoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_latents,
            encoder_attention_mask=encoder_attention_mask,
        )

        # Logits
        logits: torch.Tensor = model.lm_head(outputs.last_hidden_state)

        # Loss
        reconstruction_loss = 0.0
        if include_loss:
            assert labels is not None

            reconstruction_loss = F.cross_entropy(
                logits.reshape(-1, config.vocab_size), labels.reshape(-1)
            )

        return outputs, logits, reconstruction_loss

    def classify(
        latents: torch.Tensor,
        attention_mask: torch.Tensor,
        label: torch.Tensor | None = None,
        include_loss: bool = False,
    ):
        # Classify latents
        output: BaseModelOutputWithPastAndCrossAttentions = model.classifier(
            inputs_embeds=latents,
            attention_mask=attention_mask,
        )

        last_hidden_state = output.last_hidden_state

        # Pool hidden states
        last_hidden_state = last_hidden_state * attention_mask[..., None]
        num_tokens = torch.sum(attention_mask, dim=1, keepdim=True)
        last_hidden_state = torch.sum(last_hidden_state, dim=1) / num_tokens

        # Logits
        logits = model.cls_head(last_hidden_state)

        # Loss
        loss = 0.0
        if include_loss:
            assert label is not None

            loss = F.binary_cross_entropy_with_logits(logits[:, 0], label.float())

        return output, logits, loss

    def sample(
        encoder_latents: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
        target_label: torch.Tensor,
    ):
        sample_language_token_id = torch.where(
            target_label == 0, language0_token_id, language1_token_id
        )
        sample_language_token_id = sample_language_token_id.cpu().numpy().tolist()

        generation_config = GenerationConfig(
            max_length=max_length,
            # penalty_alpha=0.6,
            # top_k=4,
            forced_bos_token_id=sample_language_token_id,
        )

        model.gradient_checkpointing_disable()
        model.config.use_cache = True
        generation_output: torch.Tensor = model.generate(
            encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=encoder_latents
            ),
            attention_mask=encoder_attention_mask,
            generation_config=generation_config,
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable()

        return generation_output

    def samples_to_batch(
        generation_output: torch.Tensor, labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Convert from torch back to python objects
        label_list = labels.cpu().numpy().tolist()
        token_ids_list = generation_output.cpu().numpy().tolist()

        batch_inputs = []

        # Process each token sequence into a training data point
        token_ids: List[int]
        label: int
        for token_ids, label in zip(token_ids_list, label_list):
            start_index = 2  # [BOS] [LABEL_X] Text Token Ids [EOS] [PAD] ...
            end_index = -1

            try:
                end_index = token_ids.index(eos_token_id)
            except ValueError:
                pass

            # Extract text tokens from generated tokens
            text_token_ids = token_ids[start_index:end_index]

            # Make inputs
            inputs = make_inputs_fn({"token_ids": text_token_ids, "label": label})
            inputs = {
                k: (
                    torch.LongTensor(v)
                    if isinstance(v, list)
                    else torch.scalar_tensor(label, dtype=torch.int64)
                )
                for k, v in inputs.items()
            }
            batch_inputs.append(inputs)

        # Collate back to torch training batch
        return default_collate(batch_inputs)

    def get_next_batch(train_dataloader_iter):
        try:
            batch_data = next(train_dataloader_iter)
        except StopIteration:
            train_dataloader_iter = iter(train_dataloader)
            batch_data = next(train_dataloader_iter)

        batch_data = tensor_dict_to_device(batch_data, device, dtype)
        return batch_data, train_dataloader_iter

    pbar = trange(num_train_steps)
    for step_id in pbar:
        # Autoencoder step

        ## Get new batch
        # batch_data = next(train_dataloader_iter)
        # batch_data = tensor_dict_to_device(batch_data, device, dtype)

        batch_data, train_dataloader_iter = get_next_batch(train_dataloader_iter)

        ## Encode
        encoder_outputs, encoder_latents, latent_loss = encode(
            batch_data["encoder_input_ids"],
            batch_data["encoder_attention_mask"],
            include_loss=True,
        )

        ## Decode
        decoder_outputs, decoder_logits, reconstruction_loss = decode(
            batch_data["decoder_input_ids"],
            batch_data["decoder_attention_mask"],
            encoder_latents,
            batch_data["encoder_attention_mask"],
            batch_data["decoder_labels"],
            include_loss=True,
        )

        ## Adversarial
        adversarial_loss = 0
        if config.has_classifier:
            model.classifier.requires_grad_(False)
            model.cls_head.requires_grad_(False)

            classifier_outputs, classifier_logits, adversarial_loss = classify(
                encoder_latents,
                batch_data["encoder_attention_mask"],
                1 - batch_data["class_label"],
                include_loss=True,
            )

            model.classifier.requires_grad_(True)
            model.cls_head.requires_grad_(True)

        ## Backtranslation
        bt_latent_loss = 0.0
        bt_reconstruction_loss = 0.0
        if config.do_backtranslation and step_id >= ae_pretrain_steps:
            ### Get new batch
            # source_batch_data = next(train_dataloader_iter)
            # source_batch_data = tensor_dict_to_device(source_batch_data, device, dtype)

            source_batch_data, train_dataloader_iter = get_next_batch(
                train_dataloader_iter
            )

            with torch.no_grad():
                ### Encode
                encoder_outputs, encoder_latents, _ = encode(
                    source_batch_data["encoder_input_ids"],
                    source_batch_data["encoder_attention_mask"],
                    include_loss=False,
                )

                ### Sample
                generation_output = sample(
                    encoder_latents,
                    source_batch_data["encoder_attention_mask"],
                    1 - source_batch_data["class_label"],
                )

                ### To new batch
                translated_batch_data = samples_to_batch(
                    generation_output, source_batch_data["class_label"]
                )
                translated_batch_data = tensor_dict_to_device(
                    translated_batch_data, device, dtype
                )

            ## Encode
            encoder_outputs, encoder_latents, bt_latent_loss = encode(
                translated_batch_data["encoder_input_ids"],
                translated_batch_data["encoder_attention_mask"],
                include_loss=True,
            )

            ## Decode
            decoder_outputs, decoder_logits, bt_reconstruction_loss = decode(
                source_batch_data["decoder_input_ids"],
                source_batch_data["decoder_attention_mask"],
                encoder_latents,
                translated_batch_data["encoder_attention_mask"],
                source_batch_data["decoder_labels"],
                include_loss=True,
            )

        ## Step
        loss = (
            reconstruction_loss
            + bt_reconstruction_loss
            + 0.01 * latent_loss
            + 0.01 * bt_latent_loss
            + adversarial_loss
        )

        autoencoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        autoencoder_optimizer.step()

        # Classifier step
        classifier_loss = 0
        if config.has_classifier:
            ## Get new batch
            # batch_data = next(train_dataloader_iter)
            # batch_data = tensor_dict_to_device(batch_data, device, dtype)

            batch_data, train_dataloader_iter = get_next_batch(train_dataloader_iter)

            ## Encode
            with torch.no_grad():
                encoder_outputs, encoder_latents, latent_loss = encode(
                    batch_data["encoder_input_ids"],
                    batch_data["encoder_attention_mask"],
                    include_loss=True,
                )

            ## Classify
            classifier_outputs, classifier_logits, classifier_loss = classify(
                encoder_latents,
                batch_data["encoder_attention_mask"],
                batch_data["class_label"],
                include_loss=True,
            )

            loss = classifier_loss

            classifier_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            classifier_optimizer.step()

        if (step_id + 1) % save_interval == 0:
            # model.config.use_cache = True
            # model.save_pretrained(model_path)
            # model.config.use_cache = False

            os.makedirs(model_path, exist_ok=True)
            safetensors.torch.save_model(model, f"{model_path}/model.safetensors")
            tokenizer.save(f"{model_path}/tokenizer.json")

        pbar.set_postfix_str(
            f"AE: {reconstruction_loss:.2e} Z: {latent_loss:.2e} BT-AE: {bt_reconstruction_loss:.2e} BT-Z: {bt_latent_loss:.2e} ADV: {adversarial_loss:.2e} CLS: {classifier_loss:.2e}"
        )


def load_or_train_tokenizer(
    tokenizer_path: str,
    language0_dataset: datasets.Dataset,
    language1_dataset: datasets.Dataset,
    vocab_size: int,
):
    try:
        tokenizer = tokenizers.Tokenizer.from_file(f"{tokenizer_path}/tokenizer.json")
    except Exception:
        tokenizer_dataset = datasets.combine.concatenate_datasets(
            [language0_dataset, language1_dataset],  # type: ignore
            split=datasets.Split.TRAIN,
        )

        tokenizer = train_tokenizer(tokenizer_dataset, vocab_size)

        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save(f"{tokenizer_path}/tokenizer.json")

    return tokenizer


def main_train():
    # Parameters

    ## Config

    # T5-large
    if False:
        base_config = T5Config(
            feed_forward_proj="gated-silu",
            is_gated_act=True,
            dense_act_fn="silu",
            d_ff=2048,
            d_kv=64,
            d_model=1024,
            n_positions=512,
            num_heads=16,
            num_layers=24,
            relative_attention_max_distance=128,
            relative_attention_num_buckets=32,
        )

    # T5-small
    if False:
        base_config = T5Config(
            feed_forward_proj="gated-silu",
            is_gated_act=True,
            dense_act_fn="silu",
            d_ff=2048,
            d_kv=64,
            d_model=512,
            n_positions=512,
            num_heads=8,
            num_layers=6,
            relative_attention_max_distance=128,
            relative_attention_num_buckets=32,
        )

    # T5-Tiny
    base_config = T5Config(
        feed_forward_proj="gated-silu",
        is_gated_act=True,
        dense_act_fn="silu",
        d_ff=1024,
        d_kv=64,
        d_model=256,
        n_positions=512,
        num_heads=4,
        num_layers=1,  # 2
        relative_attention_max_distance=32,
        relative_attention_num_buckets=65,
        tie_word_embeddings=False,  # True
        dropout_rate=0.0,  # 0.1
    )

    config = UTT5Config(
        is_vae=True,
        has_classifier=False,
        do_backtranslation=False,
        **base_config.to_dict(),
    )

    ## Directories
    model_path = "results/varaegan/models/2024-04-27-xtransformers_small_hmm_z1e-2"
    # tokenizer_path = "results/varaegan/tokenizers/2024-04-24_hmm_vocab256"
    tokenizer_path = "results/varaegan/tokenizers/2024-04-24_hmm_vocab256"

    # language0_wikisentence_path = "data/eng_wikipedia_2016_1M-sentences.txt"
    # language1_wikisentence_path = "data/deu_wikipedia_2016_1M-sentences.txt"

    # language0_wikisentence_path = "data/simple-bilingual-stories-eng-sentences.txt"
    # language1_wikisentence_path = "data/simple-bilingual-stories-ger-sentences.txt"

    language0_wikisentence_path = "data/lang1_hmm_state.txt"
    language1_wikisentence_path = "data/lang1_hmm_output.txt"

    ## Training

    ## Tokenizer
    vocab_size = 1024

    # Prepare datasets
    language0_dataset = make_wikisentence_dataset(
        language0_wikisentence_path, shuffle=False, iterable=False, clean_lines=False
    )
    language1_dataset = make_wikisentence_dataset(
        language1_wikisentence_path, shuffle=False, iterable=False, clean_lines=False
    )

    # Load or train tokenizer
    tokenizer = load_or_train_tokenizer(
        tokenizer_path, language0_dataset, language1_dataset, vocab_size
    )

    # Train model
    train_model(model_path, config, tokenizer, language0_dataset, language1_dataset)


@torch.no_grad
def main_eval():
    # model_path = "results/varaegan/models/2024-04-15_vocab1024_t5tiny_weakz_bt_noadv"
    # tokenizer_path = "results/varaegan/tokenizers/2024-04-13_vocab1024"

    model_path = "results/varaegan/models/2024-04-27-xtransformers_small_hmm_z1e-2"
    # tokenizer_path = "results/varaegan/tokenizers/2024-04-24_hmm_vocab256"
    tokenizer_path = model_path

    device = "cuda"
    dtype = torch.float32

    # Load model
    # model = T5ForUnsupervisedTranslation.from_pretrained(model_path)

    tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_file(
        f"{tokenizer_path}/tokenizer.json"
    )

    kwargs_dict = dict(
        num_tokens=tokenizer.get_vocab_size(),
        max_seq_len=32,
        depth=4,
        heads=8,
        l2norm_embed=True,
        ff_glu=True,
        use_simple_rmsnorm=True,
        ff_no_bias=True,
        rotary_pos_emb=True,
    )

    model = XTransformerForUnsupervisedTranslation(
        dim=512, **with_prefixes(kwargs_dict, ["enc_", "dec_"])
    ).to(device=device, dtype=dtype)
    model.eval()

    safetensors.torch.load_model(model, f"{model_path}/model.safetensors")

    pad_token_id = tokenizer.token_to_id("[PAD]")
    bos_token_id = tokenizer.token_to_id("[BOS]")
    eos_token_id = tokenizer.token_to_id("[EOS]")
    language0_token_id = tokenizer.token_to_id("[LABEL_0]")
    language1_token_id = tokenizer.token_to_id("[LABEL_1]")

    # Eval loop
    while True:
        input_text = input("Input (str): ")
        target_language = input("Target Language (Literal[0, 1]): ")

        # Get latents
        encoding: tokenizers.Encoding = tokenizer.encode(input_text)
        encoder_input_ids = [bos_token_id] + encoding.ids[:30] + [eos_token_id]
        encoder_input_ids = torch.LongTensor([encoder_input_ids]).to(device)

        encoder_outputs: BaseModelOutputWithPastAndCrossAttentions = model.encoder(
            input_ids=encoder_input_ids,
            attention_mask=torch.ones_like(encoder_input_ids),
        )
        encoder_latents = encoder_outputs.last_hidden_state
        # if model.config.is_vae:
        # encoder_latents_mean = model.mean_head(encoder_latents)
        # encoder_latents = encoder_latents_mean

        # Sample latents
        latents = encoder_latents
        latents_mean = model.mean_head(latents)
        latents_log_var = model.log_var_head(latents)

        latents_noise = torch.randn_like(latents_mean)
        latents = latents_mean + latents_noise * torch.exp(0.5 * latents_log_var)
        encoder_latents = latents

        # Generate
        # generation_config = GenerationConfig(
        #    max_new_tokens=128,
        #    penalty_alpha=0.6,
        #    top_k=4,
        #    forced_bos_token_id=(
        #        language0_token_id if target_language == "0" else language1_token_id
        #    ),
        # )

        # generation_config = GenerationConfig(
        #    max_new_tokens=128,
        #    do_sample=True,
        #    top_k=0,
        #    top_p=0.95,
        #    forced_bos_token_id=(
        #        language0_token_id if target_language == "0" else language1_token_id
        #    ),
        # )

        # generation_config = GenerationConfig(
        #    max_new_tokens=128,
        #    forced_bos_token_id=(
        #        language0_token_id if target_language == "0" else language1_token_id
        #    ),
        # )

        # generation_output = model.generate(
        #    encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
        #        last_hidden_state=encoder_latents
        #    ),
        #    generation_config=generation_config,
        # )

        language_token_id = (
            language0_token_id if target_language == "0" else language1_token_id
        )

        decoder_input_ids = torch.LongTensor([[bos_token_id, language_token_id]]).to(
            device
        )
        generation_output = model.x_transformer.decoder.generate(
            decoder_input_ids,
            eos_token=eos_token_id,
            seq_len=32,
            context=encoder_latents,
            context_mask=torch.ones_like(encoder_input_ids).bool(),
            temperature=0.0,
        )

        print("----")
        print(generation_output)
        print("----")
        generation_output = generation_output.cpu().numpy().tolist()[0]
        print(encoder_input_ids)
        print(decoder_input_ids)
        print(tokenizer.decode(generation_output, skip_special_tokens=False))

        # Decoder loss
        decoder_input_ids = (
            [bos_token_id, language_token_id] + encoding.ids[: 32 - 3] + [eos_token_id]
        )
        decoder_labels = [-100] + encoding.ids[: 32 - 3] + [eos_token_id, -100]

        decoder_input_ids = torch.LongTensor([decoder_input_ids]).to(device)
        decoder_labels = torch.LongTensor([decoder_labels]).to(device)

        # Outputs
        outputs: BaseModelOutputWithPastAndCrossAttentions = model.decoder(
            input_ids=decoder_input_ids,
            attention_mask=torch.ones_like(decoder_input_ids),
            encoder_hidden_states=encoder_latents,
            encoder_attention_mask=torch.ones_like(encoder_input_ids),
        )

        # Logits
        logits: torch.Tensor = model.lm_head(outputs.last_hidden_state)

        # Loss
        reconstruction_loss = F.cross_entropy(
            logits.reshape(-1, tokenizer.get_vocab_size()), decoder_labels.reshape(-1)
        )

        print(reconstruction_loss)


if __name__ == "__main__":
    # main_train()
    main_eval()
