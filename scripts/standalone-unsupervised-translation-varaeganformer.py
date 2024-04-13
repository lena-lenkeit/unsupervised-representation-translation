import copy
import itertools
import os
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import bitsandbytes as bnb
import datasets
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
from transformers.models.t5.modeling_t5 import (
    T5Config,
    T5ForConditionalGeneration,
    T5Stack,
)

from arae.datasets import make_wikisentence_dataset


class UTT5Config(T5Config):
    def __init__(
        self,
        is_vae: bool = True,
        has_classifier: bool = True,
        num_classifier_layers: int | None = None,
        **kwargs,
    ):
        self.is_vae = is_vae
        self.has_classifier = has_classifier
        self.num_classifier_layers = (
            num_classifier_layers
            if num_classifier_layers is not None
            else kwargs["num_layers"]
        )

        super().__init__(**kwargs)


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
        super().__init__(config)
        self.model_dim = config.d_model

        self.shared = bnb.nn.StableEmbedding(config.vocab_size, config.d_model)

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

        classifier_config = copy.deepcopy(config)
        classifier_config.is_decoder = False
        classifier_config.is_encoder_decoder = False
        classifier_config.use_cache = False
        classifier_config.num_layers = config.num_classifier_layers
        self.classifier = T5Stack(classifier_config, self.shared)

        self.mean_head = nn.Linear(config.d_model, config.d_model, bias=False)
        self.log_var_head = nn.Linear(config.d_model, config.d_model, bias=False)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
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
        self.classifier.set_input_embeddings(new_embeddings)

    def _tie_weights(self):
        if self.config.tie_word_embeddings:
            self._tie_or_clone_weights(self.encoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.decoder.embed_tokens, self.shared)
            self._tie_or_clone_weights(self.classifier.embed_tokens, self.shared)


def standard_normal_log_kl_div(mean: torch.Tensor, log_var: torch.Tensor):
    return 0.5 * (torch.exp(log_var) + torch.square(mean) - 1 - log_var)


def filter_weight_decay(named_params: Iterable[Tuple[str, nn.Parameter]]):
    with_weight_decay: List[nn.Parameter] = []
    without_weight_decay: List[nn.Parameter] = []

    for n, p in named_params:
        apply_weight_decay = True

        # No weight decay for all bias terms
        if "bias" in n:
            apply_weight_decay = False

        # No weight decay for all layer norms
        if "norm" in n:
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


def train_tokenizer(dataset: datasets.Dataset):
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
        vocab_size=2**14,
        show_progress=True,
        initial_alphabet=tokenizers.pre_tokenizers.ByteLevel.alphabet(),
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[LABEL_0]", "[LABEL_1]"],
    )  # type: ignore

    # Make iterator
    def batch_iterator(batch_size: int = 1024):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    # Train tokenizer
    tokenizer.train_from_iterator(
        batch_iterator(), trainer=trainer, length=len(dataset)
    )

    return tokenizer


def train_model(
    config: UTT5Config,
    tokenizer: tokenizers.Tokenizer,
    language0_dataset: datasets.Dataset,
    language1_dataset: datasets.Dataset,
):
    # Parameters
    learning_rate = 1e-4
    weight_decay = 1e-4
    batch_size = 32
    max_seq_len = 32

    # Create dataset
    def add_label_fn(row: Dict[str, Any], label: int):
        return {"label": label}

    def tokenize_fn(row: Dict[str, Any]):
        text = row["text"]
        tokens: tokenizers.Encoding = tokenizer.encode(text)

        return {"token_ids": tokens.ids}

    def make_inputs_fn(row: Dict[str, Any]): ...

    language0_dataset = language0_dataset.shuffle().flatten_indices(keep_in_memory=True)
    language1_dataset = language1_dataset.shuffle().flatten_indices(keep_in_memory=True)

    language0_dataset = language0_dataset.to_iterable_dataset()
    language1_dataset = language1_dataset.to_iterable_dataset()

    language0_dataset = language0_dataset.map(add_label_fn, fn_kwargs={"label": 0})
    language1_dataset = language1_dataset.map(add_label_fn, fn_kwargs={"label": 1})

    train_dataset = datasets.combine.interleave_datasets(
        [language0_dataset, language1_dataset],  # type: ignore
        split=datasets.Split.TRAIN,
    )

    train_dataset = train_dataset.map(tokenize_fn)
    train_dataset = train_dataset.map(make_inputs_fn)
    train_dataset = train_dataset.with_format(
        type="torch",
        columns=[
            "encoder_input_ids",
            "encoder_attention_mask",
            "decoder_input_ids",
            "decoder_attention_mask",
            "decoder_labels",
            "class_labels",
        ],
    )

    # Update config
    config.vocab_size = tokenizer.get_vocab_size()

    # Create model
    model = T5ForUnsupervisedTranslation(config)
    model.train()
    model.gradient_checkpointing_enable()

    # Create optimizers
    autoencoder_parameters = itertools.chain(
        model.shared.named_parameters(),
        model.encoder.named_parameters(),
        model.decoder.named_parameters(),
        model.mean_head.named_parameters(),
        model.log_var_head.named_parameters(),
        model.lm_head.named_parameters(),
    )

    classifier_parameters = itertools.chain(
        model.shared.named_parameters(),
        model.classifier.named_parameters(),
        model.cls_head.named_parameters(),
    )

    autoencoder_optimizer = bnb.optim.AdamW8bit(
        make_param_groups(autoencoder_parameters, learning_rate, weight_decay)
    )

    classifier_optimizer = bnb.optim.AdamW8bit(
        make_param_groups(classifier_parameters, learning_rate, weight_decay)
    )


def load_or_train_tokenizer(
    tokenizer_path: str,
    language0_dataset: datasets.Dataset,
    language1_dataset: datasets.Dataset,
):
    try:
        tokenizer = tokenizers.Tokenizer.from_file(f"{tokenizer_path}/tokenizer.json")
    except Exception:
        tokenizer_dataset = datasets.combine.concatenate_datasets(
            [language0_dataset, language1_dataset],  # type: ignore
            split=datasets.Split.TRAIN,
        )

        tokenizer = train_tokenizer(tokenizer_dataset)

        os.makedirs(tokenizer_path, exist_ok=True)
        tokenizer.save(f"{tokenizer_path}/tokenizer.json")

    return tokenizer


def main():
    # Parameters

    ## Directories
    tokenizer_path = "results/varaegan/tokenizers/2024-04-12_vocab16384"

    language0_wikisentence_path = "data/eng_wikipedia_2016_1M-sentences.txt"
    language1_wikisentence_path = "data/deu_wikipedia_2016_1M-sentences.txt"

    ## Training

    # Prepare datasets
    language0_dataset = make_wikisentence_dataset(
        language0_wikisentence_path, shuffle=False, iterable=False, clean_lines=True
    )
    language1_dataset = make_wikisentence_dataset(
        language1_wikisentence_path, shuffle=False, iterable=False, clean_lines=True
    )

    # Load or train tokenizer
    tokenizer = load_or_train_tokenizer(
        tokenizer_path, language0_dataset, language1_dataset
    )

    # Train model
    train_model()


if __name__ == "__main__":
    main()
