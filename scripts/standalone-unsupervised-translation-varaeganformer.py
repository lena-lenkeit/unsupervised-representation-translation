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
from torch.utils.data import DataLoader
from tqdm import tqdm as tq
from tqdm import trange
from transformers import GenerationConfig
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
            return


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


def tensor_dict_to_device(tensor_dict: Dict[str, torch.Tensor], device: str):
    return {key: tensor.to(device) for key, tensor in tensor_dict.items()}


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
        special_tokens=["[PAD]", "[BOS]", "[EOS]", "[LABEL_0]", "[LABEL_1]"],
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
        encoder_input_ids = encoder_input_ids[: max_length - 2]

        ## Add special tokens
        encoder_input_ids = [bos_token_id] + encoder_input_ids + [eos_token_id]

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
    )
    train_dataloader_iter = iter(train_dataloader)

    # Update config
    config.vocab_size = tokenizer.get_vocab_size()
    config.pad_token_id = pad_token_id
    config.bos_token_id = bos_token_id
    config.eos_token_id = eos_token_id

    # Create model
    model = T5ForUnsupervisedTranslation(config)
    model = model.to(device)  # type: ignore
    model.train()
    model.gradient_checkpointing_enable()

    # Create optimizers
    autoencoder_parameters = itertools.chain(
        model.shared.named_parameters(prefix="shared"),
        model.encoder.named_parameters(prefix="encoder"),
        model.decoder.named_parameters(prefix="decoder"),
        model.mean_head.named_parameters(prefix="mean_head"),
        model.log_var_head.named_parameters(prefix="log_var_head"),
        model.lm_head.named_parameters(prefix="lm_head"),
    )

    classifier_parameters = itertools.chain(
        model.shared.named_parameters(prefix="shared"),
        model.classifier.named_parameters(prefix="classifier"),
        model.cls_head.named_parameters(prefix="cls_head"),
    )

    autoencoder_optimizer = bnb.optim.AdamW8bit(
        make_param_groups(autoencoder_parameters, learning_rate, weight_decay)
    )

    classifier_optimizer = bnb.optim.AdamW8bit(
        make_param_groups(classifier_parameters, learning_rate, weight_decay)
    )

    pbar = trange(num_train_steps)
    for i in pbar:
        # Autoencoder step

        ## Get new batch
        batch_data = next(train_dataloader_iter)
        batch_data = tensor_dict_to_device(batch_data, device)

        ## Encode
        encoder_outputs: BaseModelOutputWithPastAndCrossAttentions = model.encoder(
            batch_data["encoder_input_ids"], batch_data["encoder_attention_mask"]
        )

        ## Latents
        encoder_latents = encoder_outputs.last_hidden_state
        autoencoder_latent_loss = 0
        if config.is_vae:
            encoder_latents_mean = model.mean_head(encoder_latents)
            encoder_latents_log_var = model.log_var_head(encoder_latents)

            encoder_latents_noise = torch.randn_like(encoder_latents_mean)
            encoder_latents = encoder_latents_mean + encoder_latents_noise * torch.exp(
                0.5 * encoder_latents_log_var
            )

            autoencoder_latent_loss = standard_normal_log_kl_div(
                encoder_latents_mean, encoder_latents_log_var
            )
            autoencoder_latent_loss = autoencoder_latent_loss * batch_data[
                "encoder_attention_mask"
            ].reshape(batch_size, max_length, 1)

            autoencoder_latent_loss = torch.sum(autoencoder_latent_loss, dim=2)
            autoencoder_latent_loss = torch.sum(
                autoencoder_latent_loss, dim=1
            ) / torch.sum(batch_data["encoder_attention_mask"], dim=1)
            autoencoder_latent_loss = torch.mean(autoencoder_latent_loss, dim=0)

        ## Decode
        decoder_outputs: BaseModelOutputWithPastAndCrossAttentions = (
            model.decoder.forward(
                input_ids=batch_data["decoder_input_ids"],
                attention_mask=batch_data["decoder_attention_mask"],
                encoder_hidden_states=encoder_latents,
                encoder_attention_mask=batch_data["encoder_attention_mask"],
            )
        )

        ## Tokens
        decoder_token_logits = model.lm_head(decoder_outputs.last_hidden_state)

        ## Reconstruction loss
        autoencoder_reconstruction_loss = F.cross_entropy(
            decoder_token_logits.reshape(-1, config.vocab_size),
            batch_data["decoder_labels"].reshape(-1),
        )

        ## Step
        loss = autoencoder_reconstruction_loss + 0.1 * autoencoder_latent_loss

        autoencoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        autoencoder_optimizer.step()

        if (i + 1) % save_interval == 0:
            model.save_pretrained(model_path)
            tokenizer.save(f"{model_path}/tokenizer.json")

        pbar.set_postfix_str(
            f"AE: {autoencoder_reconstruction_loss:.2e} Z: {autoencoder_latent_loss:.2e}"
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
    base_config = T5Config(
        feed_forward_proj="gated-silu", is_gated_act=True, dense_act_fn="silu"
    )
    config = UTT5Config(**base_config.to_dict())

    ## Directories
    model_path = "results/varaegan/models/2024-04-13_vocab1024_t5tiny"
    tokenizer_path = "results/varaegan/tokenizers/2024-04-13_vocab1024"

    language0_wikisentence_path = "data/eng_wikipedia_2016_1M-sentences.txt"
    language1_wikisentence_path = "data/deu_wikipedia_2016_1M-sentences.txt"

    ## Training

    ## Tokenizer
    vocab_size = 1024

    # Prepare datasets
    language0_dataset = make_wikisentence_dataset(
        language0_wikisentence_path, shuffle=False, iterable=False, clean_lines=True
    )
    language1_dataset = make_wikisentence_dataset(
        language1_wikisentence_path, shuffle=False, iterable=False, clean_lines=True
    )

    # Load or train tokenizer
    tokenizer = load_or_train_tokenizer(
        tokenizer_path, language0_dataset, language1_dataset, vocab_size
    )

    # Train model
    train_model(model_path, config, tokenizer, language0_dataset, language1_dataset)


@torch.no_grad
def main_eval():
    model_path = "results/varaegan/models/2024-04-13_vocab1024_t5tiny"
    tokenizer_path = "results/varaegan/tokenizers/2024-04-13_vocab1024"
    device = "cuda"

    # Load model
    model = T5ForUnsupervisedTranslation.from_pretrained(model_path)
    tokenizer: tokenizers.Tokenizer = tokenizers.Tokenizer.from_file(
        f"{tokenizer_path}/tokenizer.json"
    )

    model.eval()
    model = model.to(device)  # type: ignore

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
        encoder_input_ids = [bos_token_id] + encoding.ids + [eos_token_id]
        encoder_input_ids = torch.LongTensor([encoder_input_ids]).to(device)

        encoder_outputs: BaseModelOutputWithPastAndCrossAttentions = model.encoder(
            input_ids=encoder_input_ids
        )
        encoder_latents = encoder_outputs.last_hidden_state
        if model.config.is_vae:
            encoder_latents_mean = model.mean_head(encoder_latents)
            encoder_latents = encoder_latents_mean

        # Generate
        generation_config = GenerationConfig(
            max_new_tokens=128,
            penalty_alpha=0.6,
            top_k=4,
            forced_bos_token_id=(
                language0_token_id if target_language == 0 else language1_token_id
            ),
        )

        generation_output = model.generate(
            encoder_outputs=BaseModelOutputWithPastAndCrossAttentions(
                last_hidden_state=encoder_latents
            ),
            generation_config=generation_config,
        )

        generation_output = generation_output.cpu().numpy().tolist()[0]
        print(tokenizer.decode(generation_output, skip_special_tokens=False))


if __name__ == "__main__":
    main_train()
    # main_eval()
