"""A self-contained development and testing script for small Proofs-Of-Concept (POCs)
for Unsupervised Translation"""

import itertools
import math
from typing import Callable, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from tqdm.auto import tqdm as tq
from tqdm.auto import trange


class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int | None = None,
        num_hidden_layers: int = 0,
        hidden_bias: bool = True,
        output_bias: bool = False,
        hidden_activation: Callable | None = nn.ReLU(),
        output_activation: Callable | None = None,
    ):
        super(FeedForwardNetwork, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_bias = hidden_bias
        self.output_bias = output_bias
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation

        if num_hidden_layers == 0:
            self.layers = nn.ModuleList(
                [nn.Linear(input_size, output_size, bias=output_bias)]
            )
        else:
            assert hidden_size is not None

            self.layers = nn.ModuleList(
                [nn.Linear(input_size, hidden_size, bias=hidden_bias)]
            )
            for _ in range(num_hidden_layers - 1):
                self.layers.append(
                    nn.Linear(hidden_size, hidden_size, bias=hidden_bias)
                )
            self.layers.append(nn.Linear(hidden_size, output_size, bias=output_bias))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < self.num_hidden_layers:
                if self.hidden_activation is not None:
                    x = self.hidden_activation(x)
            else:
                if self.output_activation is not None:
                    x = self.output_activation(x)

        return x


class SelfNormalizingNetwork(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int = -1,
        num_hidden_layers: int = 0,
        hidden_bias: bool = True,
        output_bias: bool = False,
        output_activation: Callable | None = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_bias = hidden_bias
        self.output_bias = output_bias
        self.output_activation = output_activation
        self.dropout = nn.AlphaDropout(dropout)

        self.layers = nn.ModuleList()
        for i in range(num_hidden_layers + 1):
            layer_input_size = hidden_size
            layer_output_size = hidden_size
            layer_bias = hidden_bias

            is_first = i == 0
            is_last = i == num_hidden_layers

            if is_first:
                layer_input_size = input_size
            if is_last:
                layer_output_size = output_size

            layer = nn.Linear(layer_input_size, layer_output_size, bias=layer_bias)
            with torch.no_grad():
                layer.weight.normal_(std=math.sqrt(1.0 / layer_input_size))
                if layer_bias:
                    layer.bias.zero_()

            self.layers.append(layer)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)

            is_last = i == self.num_hidden_layers
            if not is_last:
                x = F.selu(x)
                x = self.dropout(x)

        if self.output_activation is not None:
            x = self.output_activation(x)

        return x


class TransformerFeedForwardBlock(nn.Module):
    """Standard pre-norm transformer FF block"""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        activation: Callable = nn.ReLU(),
        dropout: float = 0.1,
    ):
        super(TransformerFeedForwardBlock, self).__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = activation

        with torch.no_grad():
            self.linear1.weight.normal_(std=0.02)
            self.linear2.weight.normal_(std=0.02)

            self.linear1.bias.zero_()
            self.linear2.bias.zero_()

    def forward(self, x: torch.Tensor):
        residual = x

        x = self.layer_norm(x)
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.dropout(x)

        x = x + residual

        return x


class TransformerFeedForwardStack(nn.Module):
    def __init__(
        self,
        num_blocks: int,
        d_model: int,
        d_ff: int,
        d_input: int,
        d_head: int,
        activation: Callable = nn.ReLU(),
        dropout: float = 0.1,
    ):
        super(TransformerFeedForwardStack, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerFeedForwardBlock(d_model, d_ff, activation, dropout)
                for _ in range(num_blocks)
            ]
        )

        self.input_norm = nn.LayerNorm(d_input)
        self.final_norm = nn.LayerNorm(d_model)

        self.input = nn.Linear(d_input, d_model)
        self.head = nn.Linear(d_model, d_head)
        with torch.no_grad():
            self.head.weight.normal_(std=0.02)
            self.input.weight.normal_(std=0.02)

            self.head.bias.zero_()
            self.input.bias.zero_()

    def forward(self, x: torch.Tensor):
        x = self.input_norm(x)
        x = self.input(x)

        for layer in self.layers:
            x = layer(x)

        x = self.final_norm(x)
        x = self.head(x)

        return x


def gaussian_pdf(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor):
    return (
        1
        / (std * math.sqrt(2 * math.pi))
        * torch.exp(-0.5 * torch.square((x - mean) / std))
    )


def gaussian_log_pdf(x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor):
    return -0.5 * (
        log_var + math.log(2 * math.pi) + torch.square(x - mean) * torch.exp(-log_var)
    )


def gaussian_nll(x: torch.Tensor, mean: torch.Tensor, log_var: torch.Tensor):
    return 0.5 * (log_var + torch.square(x - mean) * torch.exp(-log_var))


def standard_normal_log_kl_div(mean: torch.Tensor, log_var: torch.Tensor):
    return 0.5 * (torch.exp(log_var) + torch.square(mean) - 1 - log_var)


def layer_norm(x: torch.Tensor, dim: int = -1):
    return (x - torch.mean(x, dim=dim, keepdim=True)) / torch.std(
        x, dim=dim, keepdim=True
    )


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


def main_dictionary_learning_single_token_ff():
    """Extremely simple dictionary learning of two languages with a set of symbols,
    where each sentence is only a single symbol long, and symbols follow a pre-generated
    frequency distribution. The frequency distribution is identical across languages,
    but their symbols are different (occupy different tokens in token space). Here, I
    use small feed-forward networks to implement everything, for fast training and
    debugging."""

    # Generators
    np_rng = np.random.default_rng(1234)
    torch.random.manual_seed(1234)
    torch.cuda.random.manual_seed(1234)

    # Hyperparameters

    ## Device
    device = "cuda"

    ## Language
    num_tokens_per_language = 16
    language_temperature = 2.0

    ## Network
    embedding_size = 512
    latent_size = 512
    hidden_size = 512
    encoder_hidden_layers = 4
    decoder_hidden_layers = 4
    classifier_hidden_layers = 4
    dropout = 0.0
    is_vae = False
    is_wgan = False
    normalize_embeddings = False
    group_classifier = True
    classifier_group_size = 2
    # sum_language_embeddings = True

    ## Optimizer
    learning_rate = 1e-4
    weight_decay = 1e-2
    betas = (0.5, 0.99)

    ## Training
    num_steps = 1000
    batch_size = 512

    ## Losses
    adversarial_weight = 10.0
    consistency_weight = 0.0
    latent_weight = 0.1
    gp_weight = 0.0

    # Generate language frequency distribution
    # token_distribution = torch.normal(
    #    mean=0.0,
    #    std=language_temperature,
    #    size=(num_tokens_per_language,),
    #    dtype=torch.float32,
    # )

    # token_distribution = torch.zeros((num_tokens_per_language,))

    # token_distribution = Categorical(logits=token_distribution)

    # Zipf's Law
    token_distribution = 1.0 / (torch.arange(num_tokens_per_language) + 1)
    token_distribution = Categorical(probs=token_distribution)

    print(token_distribution.entropy())
    print(Categorical(probs=torch.ones_like(token_distribution.probs)).entropy())

    language_distribution = torch.ones(2)
    language_distribution = Categorical(logits=language_distribution)

    # Build networks
    token_embeddings = nn.Embedding(num_tokens_per_language * 2, embedding_size).to(
        device
    )
    language_embeddings = nn.Embedding(2, embedding_size).to(device)

    # with torch.no_grad():
    #    token_embeddings.weight.normal_(std=1e-5)
    #    language_embeddings.weight.normal_(std=1e-5)

    # with torch.no_grad():
    #    token_embeddings.weight[:num_tokens_per_language] = token_embeddings.weight[
    #        num_tokens_per_language:
    #    ]

    """
    encoder = SelfNormalizingNetwork(
        embedding_size,
        (latent_size * 2) if is_vae else latent_size,
        hidden_size,
        encoder_hidden_layers,
        # hidden_activation=nn.GELU(),
        dropout=dropout,
    ).to(device)

    decoder = SelfNormalizingNetwork(
        embedding_size + latent_size,
        num_tokens_per_language * 2,
        hidden_size,
        decoder_hidden_layers,
        # hidden_activation=nn.GELU(),
        dropout=dropout,
    ).to(device)

    classifier = SelfNormalizingNetwork(
        latent_size,
        1,
        hidden_size,
        classifier_hidden_layers,
        # hidden_activation=nn.GELU(),
        dropout=dropout,
    ).to(device)
    """

    encoder = TransformerFeedForwardStack(
        encoder_hidden_layers,
        hidden_size,
        hidden_size * 4,
        embedding_size,
        (latent_size * 2) if is_vae else latent_size,
        activation=nn.GELU(),
        dropout=dropout,
    ).to(device)

    decoder = TransformerFeedForwardStack(
        decoder_hidden_layers,
        hidden_size,
        hidden_size * 4,
        embedding_size + latent_size,
        num_tokens_per_language * 2,
        activation=nn.GELU(),
        dropout=dropout,
    ).to(device)

    classifier = TransformerFeedForwardStack(
        classifier_hidden_layers,
        hidden_size,
        hidden_size * 4,
        latent_size * classifier_group_size,
        classifier_group_size,
        activation=nn.GELU(),
        dropout=dropout,
    ).to(device)

    # Make optimizers
    """
    autoencoder_params = itertools.chain(
        token_embeddings.parameters(),
        language_embeddings.parameters(),
        encoder.parameters(),
        decoder.parameters(),
    )
    classifier_params = classifier.parameters()

    autoencoder_optimizer = optim.AdamW(
        autoencoder_params, lr=learning_rate, weight_decay=weight_decay, betas=betas
    )
    classifier_optimizer = optim.AdamW(
        classifier_params, lr=learning_rate, weight_decay=weight_decay, betas=betas
    )
    """

    autoencoder_params = itertools.chain(
        token_embeddings.named_parameters(),
        language_embeddings.named_parameters(),
        encoder.named_parameters(),
        decoder.named_parameters(),
    )
    classifier_params = classifier.named_parameters()

    autoencoder_optimizer = optim.AdamW(
        make_param_groups(autoencoder_params, learning_rate, weight_decay), betas=betas
    )
    classifier_optimizer = optim.AdamW(
        make_param_groups(classifier_params, learning_rate, weight_decay), betas=betas
    )

    autoencoder_scheduler = optim.lr_scheduler.ExponentialLR(
        autoencoder_optimizer, gamma=1.0
    )

    classifier_scheduler = optim.lr_scheduler.ExponentialLR(
        classifier_optimizer, gamma=1.0
    )

    encoder.train()
    decoder.train()
    classifier.train()

    # Train
    pbar = trange(num_steps)
    for step in pbar:
        # adversarial_weight = step / num_steps
        # consistency_weight = 0.1 * step / num_steps

        # Generate a batch of data

        ## Sample languages and base tokens

        if group_classifier:
            batch_languages = torch.arange(2, device=device)
            batch_languages = batch_languages.repeat_interleave(batch_size // 2)
        else:
            batch_languages = language_distribution.sample((batch_size,)).to(device)
        batch_tokens = token_distribution.sample((batch_size,)).to(device)

        ## Shift tokens to match their respective language
        batch_tokens = batch_tokens + batch_languages * num_tokens_per_language

        # Step Autoencoder

        ## Autoencoder loss
        batch_token_embeddings = token_embeddings(batch_tokens)
        batch_language_embeddings = language_embeddings(batch_languages)

        if normalize_embeddings:
            batch_token_embeddings = layer_norm(batch_token_embeddings)
            batch_language_embeddings = layer_norm(batch_language_embeddings)

        batch_latents = encoder(batch_token_embeddings)

        if is_vae:
            batch_mean, batch_log_var = torch.split(batch_latents, latent_size, dim=1)
            batch_latents = batch_mean + torch.normal(
                0.0, 1.0, size=batch_mean.shape, device=device
            ) * torch.exp(0.5 * batch_log_var)

        decoder_inputs = torch.cat((batch_language_embeddings, batch_latents), dim=1)
        batch_reconstructions = decoder(decoder_inputs)

        autoencoder_loss = F.cross_entropy(batch_reconstructions, batch_tokens)

        if is_vae:
            latent_loss = torch.mean(
                standard_normal_log_kl_div(batch_mean, batch_log_var)
            )
            expected_relative_entropy = torch.mean(
                torch.sum(standard_normal_log_kl_div(batch_mean, batch_log_var), dim=1),
                dim=0,
            )
        else:
            latent_loss = 0.0
            expected_relative_entropy = 0.0

        ## Adversarial loss
        if group_classifier:
            batch_cls_latents = batch_latents.reshape(
                batch_size // classifier_group_size, -1
            )

            batch_cls_languages = batch_languages.reshape(
                batch_size // classifier_group_size, classifier_group_size
            )
        else:
            batch_cls_latents = batch_latents
            batch_cls_languages = batch_languages

        classifier.requires_grad_(False)
        batch_pred_classes = classifier(batch_cls_latents)
        classifier.requires_grad_(True)

        if not group_classifier:
            batch_pred_classes = batch_pred_classes[:, 0]

        if is_wgan:
            adversarial_loss = -torch.mean(
                batch_pred_classes * (batch_cls_languages * 2 - 1)
            )
        else:
            adversarial_loss = F.binary_cross_entropy_with_logits(
                batch_pred_classes, 1 - batch_cls_languages.float()
            )

        ## Consistency loss
        with torch.no_grad():
            batch_language_embeddings = language_embeddings(1 - batch_languages)
            if normalize_embeddings:
                batch_language_embeddings = layer_norm(batch_language_embeddings)

            decoder_inputs = torch.cat(
                (batch_language_embeddings, batch_latents), dim=1
            )
            batch_reconstructions = decoder(decoder_inputs)

            batch_translated_tokens = Categorical(logits=batch_reconstructions).sample()

        batch_token_embeddings = token_embeddings(batch_translated_tokens)
        batch_language_embeddings = language_embeddings(batch_languages)

        if normalize_embeddings:
            batch_token_embeddings = layer_norm(batch_token_embeddings)
            batch_language_embeddings = layer_norm(batch_language_embeddings)

        batch_latents = encoder(batch_token_embeddings)

        if is_vae:
            batch_mean, batch_log_var = torch.split(batch_latents, latent_size, dim=1)
            batch_latents = batch_mean + torch.normal(
                0.0, 1.0, size=batch_mean.shape, device=device
            ) * torch.exp(0.5 * batch_log_var)

        decoder_inputs = torch.cat((batch_language_embeddings, batch_latents), dim=1)
        batch_reconstructions = decoder(decoder_inputs)

        consistency_loss = F.cross_entropy(batch_reconstructions, batch_tokens)

        loss = (
            autoencoder_loss
            # + latent_loss * latent_weight
            + expected_relative_entropy * latent_weight
            + adversarial_loss * adversarial_weight
            + consistency_loss * consistency_weight
        )

        autoencoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        autoencoder_optimizer.step()
        autoencoder_scheduler.step()

        # Generate a batch of data

        ## Sample languages and base tokens
        batch_tokens = token_distribution.sample((batch_size,)).to(device)
        if group_classifier:
            batch_languages = torch.arange(2, device=device)
            batch_languages = batch_languages.repeat_interleave(batch_size // 2)
        else:
            batch_languages = language_distribution.sample((batch_size,)).to(device)

        ## Shift tokens to match their respective language
        batch_tokens = batch_tokens + batch_languages * num_tokens_per_language

        # Step classifier

        ## Classifier loss
        with torch.no_grad():
            batch_token_embeddings = token_embeddings(batch_tokens)
            if normalize_embeddings:
                batch_token_embeddings = layer_norm(batch_token_embeddings)
            batch_latents = encoder(batch_token_embeddings)

            if is_vae:
                batch_mean, batch_log_var = torch.split(
                    batch_latents, latent_size, dim=1
                )
                batch_latents = batch_mean + torch.normal(
                    0.0, 1.0, size=batch_mean.shape, device=device
                ) * torch.exp(0.5 * batch_log_var)

            if group_classifier:
                batch_latents = batch_latents.reshape(
                    batch_size // classifier_group_size, -1
                )

                batch_languages = batch_languages.reshape(
                    batch_size // classifier_group_size, classifier_group_size
                )

        batch_latents.requires_grad_(True)
        batch_pred_classes = classifier(batch_latents)

        if not group_classifier:
            batch_pred_classes = batch_pred_classes[:, 0]

        if is_wgan:
            classifier_loss = torch.mean(batch_pred_classes * (batch_languages * 2 - 1))
        else:
            classifier_loss = F.binary_cross_entropy_with_logits(
                batch_pred_classes, batch_languages.float()
            )

        classifier_gradients = torch.autograd.grad(
            batch_pred_classes,
            batch_latents,
            torch.ones_like(batch_pred_classes),
            retain_graph=True,
            create_graph=True,
            only_inputs=True,
        )[0]

        classifier_grad_penalty = torch.mean(
            torch.square(torch.norm(classifier_gradients, p=2, dim=1))
        )

        loss = classifier_loss + classifier_grad_penalty * gp_weight

        classifier_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        classifier_optimizer.step()
        classifier_scheduler.step()

        pbar.set_postfix_str(
            f"AE {autoencoder_loss:.2e} CLS {classifier_loss:.2e} ADV {adversarial_loss:.2e} GRAD {classifier_grad_penalty:.2e} Z {latent_loss:.2e} E[H] {expected_relative_entropy:.2e} nats C {consistency_loss:.2e}"
        )

    encoder.eval()
    decoder.eval()
    classifier.eval()

    # Generate evaluation data
    num_eval_samples = num_tokens_per_language * 2
    eval_tokens = torch.arange(num_eval_samples).to(device)
    eval_languages = torch.zeros(num_eval_samples, dtype=torch.long).to(device)
    eval_languages[num_tokens_per_language:] = 1

    # Disable gradients for evaluation
    with torch.no_grad():
        # Embed tokens and languages
        eval_token_embeddings = token_embeddings(eval_tokens)
        eval_language_embeddings = language_embeddings(eval_languages)

        if normalize_embeddings:
            eval_token_embeddings = layer_norm(eval_token_embeddings)
            eval_language_embeddings = layer_norm(eval_language_embeddings)

        # Encode tokens
        eval_latents = encoder(eval_token_embeddings)
        if is_vae:
            eval_mean, eval_log_var = torch.split(eval_latents, latent_size, dim=1)
            eval_latents = eval_mean

        # Decode tokens for each language
        eval_reconstructions = []
        for lang in range(2):
            lang_embedding = language_embeddings(torch.tensor([lang]).to(device))
            decoder_inputs = torch.cat(
                (lang_embedding.repeat(num_eval_samples, 1), eval_latents), dim=1
            )
            reconstructions = decoder(decoder_inputs)
            reconstructions = F.softmax(reconstructions, dim=1)
            eval_reconstructions.append(reconstructions)

    # Plot the logits of each decoded token for all input tokens and both languages
    fig, axs = plt.subplots(2, 1, figsize=(10, 8))
    for lang in range(2):
        ax = axs[lang]
        logits = eval_reconstructions[lang].detach().cpu().numpy()
        im = ax.imshow(logits, cmap="magma", aspect="auto")
        # ax.set_xticks(np.arange(num_eval_samples))
        # ax.set_yticks(np.arange(num_eval_samples))
        # ax.set_xticklabels(eval_tokens.cpu().numpy())
        # ax.set_yticklabels(eval_tokens.cpu().numpy())
        ax.set_xlabel("Decoded Token")
        ax.set_ylabel("Input Token")
        ax.set_title(f"Language {lang}")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_dictionary_learning_single_token_ff()
