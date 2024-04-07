"""A self-contained development and testing script for small Proofs-Of-Concept (POCs)
for Unsupervised Translation"""

import itertools
import math
from typing import Callable

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
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.hidden_bias = hidden_bias
        self.output_bias = output_bias
        self.output_activation = output_activation

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

        if self.output_activation is not None:
            x = self.output_activation(x)

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
    num_tokens_per_language = 4
    language_temperature = 2.0

    ## Network
    embedding_size = 64
    latent_size = 64
    hidden_size = 64
    encoder_hidden_layers = 2
    decoder_hidden_layers = 2
    classifier_hidden_layers = 4

    ## Optimizer
    learning_rate = 1e-3
    weight_decay = 0.0
    betas = (0.0, 0.99)

    ## Training
    num_steps = 10000
    batch_size = 4096

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

    encoder = SelfNormalizingNetwork(
        embedding_size,
        latent_size * 2,
        hidden_size,
        encoder_hidden_layers,
        # hidden_activation=nn.GELU(),
    ).to(device)

    decoder = SelfNormalizingNetwork(
        embedding_size + latent_size,
        num_tokens_per_language * 2,
        hidden_size,
        decoder_hidden_layers,
        # hidden_activation=nn.GELU(),
    ).to(device)

    classifier = SelfNormalizingNetwork(
        latent_size,
        1,
        hidden_size,
        classifier_hidden_layers,
        # hidden_activation=nn.GELU(),
    ).to(device)

    # Make optimizers
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

    autoencoder_scheduler = optim.lr_scheduler.ExponentialLR(
        autoencoder_optimizer, gamma=0.9999
    )

    classifier_scheduler = optim.lr_scheduler.ExponentialLR(
        classifier_optimizer, gamma=0.9999
    )

    # Train
    pbar = trange(num_steps)
    for step in pbar:
        # Generate a batch of data

        ## Sample languages and base tokens
        batch_languages = language_distribution.sample((batch_size,)).to(device)
        batch_tokens = token_distribution.sample((batch_size,)).to(device)

        ## Shift tokens to match their respective language
        batch_tokens = batch_tokens + batch_languages * num_tokens_per_language

        # Step Autoencoder

        ## Autoencoder loss
        batch_token_embeddings = token_embeddings(batch_tokens)
        batch_language_embeddings = language_embeddings(batch_languages)

        batch_latents = encoder(batch_token_embeddings)

        batch_mean, batch_log_var = torch.split(batch_latents, latent_size, dim=1)
        batch_latents = batch_mean + torch.normal(
            0.0, 1.0, size=batch_mean.shape, device=device
        ) * torch.exp(0.5 * batch_log_var)

        decoder_inputs = torch.cat((batch_language_embeddings, batch_latents), dim=1)
        batch_reconstructions = decoder(decoder_inputs)

        autoencoder_loss = F.cross_entropy(batch_reconstructions, batch_tokens)

        latent_loss = torch.mean(standard_normal_log_kl_div(batch_mean, batch_log_var))
        expected_relative_entropy = torch.mean(
            torch.sum(standard_normal_log_kl_div(batch_mean, batch_log_var), dim=1),
            dim=0,
        )

        ## Adversarial loss
        classifier.requires_grad_(False)
        batch_pred_classes = classifier(batch_latents)
        classifier.requires_grad_(True)

        adversarial_loss = F.binary_cross_entropy_with_logits(
            batch_pred_classes[:, 0], 1 - batch_languages.float()
        )

        # adversarial_loss = -torch.mean(
        #    batch_pred_classes[:, 0] * (batch_languages * 2 - 1)
        # )

        ## Consistency loss
        with torch.no_grad():
            batch_language_embeddings = language_embeddings(1 - batch_languages)

            decoder_inputs = torch.cat(
                (batch_language_embeddings, batch_latents), dim=1
            )
            batch_reconstructions = decoder(decoder_inputs)

            batch_translated_tokens = Categorical(logits=batch_reconstructions).sample()

        batch_token_embeddings = token_embeddings(batch_translated_tokens)
        batch_language_embeddings = language_embeddings(batch_languages)

        batch_latents = encoder(batch_token_embeddings)

        batch_mean, batch_log_var = torch.split(batch_latents, latent_size, dim=1)
        batch_latents = batch_mean + torch.normal(
            0.0, 1.0, size=batch_mean.shape, device=device
        ) * torch.exp(0.5 * batch_log_var)

        decoder_inputs = torch.cat((batch_language_embeddings, batch_latents), dim=1)
        batch_reconstructions = decoder(decoder_inputs)

        consistency_loss = F.cross_entropy(batch_reconstructions, batch_tokens)

        loss = (
            autoencoder_loss
            + latent_loss * 1.0
            + adversarial_loss * 1.0
            + consistency_loss * 1.0
        )

        autoencoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        autoencoder_optimizer.step()
        autoencoder_scheduler.step()

        # Generate a batch of data

        ## Sample languages and base tokens
        batch_languages = language_distribution.sample((batch_size,)).to(device)
        batch_tokens = token_distribution.sample((batch_size,)).to(device)

        ## Shift tokens to match their respective language
        batch_tokens = batch_tokens + batch_languages * num_tokens_per_language

        # Step classifier

        ## Classifier loss
        with torch.no_grad():
            batch_token_embeddings = token_embeddings(batch_tokens)
            batch_latents = encoder(batch_token_embeddings)

            batch_mean, batch_log_var = torch.split(batch_latents, latent_size, dim=1)
            batch_latents = batch_mean + torch.normal(
                0.0, 1.0, size=batch_mean.shape, device=device
            ) * torch.exp(0.5 * batch_log_var)

        batch_latents.requires_grad_(True)
        batch_pred_classes = classifier(batch_latents)

        classifier_loss = F.binary_cross_entropy_with_logits(
            batch_pred_classes[:, 0], batch_languages.float()
        )

        # classifier_loss = torch.mean(
        #    batch_pred_classes[:, 0] * (batch_languages * 2 - 1)
        # )

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

        loss = classifier_loss + classifier_grad_penalty * 1.0

        classifier_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        classifier_optimizer.step()
        classifier_scheduler.step()

        pbar.set_postfix_str(
            f"AE {autoencoder_loss:.2e} CLS {classifier_loss:.2e} ADV {adversarial_loss:.2e} GRAD {classifier_grad_penalty:.2e} Z {latent_loss:.2e} E[H] {expected_relative_entropy:.2e} bits C {consistency_loss:.2e}"
        )

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

        # Encode tokens
        eval_latents = encoder(eval_token_embeddings)
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
