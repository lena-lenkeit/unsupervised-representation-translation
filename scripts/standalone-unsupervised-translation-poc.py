"""A self-contained development and testing script for small Proofs-Of-Concept (POCs)
for Unsupervised Translation"""

import itertools
from typing import Callable

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

    # Hyperparameters

    ## Language
    num_tokens_per_language = 16
    language_temperature = 2.0

    ## Network
    embedding_size = 64
    latent_size = 64
    hidden_size = 128
    hidden_layers = 4

    ## Optimizer
    learning_rate = 1e-4
    weight_decay = 0.0

    ## Training
    num_steps = 10000
    batch_size = 512

    # Generate language frequency distribution
    token_distribution = torch.normal(
        mean=0.0,
        std=language_temperature,
        size=(num_tokens_per_language,),
        dtype=torch.float32,
    )
    token_distribution = Categorical(logits=token_distribution)

    language_distribution = torch.ones(2)
    language_distribution = Categorical(logits=language_distribution)

    # Build networks
    token_embeddings = nn.Embedding(num_tokens_per_language * 2, embedding_size)
    language_embeddings = nn.Embedding(2, embedding_size)

    encoder = FeedForwardNetwork(
        embedding_size,
        latent_size,
        hidden_size,
        hidden_layers,
        hidden_activation=nn.SiLU(),
    )

    decoder = FeedForwardNetwork(
        embedding_size + latent_size,
        num_tokens_per_language * 2,
        hidden_size,
        hidden_layers,
        hidden_activation=nn.SiLU(),
    )

    classifier = FeedForwardNetwork(
        latent_size, 1, hidden_size, hidden_layers, hidden_activation=nn.SiLU()
    )

    # Make optimizers
    autoencoder_params = itertools.chain(
        token_embeddings.parameters(),
        language_embeddings.parameters(),
        encoder.parameters(),
    )
    classifier_params = classifier.parameters()

    autoencoder_optimizer = optim.AdamW(
        autoencoder_params, lr=learning_rate, weight_decay=weight_decay
    )
    classifier_optimizer = optim.AdamW(
        classifier_params, lr=learning_rate, weight_decay=weight_decay
    )

    pbar = trange(num_steps)
    for step in pbar:
        # Generate a batch of data

        ## Sample languages and base tokens
        batch_languages = language_distribution.sample_n(batch_size)
        batch_tokens = token_distribution.sample_n(batch_size)

        ## Shift tokens to match their respective language
        batch_tokens = batch_tokens + batch_languages * num_tokens_per_language

        # Step Autoencoder

        ## Autoencoder loss
        batch_token_embeddings = token_embeddings(batch_tokens)
        batch_language_embeddings = language_embeddings(batch_languages)

        batch_latents = encoder(batch_token_embeddings)

        decoder_inputs = torch.cat((batch_language_embeddings, batch_latents), dim=1)
        batch_reconstructions = decoder(decoder_inputs)

        autoencoder_loss = F.cross_entropy(batch_reconstructions, batch_tokens)

        ## Adversarial loss
        batch_token_embeddings = token_embeddings(batch_tokens)
        batch_latents = encoder(batch_token_embeddings)

        classifier.requires_grad_(False)
        batch_pred_classes = classifier(batch_latents)
        classifier.requires_grad_(True)

        adversarial_loss = F.binary_cross_entropy_with_logits(
            batch_pred_classes[:, 0], 1 - batch_languages.float()
        )

        loss = autoencoder_loss + adversarial_loss

        autoencoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        autoencoder_optimizer.step()

        # Step classifier

        ## Classifier loss
        with torch.no_grad():
            batch_token_embeddings = token_embeddings(batch_tokens)
            batch_latents = encoder(batch_token_embeddings)
        batch_pred_classes = classifier(batch_latents)

        classifier_loss = F.binary_cross_entropy_with_logits(
            batch_pred_classes[:, 0], batch_languages.float()
        )

        classifier_optimizer.zero_grad(set_to_none=True)
        classifier_loss.backward()
        classifier_optimizer.step()

        pbar.set_postfix_str(
            f"AE {autoencoder_loss:.2e} CLS {classifier_loss:.2e} ADV {adversarial_loss:.2e}"
        )


if __name__ == "__main__":
    main_dictionary_learning_single_token_ff()
