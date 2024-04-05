"""A self-contained development and testing script for small Proofs-Of-Concept (POCs)
for Unsupervised Translation"""

import itertools
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
    num_tokens_per_language = 4
    language_temperature = 2.0

    ## Network
    embedding_size = 16
    latent_size = 16
    hidden_size = 256
    encoder_hidden_layers = 1
    decoder_hidden_layers = 1
    classifier_hidden_layers = 1

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
        encoder_hidden_layers,
        hidden_activation=nn.GELU(),
    )

    decoder = FeedForwardNetwork(
        embedding_size + latent_size,
        num_tokens_per_language * 2,
        hidden_size,
        decoder_hidden_layers,
        hidden_activation=nn.GELU(),
    )

    classifier = FeedForwardNetwork(
        latent_size,
        1,
        hidden_size,
        classifier_hidden_layers,
        hidden_activation=nn.GELU(),
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

    # Train
    pbar = trange(num_steps)
    for step in pbar:
        # Generate a batch of data

        ## Sample languages and base tokens
        batch_languages = language_distribution.sample((batch_size,))
        batch_tokens = token_distribution.sample((batch_size,))

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
        classifier.requires_grad_(False)
        batch_pred_classes = classifier(batch_latents)
        classifier.requires_grad_(True)

        adversarial_loss = F.binary_cross_entropy_with_logits(
            batch_pred_classes[:, 0], 1 - batch_languages.float()
        )

        loss = autoencoder_loss + adversarial_loss * 1.0

        autoencoder_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        autoencoder_optimizer.step()

        # Step classifier

        ## Classifier loss
        with torch.no_grad():
            batch_token_embeddings = token_embeddings(batch_tokens)
            batch_latents = encoder(batch_token_embeddings)

        batch_latents.requires_grad_(True)
        batch_pred_classes = classifier(batch_latents)

        classifier_loss = F.binary_cross_entropy_with_logits(
            batch_pred_classes[:, 0], batch_languages.float()
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
            torch.square(torch.norm(classifier_gradients, p=2))
        )

        loss = classifier_loss + classifier_grad_penalty * 0.0

        classifier_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        classifier_optimizer.step()

        pbar.set_postfix_str(
            f"AE {autoencoder_loss:.2e} CLS {classifier_loss:.2e} ADV {adversarial_loss:.2e} GRAD {classifier_grad_penalty:.2e}"
        )

    # Generate evaluation data
    num_eval_samples = num_tokens_per_language * 2
    eval_tokens = torch.arange(num_eval_samples)
    eval_languages = torch.zeros(num_eval_samples, dtype=torch.long)
    eval_languages[num_tokens_per_language:] = 1

    # Disable gradients for evaluation
    with torch.no_grad():
        # Embed tokens and languages
        eval_token_embeddings = token_embeddings(eval_tokens)
        eval_language_embeddings = language_embeddings(eval_languages)

        # Encode tokens
        eval_latents = encoder(eval_token_embeddings)

        # Decode tokens for each language
        eval_reconstructions = []
        for lang in range(2):
            lang_embedding = language_embeddings(torch.tensor([lang]))
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
        logits = eval_reconstructions[lang].detach().numpy()
        im = ax.imshow(logits, cmap="magma", aspect="auto")
        ax.set_xticks(np.arange(num_eval_samples))
        ax.set_yticks(np.arange(num_eval_samples))
        ax.set_xticklabels(eval_tokens.numpy())
        ax.set_yticklabels(eval_tokens.numpy())
        ax.set_xlabel("Decoded Token")
        ax.set_ylabel("Input Token")
        ax.set_title(f"Language {lang}")
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main_dictionary_learning_single_token_ff()
