"""A self-contained development and testing script for small Proofs-Of-Concept (POCs)
for Unsupervised Translation"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def main_dictionary_learning_single_token_ff():
    """Extremely simple dictionary learning of two languages with a set of symbols,
    where each sentence is only a single symbol long, and symbols follow a pre-generated
    frequency distribution. The frequency distribution is identical across languages,
    but their symbols are different (occupy different tokens in token space). Here, I
    use small feed-forward networks to implement everything, for fast training and
    debugging."""

    np_rng = np.random.default_rng(1234)
    num_tokens_per_language = 16

    freq_distribution = np_rng.standard_normal(
        size=num_tokens_per_language, dtype=np.float32
    )


if __name__ == "__main__":
    main_dictionary_learning_simple()
