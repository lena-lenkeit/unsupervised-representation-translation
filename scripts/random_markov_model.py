"""Utilities for generating and sampling from random (hidden) Markov models."""

from dataclasses import dataclass

import numpy as np


def vectorized_categorical(pdf: np.ndarray, rng: np.random.Generator):
    """Samples from a vector of categorical distributions, specified by a matrix with
    probabilities summing to one in the rows. Based on inverse transform sampling."""

    uniform = rng.uniform(size=(pdf.shape[0], 1))
    cdf = np.cumsum(pdf, axis=1)
    idx = np.sum(cdf <= uniform, axis=1)

    return idx


@dataclass
class MarkovModel:
    transition_matrix: np.ndarray

    @property
    def num_states(self):
        return self.transition_matrix.shape[0]

    def next_state(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        next_state_probs = self.transition_matrix[state]
        return vectorized_categorical(next_state_probs, rng)


@dataclass
class HiddenMarkovModel(MarkovModel):
    output_matrix: np.ndarray

    @property
    def num_outputs(self):
        return self.output_matrix.shape[0]

    def sample_output(self, state: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        output_probs = self.output_matrix[state]
        return vectorized_categorical(output_probs, rng)


def softmax(x: np.ndarray, axis: int) -> np.ndarray:
    """Softmax along one axis."""

    return np.exp(x) / (np.sum(np.exp(x), axis=axis, keepdims=True))


def sample_matrix(
    num_states: int,
    *,
    num_outputs: int | None = None,
    temperature: float = 1.0,
    rng: np.random.Generator
) -> np.ndarray:
    """Samples a random transition or output matrix with normalized
    row-probabilities."""

    if num_outputs is None:
        num_outputs = num_states

    logits = rng.standard_normal(size=(num_states, num_outputs)) * temperature
    probs = softmax(logits, axis=1)

    return probs


def main():
    rng = np.random.default_rng(1234)

    num_parallel_streams = 1024 * 64
    num_prewarm_steps = 1024
    num_sample_steps = 128

    num_states = 64
    num_outputs = 8

    import string

    output_alphabet = string.ascii_uppercase

    transition_temperature = 2.0
    output_temperature = 2.0

    shared_transition_matrix = sample_matrix(
        num_states, temperature=transition_temperature, rng=rng
    )

    lang1_output_matrix = sample_matrix(
        num_states, num_outputs=num_outputs, temperature=output_temperature, rng=rng
    )

    lang2_output_matrix = sample_matrix(
        num_states, num_outputs=num_outputs, temperature=output_temperature, rng=rng
    )

    lang1_hmm = HiddenMarkovModel(shared_transition_matrix, lang1_output_matrix)
    lang2_hmm = HiddenMarkovModel(shared_transition_matrix, lang2_output_matrix)

    lang1_state = rng.integers(num_states, size=num_parallel_streams)
    lang2_state = rng.integers(num_states, size=num_parallel_streams)

    lang1_trajectories = []
    lang2_trajectories = []

    from tqdm.auto import trange

    for i in trange(num_prewarm_steps + num_sample_steps):
        if i >= num_prewarm_steps:
            lang1_trajectories.append(lang1_hmm.sample_output(lang1_state, rng))
            lang2_trajectories.append(lang2_hmm.sample_output(lang2_state, rng))

        lang1_state = lang1_hmm.next_state(lang1_state, rng)
        lang2_state = lang2_hmm.next_state(lang2_state, rng)

    lang1_trajectories = np.stack(lang1_trajectories, axis=1)
    lang2_trajectories = np.stack(lang2_trajectories, axis=1)

    with open("data/lang1_hmm.txt", mode="w") as f:
        f.write(
            "\n".join(
                [
                    "".join([output_alphabet[symbol] for symbol in trajectory])
                    for trajectory in lang1_trajectories
                ]
            )
        )

    with open("data/lang2_hmm.txt", mode="w") as f:
        f.write(
            "\n".join(
                [
                    "".join([output_alphabet[symbol] for symbol in trajectory])
                    for trajectory in lang2_trajectories
                ]
            )
        )


if __name__ == "__main__":
    main()
