# Multimodal Unsupervised Machine Translation

This repository and research project is currently under heavy development. It investigates methods from Unsupervised Machine Translation combined with other advancements to eventually achieve Universal Multi-Modal / Cross-Modal Unsupervised Translation between modalities. The high-level plan is to apply these methods to directly encode and translate DNN weights and activations, as well as for fully automated dictionary learning of SAE neurons and similar structures. It would then be interesting to investigate the structure of learned representations further. If successful, this could provide another high-level approach to addressing model interpretability, oversight, and control.

## Background

Fully Unsupervised Machine Translation without parallel data between languages (i.e., matched pairs of text with the same content but in different languages) has been around for a while. However, since the rise of LLMs, interest in this topic seems to have faded. Given the apparent difficulty of the task, these methods work surprisingly well. The main idea behind these methods is that languages and concepts have the same or very similar structure in some kind of latent or embedding space. Thus, translation without parallel data is possible by either enforcing shared latent spaces between languages or matching latent spaces via learned transformations.

This underlying hypothesis of similar distributional structure of language is known as the [Distributional Hypothesis](https://www.tandfonline.com/doi/abs/10.1080/00437956.1954.11659520). Applying a related idea directly to the structure of the physical world yields the [Natural Abstraction Hypothesis](https://www.alignmentforum.org/tag/natural-abstraction), which argues for the existence of natural abstractions within models of the physical world. Recent work on large language and vision models identified the [Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987): when scaling language and vision models, learned internal representations converge to the same statistical model. Multiple other reports show results that would naturally be expected if this class of hypotheses were true, such as model stitching—even between modalities—being surprisingly easy.

## Approach

The main idea is to combine various approaches for Unsupervised Machine Translation and Unsupervised Representation Learning with modern models (i.e., transformers). So far, the following has been tested and investigated:

### Model Types

- Decoder-only LLMs
  - EleutherAI's Pythia
- Encoder-Decoder LLMs
  - T5
  - Flan-T5
  - Models trained from scratch via lucidrain's x-transformers
- FFNs for prototyping on toy datasets

### Losses

- Autoencoder/Reconstruction Loss: Basic loss to get models to actually encode tokens to and decode tokens from a latent space.
- Consistency/Backtranslation Loss: Ensures consistency across translation cycles.
- Adversarial Loss: Removes information about the language in the latent space and guides cross-language latent space alignment via an adversarial generator-classifier setup.
- Variational/Information Bottleneck Loss: Forces the latent space to contain as little information as possible, filtering out language information and leading to latent spaces with clean, language-independent representations.

### Datasets

- One million sentences from Wikipedia in English and German (unpaired) from the Leipzig Corpora.
- 10k simple stories (paired but treated as unpaired).
- Synthetic data from random Hidden Markov Models with identical transition but different output matrices.

## Current Status

Initial tests were performed by fine-tuning Decoder-only and Encoder-Decoder LLMs from HuggingFace, which did not work as expected. Subsequently, a proof-of-concept was created using an extremely simple one-token synthetic language with FFNs, which was successful. Currently, transformers are being trained from scratch on much simpler data with positive initial results.

As a caveat, the repository is currently in a rough state. The code for training HuggingFace models and custom x-transformers is entirely separate. The training script for the custom x-transformers is large and needs rework to become the core framework of the repository.

The goal is to use this as a test harness to identify necessary training components/losses for good model performance, then scale the approach to more complex datasets upon finding a good configuration. Additionally, datasets of neuron activation patterns from SAEs will be incorporated to achieve automated dictionary learning/labeling of SAE neurons.

Future work will focus on developing custom encoders and decoders for NN weights and activations to support SAE and general NN investigation directly.

## To-Do

- Rework repository to V2, with the new training script promoted to the top and turned into a library/framework.
- Run test harness to identify necessary training components.
- Scale up to the Wikipedia sentence dataset.
- Write an initial report once a good model configuration is found.
- Experiment with more exotic approaches, such as translating between text and images or audio and images.
- If successful, write another short report.
- Explore even more exotic applications, such as translating between NN weights and text or SAE neurons and text.
  - Investigate other dictionary learning methods and adapt them to work with SAEs, like [this one](https://arxiv.org/abs/1710.04087).
- Perform an in-depth investigation of the learned representations.
- Explore new frontiers (here be dragons).