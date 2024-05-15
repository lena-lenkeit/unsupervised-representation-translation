# Multimodal Unsupervised Machine Translation
This is a repository and research project currently still under heavy development investigating methods from Unsupervised Machine Translation combined with other advancements to eventually achieve Universal Multi-Modal / Cross-Modal Unsupervised Translation between modalities. The high level plan is to see if this can be applied to directly encode and translate DNN weights and activations, as well as for fully-automated dictionary learning of SAE neurons and similar. It would then also be interesting to investigate the structure of learned representations further. If this works, this could be another high-level angle from which to attack model interpretability, oversight and control.

## Background
Fully Unsupervised Machine Translation without parallel data between languages (matched pairs of text with the same content but different languages) has been around for a while, though since the rise of LLMs, interest in this topic seems to have faded. However, given the apparent difficulty of the task, these methods work surprisingly well. The main idea behind these methods is that of language and concepts having the same or very similar structure in some kind of latent or embedding space, such that translation even without parallel data is possible by either enforcing shared latent spaces between languages or matching latent spaces via learned transformations.

This underlying hypothesis of similar distributional structure of language is known as the [Distributional Hypothesis](https://www.tandfonline.com/doi/abs/10.1080/00437956.1954.11659520). Applying a related idea directly to the structure of the physical world yields the [Natural Abstraction Hypothesis](https://www.alignmentforum.org/tag/natural-abstraction), arguing for the existence of natural abstractions within models of the physical world. Recent work on large language and vision models identified [The Platonic Representation Hypothesis](https://arxiv.org/abs/2405.07987): When scaling language and vision models, learned internal representations converge to the same statistical model. There are a multitude of other reports showing results one would naturally expect if this class of hypotheses were true, such as model stitching even between modalities being surprisingly easy.

## Approach
The main idea is to combine a variety of approaches for Unsupervised Machine Translation and Unsupervised Representation Learning with modern models (i.e. transformers). So far, I'm testing with and have investigated the following.

- Model Types
  - Decoder-only LLMs
    - EleutherAI's Pythia
  - Encoder-Decoder LLMs
    - T5
    - Flan-T5
    - Models trained from scratch via lucidrain's x-transformers
  - FFNs for prototyping on toy datasets
- Losses
  - Autoencoder / Reconstruction Loss (basic loss to get models to actually encode tokens to and decode tokens from a latent space)
  - Consistency / Backtranslation Loss (ensures consistency across translation cycles)
  - Adversarial Loss (to remove information about the language in the latent space / to guide cross-language latent space alignment, via an adversarial generator classifier setup)
  - Variational / Information Bottleneck Loss (to force the latent space to contain as little information as possible, which should also filter out language information and lead to latent spaces with clean, language-independent representations)
- Datasets
  - 1 Million sentences from Wikipedia in English and German (unpaired) from the Leipzig Corpora.
  - 10k simple stories (paired, but treated as unpaired)
  - Synthetic data from random hidden Markov models with identical transition but different output matrices.

## Current Status
I performed initial tests by finetuning Decoder-only and Encoder-Decoder LLMs from HugginFace. However, this didn't work at all. I then created a proof-of-concept on an extremely simple one-token synthetic language with FNNs, which worked. Extending the POC, I'm currently training Transformers from scratch on much simpler data, with positive initial results.

As such however, the repository is currently in a rough state. The code for training HuggingFace models and custom x-transformers is completely separate, but the training script for the custom x-transformers is getting quite large, and I'd like to rework the entire repository to be based around the new training script instead.

I'm looking to then use this as a test harness to see which training approaches / losses are actually necessary to get good models, and to then scale the entire approach up to more complicated datasets once I've found a good configuration. Additionally, I'd also like to incorporate datasets of neuron activation patterns from SAEs, to hopefully get automated dictionary learning / labeling of SAE neurons working.

Eventually, work would have to focus on also developing custom encoders and decoders for NN weights and activations, both to support SAEs and general investigation into NNs directly.

## To-Do

- Rework repository to V2, with new training script promoted to the top and turned into a library / framework.
- Run test harness to identify necessary training components.
- Scale up to wikisentence dataset.
- At this point, writing an initial report might be good.
- Try more exotic approaches, such as translating between images and text or audio and images.
- If this works, another short report might make sense.
- Even more exotic, translate between NN weights and text, or SAE neurons and text.
  - Also look at other dictionary learning methods for SAEs directly, like https://arxiv.org/abs/1710.04087.
- If this works, investigate the learned representations in-depth.
- Here be dragons.