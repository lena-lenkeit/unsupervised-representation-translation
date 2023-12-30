from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Int64


# Custom loss functions
def clm_loss_fn(
    logits: Float[torch.Tensor, "batch sequence features"],
    labels: Int64[torch.Tensor, "batch sequence"],
    mask: Optional[Bool[torch.Tensor, "batch sequence"]] = None,
) -> Float[torch.Tensor, ""]:
    # Shift so that tokens < n predict n
    logits = logits[..., :-1, :].contiguous()
    labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = rearrange(logits, "b s f -> (b s) f")
    labels = rearrange(labels, "b s -> (b s)")

    # Calculate next-token loss
    if mask is None:
        loss = F.cross_entropy(logits, labels, reduction="mean")
    else:
        loss = F.cross_entropy(logits, labels, reduction="none")
        mask = mask[..., 1:].contiguous()
        mask = rearrange(mask, "b s -> (b s)")
        loss = torch.mean(loss[mask])

    return loss


def embedding_loss_fn(model_output, dataset_labels):
    embeddings = ...  # Logic to extract embeddings from `model_output`

    similarity_metric = ...  # Define a distance or similarity metric
    # For each pair of embeddings, record a similarity score
    target_scores = torch.ones((len(embeddings),), device=embeddings.device) * -1
    target_scores[torch.tensor(dataset_labels, device=embeddings.device) == 1] = 1
    # Use a loss function like MarginRankingLoss for instance, or write a custom one
    loss = ...  # Compute the loss based on similarity and target_scores

    return loss


def classification_loss_fn(model_output, dataset_labels):
    logits = model_output.logits[
        :, -1, :
    ]  # Assume we output a classification token at the last position

    # Prepare the labels for classification loss
    labels = torch.tensor(dataset_labels, dtype=torch.long, device=logits.device)

    # Binary classification loss, can use BCEWithLogitsLoss or CrossEntropyLoss for multi-class
    loss_fct = torch.nn.CrossEntropyLoss()  # Using two classes, hence CrossEntropy
    loss = loss_fct(logits, labels)

    return loss
