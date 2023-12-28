# See text style transfer for a similar approach
# e.g. So Different Yet So Alike! Constrained Unsupervised Text Style Transfer https://aclanthology.org/2022.acl-long.32.pdf
# e.g. Cycle-Consistent Adversarial Autoencoders for Unsupervised Text Style Transfer https://arxiv.org/pdf/2010.00735.pdf

# This variant tests using similar approaches, but to perform translation based on monolingual text sources / unpaired text data
# In addition, the encoder, decoder, and classifier, will all be the same model
# If this works, I'll try extending this to other data types (image-text, weights-text, activations-text, image-audio, etc.)
# If that works, this might result in a powerful supervision technique, or more

import random
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import optree
import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Bool, Float, Int, Int64
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainingArguments,
)


# Define a custom dataset
class CustomDataset(Dataset):
    def __init__(
        self, tokenizer: PreTrainedTokenizerBase, file_A, file_B, max_length: int
    ):
        with open(file_A, "r") as f:
            self.sentences_A = [line.strip() for line in f.readlines()]

        with open(file_B, "r") as f:
            self.sentences_B = [line.strip() for line in f.readlines()]

        self.dataset = self.sentences_A + self.sentences_B
        self.labels = [0] * len(self.sentences_A) + [1] * len(self.sentences_B)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx: int):
        text = self.dataset[idx]
        label = self.labels[idx]
        cls_token = CLASS_A_TOKEN if label == 0 else CLASS_B_TOKEN

        # Helper function to truncate token IDs and create attention mask
        def pad_tokens(token_ids: List[int], max_length: int, pad_token_id: int):
            attention_mask = [1] * len(token_ids)

            # Pad token_ids and attention mask to max_length
            padding_length = max_length - len(token_ids)
            token_ids = token_ids + ([pad_token_id] * padding_length)
            attention_mask = attention_mask + ([0] * padding_length)

            return token_ids, attention_mask

        # Function to prepare model inputs from token ids directly
        def prepare_model_inputs(
            prefix_token_ids: List[int],
            text_token_ids: List[int],
            postfix_token_ids: List[int],
            max_length: int,
            pad_token_id: int,
        ):
            special_length = len(prefix_token_ids) + len(postfix_token_ids)
            text_max_length = max_length - special_length

            text_token_ids = text_token_ids[:text_max_length]
            sequence_token_ids = prefix_token_ids + text_token_ids + postfix_token_ids
            assert len(sequence_token_ids) <= max_length, "Token sequence too long!"

            token_ids, attention_mask = pad_tokens(
                sequence_token_ids, max_length, pad_token_id
            )

            assert len(token_ids) == max_length, "Length mismatch"
            assert len(attention_mask) == max_length, "Length mismatch"

            # Convert lists to PyTorch tensors
            input_ids = torch.tensor(token_ids, dtype=torch.long)
            attention_mask = torch.tensor(attention_mask, dtype=torch.long)

            return {"input_ids": input_ids, "attention_mask": attention_mask}

        # Token IDs for special tokens, placeholders, and padding
        pad_token_id = self.tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        cls_token_id = self.tokenizer.convert_tokens_to_ids(cls_token)
        task_clm_token_id = self.tokenizer.convert_tokens_to_ids(TASK_CLM_TOKEN)
        task_encoding_token_id = self.tokenizer.convert_tokens_to_ids(
            TASK_ENCODING_TOKEN
        )
        task_decoding_token_id = self.tokenizer.convert_tokens_to_ids(
            TASK_DECODING_TOKEN
        )
        task_classification_token_id = self.tokenizer.convert_tokens_to_ids(
            TASK_CLASSIFICATION_TOKEN
        )
        embedding_placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            EMBEDDING_PLACEHOLDER_TOKEN
        )
        score_placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            SCORE_PLACEHOLDER_TOKEN
        )

        # Tokenize the text into token IDs
        text_token_ids = self.tokenizer(
            text, add_special_tokens=False
        ).input_ids  # Don't add special tokens

        # Tokenize CLM task
        clm_inputs = prepare_model_inputs(
            [task_clm_token_id], text_token_ids, [], self.max_length, pad_token_id
        )

        # Tokenize encoding task
        enc_inputs = prepare_model_inputs(
            [task_encoding_token_id, cls_token_id],
            text_token_ids,
            [embedding_placeholder_token_id],
            self.max_length,
            pad_token_id,
        )

        # Tokenize decoding task
        dec_inputs = prepare_model_inputs(
            [task_decoding_token_id, cls_token_id, embedding_placeholder_token_id],
            text_token_ids,
            [],
            self.max_length,
            pad_token_id,
        )

        # Tokenize classification task (no need to handle text, just tokens)
        cls_inputs = prepare_model_inputs(
            [
                task_classification_token_id,
                embedding_placeholder_token_id,
                score_placeholder_token_id,
            ],
            [],
            [],
            self.max_length,
            pad_token_id,
        )

        # Now you have clm_inputs, enc_inputs, dec_inputs, and cls_inputs dictionaries with 'input_ids' and 'attention_mask' as keys

        # Get indices of special tokens for replacement
        # enc_emb_pos = enc_inputs.input_ids

        return {
            "clm_inputs": clm_inputs,
            "enc_inputs": enc_inputs,
            "dec_inputs": dec_inputs,
            "cls_inputs": cls_inputs,
            "cls_id": label,
            "cls_token_id": cls_token_id,
        }


# Define a custom collator
class PyTreeCollator:
    """Collated leaves of a list of PyTrees with identical structure into batches"""

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        def map_fn(x: Any, *xs: Any) -> Any:
            if isinstance(x, torch.Tensor):
                y = torch.stack((x,) + xs, dim=0)
            elif isinstance(x, np.ndarray):
                y = np.stack((x,) + xs, axis=0)
                y = torch.from_numpy(y)
            elif isinstance(x, (float, int, bool)):
                y = np.asarray((x,) + xs)
                y = torch.from_numpy(y)
            else:
                raise TypeError(x)

            return y

        return optree.tree_map(map_fn, features[0], *features[1:])


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
        mask = mask[..., :-1].contiguous()
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


class LMInputs(TypedDict):
    input_ids: Int64[torch.Tensor, "batch sequence"]
    attention_mask: Int64[torch.Tensor, "batch sequence"]


class Batch(TypedDict):
    clm_inputs: LMInputs
    enc_inputs: LMInputs
    dec_inputs: LMInputs
    cls_inputs: LMInputs
    cls_id: Int64[torch.Tensor, "batch"]
    cls_token_id: Int64[torch.Tensor, "batch"]


class CustomTrainer(Trainer):
    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Batch,
        return_outputs: bool = False,
    ):
        """
        def find_token_pos(
            token_ids: torch.LongTensor, target_id: int
        ) -> torch.LongTensor:
            _, token_pos = torch.where(token_ids == target_id)
            return token_pos

        def gather_1d(
            input: torch.Tensor, dim: int, index: torch.LongTensor
        ) -> torch.Tensor:
            shape = input.shape
            shape[dim] = 1

            return torch.gather(input, dim, index.broadcast_to(shape))
        """

        def gather_from_tokens(
            key: Int64[torch.Tensor, "batch sequence"],
            values: Float[torch.Tensor, "batch sequence features"],
            query: int,
        ) -> Float[torch.Tensor, "batch features"]:
            batch, sequence, features = values.shape

            _, index = torch.where(key == query)
            assert index.shape[0] == values.shape[0], "Matched multiple tokens"

            # With gather (untested)
            # """
            index = index.reshape(batch, 1, 1)
            index = index.expand(batch, 1, features)
            out = torch.gather(values, 1, index)
            out = out.reshape(batch, features)
            # """

            # With indexing (untested)
            """
            out = values[:, pos]
            """

            # With select (1d gather / indexing)
            # out = torch.select(values, 1, index)

            return out

        def scatter_to_tokens(
            key: Int64[torch.Tensor, "batch sequence"],
            source: Float[torch.Tensor, "batch features"],
            values: Float[torch.Tensor, "batch sequence features"],
            query: int,
        ) -> Float[torch.Tensor, "batch sequence features"]:
            batch, sequence, features = values.shape

            _, index = torch.where(key == query)
            assert index.shape[0] == values.shape[0], "Matched multiple tokens"

            # With scatter (untested)
            # """
            index = index.reshape(batch, 1, 1)
            index = index.expand(batch, 1, features)
            source = source.reshape(batch, 1, features)
            out = torch.scatter(values, 1, index, source)
            # """

            # With select_scatter (1d scatter / indexing)
            # out = torch.select_scatter(values, source, 1, index)

            return out

        def get_cls_loss(
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            cls_ids: torch.Tensor,
            embeddings: Float[torch.Tensor, "batch features"],
        ) -> Float[torch.Tensor, ""]:
            input_embeddings = model.get_input_embeddings()(input_ids)
            input_embeddings = scatter_to_tokens(
                key=input_ids,
                source=embeddings,
                values=input_embeddings,
                query=embedding_placeholder_token_id,
            )

            outputs = model(
                inputs_embeds=input_embeddings,
                attention_mask=attention_mask,
            )

            logits = gather_from_tokens(
                key=input_ids,
                values=outputs["logits"],
                query=score_placeholder_token_id,
            )
            logits = logits[:, [cls_a_token_id, cls_b_token_id]]
            loss = F.cross_entropy(logits, cls_ids)

            return loss

        # Get token ids
        embedding_placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            EMBEDDING_PLACEHOLDER_TOKEN
        )
        score_placeholder_token_id = self.tokenizer.convert_tokens_to_ids(
            SCORE_PLACEHOLDER_TOKEN
        )

        cls_a_token_id = self.tokenizer.convert_tokens_to_ids(CLASS_A_TOKEN)
        cls_b_token_id = self.tokenizer.convert_tokens_to_ids(CLASS_B_TOKEN)

        # Causal language modelling loss
        clm_inputs = inputs.pop("clm_inputs")

        clm_outputs = model(
            input_ids=clm_inputs["input_ids"],
            attention_mask=clm_inputs["attention_mask"],
        )
        clm_loss = clm_loss_fn(
            logits=clm_outputs["logits"],
            labels=clm_inputs["input_ids"],
            mask=clm_inputs["attention_mask"],
        )

        # Autoencoder loss

        ## Encode
        enc_inputs = inputs.pop("enc_inputs")
        enc_outputs = model(
            input_ids=enc_inputs["input_ids"],
            attention_mask=enc_inputs["attention_mask"],
            output_hidden_states=True,
        )

        enc_embeddings = gather_from_tokens(
            key=enc_inputs["input_ids"],
            values=enc_outputs["hidden_states"][-1],
            query=embedding_placeholder_token_id,
        )

        ## Decode

        ### Construct decoding input
        dec_inputs = inputs.pop("dec_inputs")

        dec_input_embeddings = model.get_input_embeddings()(dec_inputs["input_ids"])
        dec_input_embeddings = scatter_to_tokens(
            key=dec_inputs["input_ids"],
            source=enc_embeddings,
            values=dec_input_embeddings,
            query=embedding_placeholder_token_id,
        )

        ## Loss
        dec_outputs = model(
            inputs_embeds=dec_input_embeddings,
            attention_mask=dec_inputs["attention_mask"],
        )

        ae_loss = clm_loss_fn(
            logits=dec_outputs["logits"],
            labels=dec_inputs["input_ids"],
            mask=dec_inputs["attention_mask"],
        )

        # Matching loss
        cls_inputs = inputs.pop("cls_inputs")

        model.requires_grad_(False)
        adv_loss = get_cls_loss(
            input_ids=cls_inputs["input_ids"],
            attention_mask=cls_inputs["attention_mask"],
            cls_ids=1 - inputs["cls_id"],
            embeddings=enc_embeddings,
        )
        model.requires_grad_(True)

        # Classification loss
        cls_loss = get_cls_loss(
            input_ids=cls_inputs["input_ids"],
            attention_mask=cls_inputs["attention_mask"],
            cls_ids=inputs["cls_id"],
            embeddings=enc_embeddings.detach(),
        )

        self.log_metrics(
            split="train",
            metrics={
                "clm_loss": clm_loss,
                "ae_loss": ae_loss,
                "adv_loss": adv_loss,
                "cls_loss": cls_loss,
            },
        )

        return clm_loss + ae_loss + adv_loss + cls_loss


if __name__ == "__main__":
    # model_name = "tiiuae/falcon-rw-1b"
    model_name = "EleutherAI/gpt-neo-125m"
    file_A = "data/eng_wikipedia_2016_1M-sentences.txt"
    file_B = "data/deu_wikipedia_2016_1M-sentences.txt"
    max_length = 64

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        model_name, device_map="auto", torch_dtype="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print(model.hf_device_map, model.dtype)
    print(tokenizer("ABC"))

    # Modify to include extra tokens
    TASK_CLM_TOKEN = "<|CLM-TASK|>"
    TASK_ENCODING_TOKEN = "<|ENC-TASK|>"
    TASK_DECODING_TOKEN = "<|DEC-TASK|>"
    TASK_CLASSIFICATION_TOKEN = "<|CLS-TASK|>"

    CLASS_A_TOKEN = "<|CLS-A|>"
    CLASS_B_TOKEN = "<|CLS-B|>"

    EMBEDDING_PLACEHOLDER_TOKEN = "<|EMB-PLACEHOLDER|>"
    SCORE_PLACEHOLDER_TOKEN = "<|SCORE-PLACEHOLDER|>"

    tokenizer.add_tokens(
        [
            TASK_CLM_TOKEN,
            TASK_ENCODING_TOKEN,
            TASK_DECODING_TOKEN,
            TASK_CLASSIFICATION_TOKEN,
            CLASS_A_TOKEN,
            CLASS_B_TOKEN,
            EMBEDDING_PLACEHOLDER_TOKEN,
            SCORE_PLACEHOLDER_TOKEN,
        ],
        special_tokens=True,
    )

    model.resize_token_embeddings(len(tokenizer))

    print(tokenizer(f"{TASK_ENCODING_TOKEN}{CLASS_A_TOKEN}ABC"))

    # Make dataset
    dataset = CustomDataset(tokenizer, file_A, file_B, max_length)
    print(dataset[0])
    print(dataset[-1])

    data_collator = PyTreeCollator()

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=2,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        remove_unused_columns=False,
    )

    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()
