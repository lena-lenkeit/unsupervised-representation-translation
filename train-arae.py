# See text style transfer for a similar approach
# e.g. So Different Yet So Alike! Constrained Unsupervised Text Style Transfer https://aclanthology.org/2022.acl-long.32.pdf
# e.g. Cycle-Consistent Adversarial Autoencoders for Unsupervised Text Style Transfer https://arxiv.org/pdf/2010.00735.pdf

# This variant tests using similar approaches, but to perform translation based on monolingual text sources / unpaired text data
# In addition, the encoder, decoder, and classifier, will all be the same model
# If this works, I'll try extending this to other data types (image-text, weights-text, activations-text, image-audio, etc.)
# If that works, this might result in a powerful supervision technique, or more

import random
from typing import List

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
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

        # A function to truncate tokens to a specific max_length while keeping special tokens
        def truncate_tokens(
            tokens: List[int], max_length: int, num_special_tokens: int
        ):
            if len(tokens) > max_length - num_special_tokens:
                # Truncate the tokens to max_length minus the space required for special tokens
                return tokens[: max_length - num_special_tokens]
            return tokens

        # Tokenize the text to get its tokens
        text_tokens = self.tokenizer.tokenize(text)

        # Helper to prepare inputs with appropriate truncation
        def prepare_inputs(prefix: str, text_tokens: List[int], postfix: str = ""):
            tokens = (
                self.tokenizer.tokenize(prefix)
                + truncate_tokens(
                    text_tokens,
                    self.max_length,
                    len(self.tokenizer.tokenize(prefix + postfix)),
                )
                + self.tokenizer.tokenize(postfix)
            )
            return self.tokenizer.convert_tokens_to_string(tokens)

        # Tokenize CLM task
        clm_text = prepare_inputs(TASK_CLM_TOKEN, text_tokens)
        clm_inputs = self.tokenizer(
            clm_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # Tokenize encoding task
        enc_text = prepare_inputs(
            TASK_ENCODING_TOKEN + cls_token, text_tokens, EMBEDDING_PLACEHOLDER_TOKEN
        )
        enc_inputs = self.tokenizer(
            enc_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # Tokenize decoding task
        dec_text = prepare_inputs(
            TASK_DECODING_TOKEN + cls_token + EMBEDDING_PLACEHOLDER_TOKEN, text_tokens
        )
        dec_inputs = self.tokenizer(
            dec_text,
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        # For the classification task, we don't have a variable length text input so we can tokenize it directly
        cls_inputs = self.tokenizer(
            f"{TASK_CLASSIFICATION_TOKEN}{EMBEDDING_PLACEHOLDER_TOKEN}{SCORE_PLACEHOLDER_TOKEN}",
            return_tensors="pt",
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
        )

        return {
            "clm_inputs": clm_inputs,
            "enc_inputs": enc_inputs,
            "dec_inputs": dec_inputs,
            "cls_inputs": cls_inputs,
            "cls_token": cls_token,
        }


# Define a custom collator
class MyDataCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def collate_batch(self, batch):
        # Implement the collation logic considering the complex objectives
        # and multiple task tokens
        pass


# Custom loss functions
def clm_loss_fn(outputs, labels):
    loss_fct = CrossEntropyLoss()
    lm_logits = outputs.logits

    # Shift so that tokens < n predict n
    shift_logits = lm_logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
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


class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        input_token_ids = inputs.pop("input_ids")
        input_class_labels = inputs.pop("input_labels")

        # Causal language modelling loss
        clm_outputs = model(input_ids=input_token_ids)
        clm_loss = clm_loss_fn(clm_outputs, input_token_ids)

        # Autoencoder loss

        ## Encode
        enc_inputs = inputs.pop("enc_inputs")
        enc_outputs = model(**enc_inputs)

        emb_token_idx = inputs.pop("emb_token_idx")
        enc_emb_values = torch.gather(inputs=enc_outputs, dim=1, index=emb_token_idx)

        ## Decode
        # dec_task_token_id = self.tokenizer.convert_tokens_to_ids(TASK_DECODING_TOKEN)

        ### Construct decoding input
        dec_inputs = inputs.pop("dec_inputs")
        dec_input_ids = dec_inputs.input_ids
        dec_input_embeddings = model.embedding(dec_input_ids)

        ## Loss

        # Matching loss

        # Classification loss


if __name__ == "__main__":
    # model_name = "tiiuae/falcon-rw-1b"
    model_name = "EleutherAI/gpt-neo-125m"
    file_A = "path_to_dataset_A.txt"
    file_B = "path_to_dataset_B.txt"
    max_length = 128

    # Load model
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

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

    # Make dataset
    dataset = CustomDataset(tokenizer, file_A, file_B, max_length)
    data_collator = MyDataCollator(tokenizer)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        num_train_epochs=3,
        per_device_train_batch_size=8,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
    )

    # Trainer
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator.collate_batch,
        train_dataset=dataset,
        tokenizer=tokenizer,
        compute_loss=None,  # Define your custom loss function logic in the Trainer instead
    )

    # Start training
    trainer.train()
