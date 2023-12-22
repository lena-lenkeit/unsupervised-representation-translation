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

        # Helper function to truncate token IDs and create attention mask
        def truncate_and_pad(token_ids: List[int], max_length: int, pad_token_id: int):
            attention_mask = [1] * len(token_ids)

            if len(token_ids) > max_length:
                # Truncate the token_ids and attention mask to max_length
                token_ids = token_ids[:max_length]
                attention_mask = attention_mask[:max_length]
            else:
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
            sequence_token_ids = prefix_token_ids + text_token_ids + postfix_token_ids
            truncated_token_ids, attention_mask = truncate_and_pad(
                sequence_token_ids, max_length, pad_token_id
            )

            # Convert lists to PyTorch tensors
            input_ids = torch.tensor([truncated_token_ids], dtype=torch.long)
            attention_mask = torch.tensor([attention_mask], dtype=torch.long)

            return {"input_ids": input_ids, "attention_mask": attention_mask}

        # Token IDs for special tokens, placeholders, and padding
        pad_token_id = self.tokenizer.pad_token_id
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
        return {
            "clm_inputs": clm_inputs,
            "enc_inputs": enc_inputs,
            "dec_inputs": dec_inputs,
            "cls_inputs": cls_inputs,
            "cls_token_id": cls_token_id,
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
def clm_loss_fn(outputs, labels) -> torch.Tensor:
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
        # Causal language modelling loss
        clm_inputs = inputs.pop("clm_inputs")

        clm_outputs = model(**clm_inputs)
        clm_loss = clm_loss_fn(clm_outputs, clm_inputs.input_ids)

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
