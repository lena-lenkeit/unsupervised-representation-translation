# See text style transfer for a similar approach
# e.g. So Different Yet So Alike! Constrained Unsupervised Text Style Transfer https://aclanthology.org/2022.acl-long.32.pdf
# e.g. Cycle-Consistent Adversarial Autoencoders for Unsupervised Text Style Transfer https://arxiv.org/pdf/2010.00735.pdf

# This variant tests using similar approaches, but to perform translation based on monolingual text sources / unpaired text data
# In addition, the encoder, decoder, and classifier, will all be the same model
# If this works, I'll try extending this to other data types (image-text, weights-text, activations-text, image-audio, etc.)
# If that works, this might result in a powerful supervision technique, or more

from typing import Any, Dict, List, Optional, TypedDict

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)

from arae.collators import PyTreeCollator
from arae.datasets import ARAEDataset
from arae.tokens import ARAETokens, LabelTokens, PlaceholderTokens, TaskTokens, Token
from arae.trainers import ARAETrainer
from arae.utils import add_tokens_to_model

if __name__ == "__main__":
    # Hyperparameters

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

    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizer)

    # Modify to include extra tokens
    tokens = add_tokens_to_model(model, tokenizer)

    # Make dataset
    dataset = ARAEDataset(tokenizer, file_A, file_B, max_length, tokens)
    collator = PyTreeCollator()

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
    trainer = ARAETrainer(
        tokens=tokens,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()
