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

    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, PreTrainedTokenizer)

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

    def assert_single(token_id: int | List[int]) -> int:
        if isinstance(token_id, list):
            raise TypeError(token_id)

        return token_id

    tokens = ARAETokens(
        task=TaskTokens(
            modeling=Token(
                id=assert_single(tokenizer.convert_tokens_to_ids(TASK_CLM_TOKEN)),
                text=TASK_CLM_TOKEN,
            ),
            encoding=Token(
                id=assert_single(tokenizer.convert_tokens_to_ids(TASK_ENCODING_TOKEN)),
                text=TASK_ENCODING_TOKEN,
            ),
            decoding=Token(
                id=assert_single(tokenizer.convert_tokens_to_ids(TASK_DECODING_TOKEN)),
                text=TASK_DECODING_TOKEN,
            ),
            classification=Token(
                id=assert_single(
                    tokenizer.convert_tokens_to_ids(TASK_CLASSIFICATION_TOKEN)
                ),
                text=TASK_CLASSIFICATION_TOKEN,
            ),
        ),
        placeholder=PlaceholderTokens(
            embedding=Token(
                id=assert_single(
                    tokenizer.convert_tokens_to_ids(EMBEDDING_PLACEHOLDER_TOKEN)
                ),
                text=EMBEDDING_PLACEHOLDER_TOKEN,
            ),
            label=Token(
                id=assert_single(
                    tokenizer.convert_tokens_to_ids(SCORE_PLACEHOLDER_TOKEN)
                ),
                text=SCORE_PLACEHOLDER_TOKEN,
            ),
        ),
        label=LabelTokens(
            a=Token(
                id=assert_single(tokenizer.convert_tokens_to_ids(CLASS_A_TOKEN)),
                text=CLASS_A_TOKEN,
            ),
            b=Token(
                id=assert_single(tokenizer.convert_tokens_to_ids(CLASS_B_TOKEN)),
                text=CLASS_B_TOKEN,
            ),
        ),
    )

    # Make dataset
    dataset = ARAEDataset(tokenizer, file_A, file_B, max_length, tokens)

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
    trainer = ARAETrainer(
        tokens=tokens,
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Start training
    trainer.train()
