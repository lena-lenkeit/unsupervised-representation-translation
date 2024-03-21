import functools

import datasets
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments

from arae.datasets import make_wikisentence_dataset
from arae.trainers import EncoderDecoderLMForUnsupervisedTranslationTrainer


def main():
    model_path = "google/flan-t5-small"
    TokenizerType = T5Tokenizer
    ModelType = T5ForConditionalGeneration

    save_dir = "results/flan-t5-small-encdecv2"
    max_steps = 100000
    per_device_train_batch_size = 32
    learning_rate = 1e-4

    # Load model and tokenizer
    tokenizer = TokenizerType.from_pretrained(model_path)
    assert isinstance(tokenizer, TokenizerType)

    model = ModelType.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float32, dropout_rate=0.1
    )
    assert isinstance(model, ModelType)

    # Add label tokens
    tokenizer.add_tokens(["[LABEL_0]", "[LABEL_1]"], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    # Load datasets
    dataset1 = make_wikisentence_dataset("data/eng_wikipedia_2016_1M-sentences.txt")
    dataset2 = make_wikisentence_dataset("data/deu_wikipedia_2016_1M-sentences.txt")

    # Add labels to datasets
    # def add_labels_fn(row, *, labels: int):
    #    return {"text": f"[LABEL_{labels}]" + row["text"], "labels": labels}

    def add_labels_fn(row, *, labels: int):
        return {"labels": labels}

    dataset1 = dataset1.map(add_labels_fn, fn_kwargs={"labels": 0})
    dataset2 = dataset2.map(add_labels_fn, fn_kwargs={"labels": 1})

    # Construct final dataset
    dataset = datasets.interleave_datasets([dataset1, dataset2])
    dataset = dataset.map(
        lambda row: tokenizer(row["text"], max_length=64, truncation=True)
    )
    dataset = dataset.select_columns(["input_ids", "attention_mask", "labels"])

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=save_dir,
        max_steps=max_steps,
        per_device_train_batch_size=per_device_train_batch_size,
        learning_rate=learning_rate,
        lr_scheduler_type="constant",
        warmup_steps=0,
        save_total_limit=1,
        save_steps=100,
        logging_steps=10,
        logging_first_step=True,
        gradient_checkpointing=True,
        weight_decay=0.0,
        optim="adamw_bnb_8bit",
        remove_unused_columns=False,
    )

    trainer = EncoderDecoderLMForUnsupervisedTranslationTrainer(
        label0_token_id=tokenizer.convert_tokens_to_ids("[LABEL_0]"),  # type: ignore
        label1_token_id=tokenizer.convert_tokens_to_ids("[LABEL_1]"),  # type: ignore
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=training_args,
    )

    trainer.train()


if __name__ == "__main__":
    main()
