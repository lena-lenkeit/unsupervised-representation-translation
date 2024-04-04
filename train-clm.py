import base64
import logging
from typing import Any, Dict

import datasets
import torch
import xor_cipher
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


def make_wikisentence_dataset(filepath: str):
    def clean_line(row: Dict[str, Any]) -> Dict[str, Any]:
        text: str = row["text"]
        text = text.split(maxsplit=1)[-1].strip()

        return {"text": text}

    dataset = datasets.load_dataset("text", split="train", data_files=filepath)
    dataset = dataset.shuffle().to_iterable_dataset().map(clean_line)

    return dataset


def make_instruct_translate_dataset(
    instruct_dataset_path: str, wikisentence_file_path: str, xor_key: str, tokenizer
):
    def encode_text(row: Dict[str, Any]) -> Dict[str, Any]:
        text: str = row["text"]
        text_xor = xor_cipher.cyclic_xor(text.encode("utf-8"), xor_key.encode("utf-8"))
        text_xor_b64 = base64.standard_b64encode(text_xor).decode("utf-8")

        return {"text": text_xor_b64}

    def messages_to_text(row: Dict[str, Any]) -> Dict[str, Any]:
        messages = row["messages"]
        text = []

        for message in messages:
            text.append(f"{message['role']}: {message['content']}")

        return {"text": "\n".join(text)}

    instruct_dataset = datasets.load_dataset(instruct_dataset_path, split="train")
    instruct_dataset = instruct_dataset.shuffle()
    instruct_dataset = instruct_dataset.to_iterable_dataset()
    instruct_dataset = instruct_dataset.select_columns("messages")
    instruct_dataset = instruct_dataset.map(messages_to_text, remove_columns="messages")

    wikisentence_dataset = make_wikisentence_dataset(wikisentence_file_path)
    wikisentence_crypt_dataset = wikisentence_dataset.map(encode_text)

    dataset = datasets.interleave_datasets(
        [instruct_dataset, wikisentence_dataset, wikisentence_crypt_dataset],
        [0.5, 0.25, 0.25],
    )

    dataset = ConstantLengthDataset(
        tokenizer,
        dataset,
        dataset_text_field="text",
        infinite=True,
        seq_length=128,
    )
    return dataset


def main():
    logger = logging.getLogger("transformers")
    logger.setLevel(logging.ERROR)

    model_path = "EleutherAI/pythia-160m"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    training_arguments = TrainingArguments(
        output_dir="./results/pythia-160m-clm",
        max_steps=100000,
        logging_steps=10,
        weight_decay=0.0,
        warmup_steps=0,
        save_total_limit=1,
        learning_rate=5e-5,
        per_device_train_batch_size=16,
        lr_scheduler_type="constant",
        gradient_checkpointing=True,
        optim="adamw_bnb_8bit",
    )

    dataset = make_instruct_translate_dataset(
        "allenai/tulu-v2-sft-mixture",
        "data/eng_wikipedia_2016_1M-sentences.txt",
        "A8g2",
        tokenizer,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_arguments,
        train_dataset=dataset,
        packing=True,
    )

    trainer.train()


if __name__ == "__main__":
    main()
