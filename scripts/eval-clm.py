import base64
import logging
from typing import Any, Dict

import datasets
import torch
import xor_cipher
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    TrainingArguments,
)
from trl import SFTTrainer
from trl.trainer import ConstantLengthDataset


@torch.no_grad()
def main():
    # logger = logging.getLogger("transformers")
    # logger.setLevel(logging.ERROR)

    model_path = "./results/pythia-160m-clm/checkpoint-27500"
    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    """
    generation_config = GenerationConfig(
        max_new_tokens=128,
        do_sample=True,
        top_p=0.95,
        top_k=0,
    )
    """

    generation_config = GenerationConfig(
        max_new_tokens=128,
        penalty_alpha=0.6,
        top_k=4,
    )

    while True:
        prompt = input("Prompt: ")
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
        outputs = model.generate(
            **inputs,
            max_new_tokens=128,
            penalty_alpha=0.6,
            top_k=4,
        )
        print(tokenizer.batch_decode(outputs))


if __name__ == "__main__":
    main()
