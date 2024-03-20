import functools

import datasets
import torch
from transformers import T5ForConditionalGeneration, T5Tokenizer, TrainingArguments

from arae.datasets import make_wikisentence_dataset
from arae.trainers import EncoderDecoderLMForUnsupervisedTranslationTrainer


def main():
    model_path = "results/flan-t5-small-encdecv2/checkpoint-500"
    TokenizerType = T5Tokenizer
    ModelType = T5ForConditionalGeneration

    # Load model and tokenizer
    tokenizer = TokenizerType.from_pretrained(model_path)
    assert isinstance(tokenizer, TokenizerType)

    model = ModelType.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float32, dropout_rate=0.1
    )
    assert isinstance(model, ModelType)

    while True:
        text_input = input("Encoder Input Text: ")

        token_inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        token_outputs = model.generate(**token_inputs)
        text_output = tokenizer.batch_decode(token_outputs)

        print(text_output)


if __name__ == "__main__":
    main()
