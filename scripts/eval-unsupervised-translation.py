import functools

import datasets
import torch
from transformers import (
    GenerationConfig,
    T5ForConditionalGeneration,
    T5Tokenizer,
    TrainingArguments,
)

from arae.datasets import make_wikisentence_dataset
from arae.models import T5ForUnsupervisedTranslation
from arae.trainers import EncoderDecoderLMForUnsupervisedTranslationTrainer


def main():
    model_path = "results/flan-t5-small-encdecv2-custom5-customloop"
    TokenizerType = T5Tokenizer
    ModelType = T5ForUnsupervisedTranslation

    # Load model and tokenizer
    # tokenizer = TokenizerType.from_pretrained(model_path)
    tokenizer = TokenizerType.from_pretrained("google/flan-t5-small")
    tokenizer.add_tokens(
        ["[LABEL_0]", "[LABEL_1]", "[CLS]", "[MASK]"], special_tokens=True
    )
    assert isinstance(tokenizer, TokenizerType)

    model = ModelType.from_pretrained(
        model_path, device_map="auto", torch_dtype=torch.float32, dropout_rate=0.1
    )
    assert isinstance(model, ModelType)

    print(tokenizer.convert_tokens_to_ids("[LABEL_0]"))

    model.train()

    while True:
        text_input = input("Encoder Input Text: ")

        token_inputs = tokenizer(text_input, return_tensors="pt").to(model.device)
        # decoder_inputs = tokenizer("[LABEL_1]", return_tensors="pt").to(model.device)

        config = GenerationConfig(
            do_sample=True,
            top_k=0,
            top_p=0.95,
        )
        token_outputs = model.generate(
            **token_inputs,
            decoder_start_token_id=tokenizer.convert_tokens_to_ids("[LABEL_0]"),
            # decoder_input_ids=decoder_inputs.input_ids[:, :1],
            generation_config=config
        )
        text_output = tokenizer.batch_decode(token_outputs)

        print(token_inputs)
        # print(decoder_inputs)
        print(token_outputs)
        print(text_output)


if __name__ == "__main__":
    main()
