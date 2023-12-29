# See text style transfer for a similar approach
# e.g. So Different Yet So Alike! Constrained Unsupervised Text Style Transfer https://aclanthology.org/2022.acl-long.32.pdf
# e.g. Cycle-Consistent Adversarial Autoencoders for Unsupervised Text Style Transfer https://arxiv.org/pdf/2010.00735.pdf

# This variant tests using similar approaches, but to perform translation based on monolingual text sources / unpaired text data
# In addition, the encoder, decoder, and classifier, will all be the same model
# If this works, I'll try extending this to other data types (image-text, weights-text, activations-text, image-audio, etc.)
# If that works, this might result in a powerful supervision technique, or more

from typing import Any, Dict, List, Optional, TypedDict

import torch
from torch.distributions import Categorical
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    TrainingArguments,
)
from transformers.modeling_outputs import CausalLMOutputWithPast

from arae.collators import PyTreeCollator
from arae.datasets import ARAEDataset, prepare_model_inputs
from arae.tokens import ARAETokens, LabelTokens, PlaceholderTokens, TaskTokens, Token
from arae.trainers import ARAETrainer
from arae.utils import add_tokens_to_model, gather_from_tokens, scatter_to_tokens

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

    # Enter eval loop
    while True:
        input_text = input("Input Text: ")
        output_class = input("Output Class ID: ")

        cls_token_id = tokens.label.a.id if output_class == 0 else tokens.label.b.id
        text_token_ids = tokenizer.encode(input_text, add_special_tokens=False)

        pad_token_id = tokenizer.pad_token_id
        if pad_token_id is None:
            pad_token_id = 0

        # Tokenize encoding task
        enc_inputs = prepare_model_inputs(
            [tokens.task.encoding.id, cls_token_id],
            text_token_ids,
            [tokens.placeholder.embedding.id],
            max_length,
            pad_token_id,
        )

        enc_input_ids = enc_inputs.input_ids.reshape(1, -1)
        enc_input_ids = torch.from_numpy(enc_input_ids).cuda()

        enc_attention_mask = enc_inputs.attention_mask.reshape(1, -1)
        enc_attention_mask = torch.from_numpy(enc_attention_mask).cuda()

        # Encode
        enc_outputs = model(
            input_ids=enc_input_ids,
            attention_mask=enc_attention_mask,
            output_hidden_states=True,
        )

        assert isinstance(enc_outputs, CausalLMOutputWithPast)
        assert isinstance(enc_outputs.hidden_states, tuple)

        enc_embeddings = gather_from_tokens(
            key=enc_input_ids,
            values=enc_outputs.hidden_states[-1],
            query=tokens.placeholder.embedding.id,
        )

        # Iteratively decode
        output_token_ids = []

        for i in range(max_length - 3):
            dec_input_ids = [
                tokens.task.decoding.id,
                cls_token_id,
                tokens.placeholder.embedding.id,
            ]

            dec_input_ids = torch.tensor([dec_input_ids + output_token_ids]).cuda()

            # Construct decoding input
            dec_input_embeddings = model.get_input_embeddings()(dec_input_ids)
            dec_input_embeddings = scatter_to_tokens(
                key=dec_input_ids,
                source=enc_embeddings,
                values=dec_input_embeddings,
                query=tokens.placeholder.embedding.id,
            )

            # Query next token
            dec_outputs = model(
                input_embeddings=dec_input_embeddings,
            )

            assert isinstance(dec_outputs, CausalLMOutputWithPast)

            token_dist = Categorical(logits=dec_outputs.logits[0, -1])
            next_token = token_dist.sample()

            output_token_ids.append(next_token.item())

        output_text = tokenizer.decode(output_token_ids)
        print(output_text)
