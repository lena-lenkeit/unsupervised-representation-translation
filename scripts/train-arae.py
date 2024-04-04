import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
    TrainingArguments,
)

from arae.collators import PyTreeCollator
from arae.datasets import ARAEDataset, EncDecDataset
from arae.models import ModelType
from arae.trainers import ARAETrainer, EncDecTrainer
from arae.utils import add_tokens_to_model


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Load model
    model = hydra.utils.instantiate(
        cfg.model, device_map="auto", torch_dtype=torch.float32
    )
    tokenizer = hydra.utils.instantiate(cfg.tokenizer)

    assert isinstance(model, PreTrainedModel)
    assert isinstance(tokenizer, (PreTrainedTokenizer, PreTrainedTokenizerFast))

    # Modify to include extra tokens
    tokens = add_tokens_to_model(model, tokenizer)

    # Make dataset
    dataset = hydra.utils.instantiate(cfg.dataset, tokenizer=tokenizer, tokens=tokens)
    collator = PyTreeCollator()

    # Define training arguments
    training_args = TrainingArguments(**cfg.training)

    # Trainer
    trainer = hydra.utils.instantiate(cfg.trainer)(
        tokens=tokens,
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    # Start training with optional resume_from_checkpoint
    trainer.train(resume_from_checkpoint=cfg.get("resume_from_checkpoint"))


if __name__ == "__main__":
    main()
