import hydra
import torch
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments

from arae.collators import PyTreeCollator
from arae.datasets import ARAEDataset
from arae.trainers import ARAETrainer
from arae.utils import add_tokens_to_model


@hydra.main(config_path="conf", config_name="config", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.name, device_map="auto", torch_dtype=torch.float32
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.model.name)

    # Modify to include extra tokens
    tokens = add_tokens_to_model(model, tokenizer)

    # Make dataset
    dataset = ARAEDataset(
        tokenizer,
        cfg.data.file_A,
        cfg.data.file_B,
        cfg.data.max_length,
        tokens,
        cfg.data.num_cls_emb_tokens,
    )
    collator = PyTreeCollator()

    # Define training arguments
    training_args = TrainingArguments(**cfg.training)

    # Trainer
    trainer = ARAETrainer(
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
