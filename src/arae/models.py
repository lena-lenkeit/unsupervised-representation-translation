from enum import Enum

from transformers import T5ForConditionalGeneration


class ModelType(str, Enum):
    CAUSAL = "CAUSAL"
    T5LIKE_ENC_DEC = "T5LIKE_ENC_DEC"
