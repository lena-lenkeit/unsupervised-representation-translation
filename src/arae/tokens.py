from typing import NamedTuple


class Token(NamedTuple):
    id: int
    text: str


class TaskTokens(NamedTuple):
    modeling: Token
    encoding: Token
    decoding: Token
    classification: Token


class PlaceholderTokens(NamedTuple):
    embedding: Token
    label: Token


class LabelTokens(NamedTuple):
    a: Token
    b: Token


class ARAETokens(NamedTuple):
    task: TaskTokens
    placeholder: PlaceholderTokens
    label: LabelTokens
