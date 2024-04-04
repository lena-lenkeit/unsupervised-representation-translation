import base64
from typing import Any, Dict

import xor_cipher


def encode_text(text: str, xor_key: str) -> str:
    text_xor = xor_cipher.cyclic_xor(text.encode("utf-8"), xor_key.encode("utf-8"))
    text_xor_b64 = base64.standard_b64encode(text_xor).decode("utf-8")

    return text_xor_b64


def decode_text(text: str, xor_key: str) -> str:
    text_xor = base64.standard_b64decode(text)
    text = xor_cipher.cyclic_xor(text_xor, xor_key.encode("utf-8")).decode("utf-8")

    return text


def main():
    key = "A8g2"

    while True:
        text = input("Message: ")
        print(encode_text(text, key))
        print(decode_text(text, key))


if __name__ == "__main__":
    main()
