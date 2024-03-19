# On T5 models and masks

- When supplying only labels to the decoder, the decoder_input_ids are a right-shifted version of the labels, with the first token being the decoder start token (for T5, this is typically the padding token). Masked labels (of index -100, not contributing to the loss), are then also replaced with padding tokens in the decoder_input_ids.
- When not providing attention_mask (copied from transformers modeling_utils)
  - Provided a padding mask of dimensions [batch_size, seq_length]
    - if the model is a decoder, apply a causal mask in addition to the padding mask
    - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
  - However, there seems to be no mechanism automatically removing padding tokens from the attention_mask
    - For the decoder, this won't be an issue, since the loss isn't computed at the padding token positions anyway (due to how token_ids are derived from the labels)
    - For the encoder, this presents an issue
- As such, to ensure maximum compatibility across version changes, it might be better to fully derive all inputs, outputs, and masks for the encoder and decoder.