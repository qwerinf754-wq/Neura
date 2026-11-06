"""
Neura Text Generator
"""

from .sampler import Sampler
from neura.inference.kv_cache import KVCache
from neura.tokenizer import VOCAB


class Generator:

    def __init__(
        self,
        model,
        sampler=None,
    ):

        self.model = model

        self.sampler = sampler or Sampler(
            blocked_tokens=[
                0,  # <PAD>
                1,  # <UNK>
                2,  # <BOS>
                *VOCAB.blocked_token_ids(),
            ]
        )

        self.kv_cache = KVCache()

    def _clear_cache(self):

        clear = getattr(
            self.model,
            "clear_cache",
            None,
        )

        if clear is not None:
            clear()

    def _forward_prefill(self, tokens):

        return self.model.forward(
            tokens,
            use_cache=True,
        )

    def _forward_incremental(self, token):

        return self.model.forward(
            [token],
            use_cache=True,
        )

    def _select(
        self,
        scores,
        tokens,
        temperature,
        top_k,
        repetition_penalty,
    ):

        if repetition_penalty:

            scores = self.sampler.apply_repetition_penalty(
                scores,
                tokens,
                repetition_penalty,
            )

        return self.sampler.select(
            scores,
            len(scores),
            temperature,
            top_k,
        )

    def generate(
        self,
        tokens,
        max_tokens=20,
        temperature=1.0,
        top_k=None,
        eos_token=3,
        repetition_penalty=None,
    ):

        if not isinstance(tokens, list):
            tokens = list(tokens)

        if not tokens:
            raise ValueError(
                "tokens cannot be empty"
            )

        if not isinstance(max_tokens, int):
            raise TypeError(
                "max_tokens must be an integer"
            )

        if max_tokens <= 0:
            raise ValueError(
                "max_tokens must be positive"
            )

        self._clear_cache()

        output = list(tokens)

        try:

            # -------------------------------------------------
            # Prefill
            #
            # Process the prompt exactly once.
            # -------------------------------------------------

            logits = self._forward_prefill(
                output
            )

            if not logits:
                raise RuntimeError(
                    "Model returned empty logits during prefill"
                )

            scores = logits[-1]

            # -------------------------------------------------
            # Autoregressive incremental decoding
            # -------------------------------------------------

            for _ in range(max_tokens):

                token = self._select(
                    scores,
                    output,
                    temperature,
                    top_k,
                    repetition_penalty,
                )

                output.append(token)

                if token == eos_token:
                    break

                # -------------------------------------------------
                # Incremental step
                #
                # Only the newly generated token enters the model.
                # Previous tokens remain inside the model KV cache.
                # -------------------------------------------------

                logits = self._forward_incremental(
                    token
                )

                if not logits:
                    raise RuntimeError(
                        "Model returned empty logits during incremental decoding"
                    )

                scores = logits[-1]

            return output

        finally:

            # Do not leak generation state into the next request.
            self._clear_cache()

