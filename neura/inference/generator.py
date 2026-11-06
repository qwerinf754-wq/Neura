"""
Neura Text Generator
"""

from neura.inference.sampler import Sampler
from neura.inference.stopping import StoppingCriteria
from neura.inference.cache import KVCache


class TextGenerator:

    def __init__(
        self,
        model,
        tokenizer,
        eos_token=3
    ):
        self.model = model
        self.tokenizer = tokenizer

        self.cache = KVCache()

        self.stopper = StoppingCriteria(
            eos_token=eos_token
        )

    def reset(self):
        self.cache.clear()

    def _sample(
        self,
        logits,
        strategy="greedy",
        temperature=1.0,
        top_k=40,
        top_p=0.9
    ):

        if strategy == "greedy":
            return Sampler.greedy(logits)

    def generate(
        self,
        token_ids,
        max_new_tokens=32,
        strategy="greedy",
        temperature=1.0,
        top_k=40,
        top_p=0.9
    ):

        self.reset()

        generated = list(token_ids)

        while not self.stopper.should_stop(generated):

            logits_sequence = self.model.forward(generated)

            if not logits_sequence:
                break

            logits = logits_sequence[-1]

            next_token = self._sample(
                logits,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            generated.append(next_token)

            self.cache.add(
                next_token,
                self.model._last_hidden
            )

            if len(generated) >= len(token_ids) + max_new_tokens:
                break

        return generated


        if strategy == "temperature":
            return Sampler.temperature(
                logits,
                temperature
            )

        if strategy == "top_k":
            return Sampler.top_k(
                logits,
                top_k,
                temperature
            )

        if strategy == "top_p":
            return Sampler.top_p(
                logits,
                top_p,
                temperature
            )

        return Sampler.greedy(logits)

    def generate(
        self,
        token_ids,
        max_new_tokens=32,
        strategy="greedy",
        temperature=1.0,
        top_k=40,
        top_p=0.9
    ):

        self.reset()

        generated = list(token_ids)

        while not self.stopper.should_stop(generated):

            logits_sequence = self.model.forward(generated)

            if not logits_sequence:
                break

            logits = logits_sequence[-1]

            next_token = self._sample(
                logits,
                strategy=strategy,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p
            )

            generated.append(next_token)

            self.cache.add(
                next_token,
                self.model._last_hidden
            )

            if len(generated) >= len(token_ids) + max_new_tokens:
                break

        return generated

