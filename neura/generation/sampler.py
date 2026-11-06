"""
Neura Smart Token Sampler
"""


import math
import random


class Sampler:

    def __init__(
        self,
        blocked_tokens=None
    ):

        self.blocked_tokens = blocked_tokens or []


    def softmax(self, values):

        maximum = max(values)

        exp_values = [
            math.exp(v - maximum)
            for v in values
        ]

        total = sum(exp_values)

        return [
            v / total
            for v in exp_values
        ]


    def apply_repetition_penalty(
        self,
        scores,
        previous_tokens,
        penalty=1.2
    ):

        if penalty <= 0:
            raise ValueError(
                "penalty must be > 0"
            )

        adjusted = list(scores)

        for token in previous_tokens:

            if token < len(adjusted):

                if adjusted[token] > 0:
                    adjusted[token] /= penalty
                else:
                    adjusted[token] *= penalty

        return adjusted


    def top_p_filter(
        self,
        candidates,
        probabilities,
        top_p=0.9
    ):

        pairs = list(
            zip(
                candidates,
                probabilities
            )
        )

        pairs.sort(
            key=lambda x: x[1],
            reverse=True
        )

        selected = []
        total = 0.0

        for token, probability in pairs:

            selected.append(
                (token, probability)
            )

            total += probability

            if total >= top_p:
                break

        return selected


    def sample(self, probabilities):

        r = random.random()

        cumulative = 0.0

        for index, probability in enumerate(probabilities):

            cumulative += probability

            if r <= cumulative:
                return index

        return len(probabilities) - 1


    def select(
        self,
        scores,
        vocab_size,
        temperature=1.0,
        top_k=None
    ):

        limit = min(
            len(scores),
            vocab_size
        )

        candidates = list(
            range(limit)
        )

        blocked = set(
            self.blocked_tokens
        )

        candidates = [
            i for i in candidates
            if i not in blocked
        ]

        if not candidates:
            raise ValueError(
                "no valid candidate tokens remain"
            )

        # Temperature scaling
        if temperature <= 0:
            raise ValueError(
                "temperature must be > 0"
            )

        scaled = {
            i: scores[i] / temperature
            for i in candidates
        }

        # Top-K filtering
        if top_k is not None:

            top_k = max(
                1,
                min(top_k, len(candidates))
            )

            candidates = sorted(
                candidates,
                key=lambda i: scaled[i],
                reverse=True
            )[:top_k]

        values = [
            scaled[i]
            for i in candidates
        ]

        probabilities = self.softmax(values)

        choice = self.sample(
            probabilities
        )

        return candidates[choice]
