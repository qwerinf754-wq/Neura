"""
Neura Inference Sampler
Greedy / Temperature / Top-K / Top-P
"""

import math
import random


class Sampler:

    @staticmethod
    def softmax(logits, temperature=1.0):

        if temperature <= 0:
            temperature = 1.0

        scaled = [
            x / temperature
            for x in logits
        ]

        maximum = max(scaled)

        exps = [
            math.exp(x - maximum)
            for x in scaled
        ]

        total = sum(exps)

        if total == 0:
            return [
                0.0
                for _ in logits
            ]

        return [
            x / total
            for x in exps
        ]

    @staticmethod
    def greedy(logits):

        return max(
            range(len(logits)),
            key=lambda i: logits[i]
        )

    @staticmethod
    def temperature(logits, temperature=1.0):

        probs = Sampler.softmax(
            logits,
            temperature
        )

        return random.choices(
            range(len(probs)),
            weights=probs,
            k=1
        )[0]

    @staticmethod
    def top_k(
        logits,
        k=40,
        temperature=1.0
    ):

        probs = Sampler.softmax(
            logits,
            temperature
        )

        ranked = sorted(
            enumerate(probs),
            key=lambda x: x[1],
            reverse=True
        )[:k]

        ids = [
            i
            for i, _ in ranked
        ]

        weights = [
            p
            for _, p in ranked
        ]

        return random.choices(
            ids,
            weights=weights,
            k=1
        )[0]

    @staticmethod
    def top_p(
        logits,
        p=0.9,
        temperature=1.0
    ):

        probs = Sampler.softmax(
            logits,
            temperature
        )

        ranked = sorted(
            enumerate(probs),
            key=lambda x: x[1],
            reverse=True
        )

        ids = []
        weights = []

        cumulative = 0.0

        for idx, prob in ranked:

            ids.append(idx)
            weights.append(prob)

            cumulative += prob

            if cumulative >= p:
                break

        return random.choices(
            ids,
            weights=weights,
            k=1
        )[0]
