"""
Neura Cross Entropy Loss
"""

import math


class CrossEntropyLoss:

    def __init__(self):
        self.last_loss = 0.0
        self.last_gradients = []


    def _softmax(self, scores):

        if not scores:
            return []

        maximum = max(scores)

        values = [
            math.exp(
                value - maximum
            )
            for value in scores
        ]

        total = sum(values)

        if total == 0:
            return [
                0.0
                for _ in scores
            ]

        return [
            value / total
            for value in values
        ]


    def forward(
        self,
        prediction,
        target
    ):

        if not prediction:
            self.last_loss = 0.0
            return 0.0

        # Sequence causal language model mode
        if isinstance(
            prediction[0],
            list
        ) and isinstance(
            target,
            list
        ):

            if len(prediction) != len(target):
                raise ValueError(
                    "Prediction and target sequence "
                    "length mismatch."
                )

            total_loss = 0.0
            sequence_gradients = []

            for logits, token in zip(
                prediction,
                target
            ):

                probabilities = self._softmax(
                    logits
                )

                probability = max(
                    probabilities[token],
                    1e-12
                )

                total_loss += -math.log(
                    probability
                )

                grad = [
                    value
                    for value in probabilities
                ]

                if 0 <= token < len(grad):
                    grad[token] -= 1.0

                sequence_gradients.append(
                    grad
                )

            self.last_loss = total_loss
            self.last_gradients = sequence_gradients

            return self.last_loss


        # Single token mode
        probabilities = self._softmax(
            prediction
        )

        probability = probabilities[
            target
        ]

        probability = max(
            probability,
            1e-12
        )

        self.last_gradients = [
            value
            for value in probabilities
        ]

        if 0 <= target < len(
            self.last_gradients
        ):
            self.last_gradients[target] -= 1.0

        self.last_loss = -math.log(
            probability
        )

        return self.last_loss


    def gradients(self):
        return self.last_gradients
