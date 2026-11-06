"""
Neura AdamW Optimizer
"""

import math


class AdamW:

    def __init__(
        self,
        learning_rate=0.0001,
        weight_decay=0.01,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8
    ):

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay

        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon

        self.step_count = 0

        self.m = {}
        self.v = {}

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def _is_number(self, value):

        return isinstance(
            value,
            (int, float)
        )

    def _zeros_like(self, value):

        if self._is_number(value):
            return 0.0

        if isinstance(value, list):
            return [
                self._zeros_like(x)
                for x in value
            ]

        if isinstance(value, tuple):
            return tuple(
                self._zeros_like(x)
                for x in value
            )

        raise TypeError(
            f"Unsupported parameter type: "
            f"{type(value).__name__}"
        )

    def _update_value(
        self,
        name,
        parameter,
        gradient
    ):

        if self._is_number(parameter):

            if not self._is_number(gradient):
                raise TypeError(
                    f"Gradient type mismatch for "
                    f"{name}: "
                    f"parameter={type(parameter).__name__}, "
                    f"gradient={type(gradient).__name__}"
                )

            if name not in self.m:
                self.m[name] = 0.0
                self.v[name] = 0.0

            self.m[name] = (
                self.beta1 * self.m[name]
                + (1.0 - self.beta1) * gradient
            )

            self.v[name] = (
                self.beta2 * self.v[name]
                + (1.0 - self.beta2)
                * gradient
                * gradient
            )

            bias_correction_1 = (
                1.0
                - self.beta1 ** self.step_count
            )

            bias_correction_2 = (
                1.0
                - self.beta2 ** self.step_count
            )

            m_hat = (
                self.m[name]
                / bias_correction_1
            )

            v_hat = (
                self.v[name]
                / bias_correction_2
            )

            update = (
                m_hat
                / (
                    math.sqrt(v_hat)
                    + self.epsilon
                )
            )

            parameter = (
                parameter
                - self.learning_rate * update
            )

            parameter = (
                parameter
                - self.learning_rate
                * self.weight_decay
                * parameter
            )

            return parameter

        if isinstance(parameter, list):

            if not isinstance(gradient, (list, tuple)):
                raise TypeError(
                    f"Gradient type mismatch for "
                    f"{name}"
                )

            if len(parameter) != len(gradient):
                raise ValueError(
                    f"Shape mismatch for {name}: "
                    f"{len(parameter)} != "
                    f"{len(gradient)}"
                )

            return [
                self._update_value(
                    f"{name}.{i}",
                    parameter[i],
                    gradient[i]
                )
                for i in range(len(parameter))
            ]

        if isinstance(parameter, tuple):

            if not isinstance(gradient, (list, tuple)):
                raise TypeError(
                    f"Gradient type mismatch for "
                    f"{name}"
                )

            if len(parameter) != len(gradient):
                raise ValueError(
                    f"Shape mismatch for {name}"
                )

            return tuple(
                self._update_value(
                    f"{name}.{i}",
                    parameter[i],
                    gradient[i]
                )
                for i in range(len(parameter))
            )

        raise TypeError(
            f"Unsupported parameter type for "
            f"{name}: "
            f"{type(parameter).__name__}"
        )

    # ---------------------------------------------------------
    # Optimization
    # ---------------------------------------------------------

    def step(self, parameters):

        self.step_count += 1

        for name in parameters.names():

            value = parameters.get(name)
            gradient = parameters.gradient(name)

            if gradient is None:
                continue

            updated = self._update_value(
                name,
                value,
                gradient
            )

            parameters.update(
                name,
                updated
            )

    # ---------------------------------------------------------
    # Gradient reset
    # ---------------------------------------------------------

    def zero_grad(self, parameters):

        for name in parameters.names():

            value = parameters.get(name)

            parameters.set_gradient(
                name,
                self._zeros_like(value)
            )
