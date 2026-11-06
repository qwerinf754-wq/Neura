"""
Neura Parameters Container
==========================

Global parameter registry for the Neura reference backend.

The registry supports:
- unique parameter names
- hierarchical namespaces
- gradients
- state dictionaries
- parameter counting
- gradient accumulation
"""

from __future__ import annotations

from copy import deepcopy
from numbers import Number
from typing import Any


class Parameters:
    """Central registry for trainable parameters."""

    def __init__(self):
        self.weights: dict[str, Any] = {}
        self.grads: dict[str, Any] = {}
        self.count = 0

    def clear(self) -> None:
        """Remove all registered parameters and gradients."""
        self.weights.clear()
        self.grads.clear()
        self.count = 0

    # -----------------------------------------------------------------
    # Registration
    # -----------------------------------------------------------------

    def add(self, name: str, value: Any) -> str:
        """Register a parameter.

        If the parameter already exists, its value is replaced without
        increasing the parameter count.
        """
        self._validate_name(name)

        if name not in self.weights:
            self.count += self._num_elements(value)

        self.weights[name] = value

        if name not in self.grads:
            self.grads[name] = self._zeros_like(value)

        return name

    def add_namespaced(
        self,
        namespace: str,
        name: str,
        value: Any,
    ) -> str:
        """Register a parameter inside a namespace."""
        full_name = self.join_name(namespace, name)
        return self.add(full_name, value)

    # -----------------------------------------------------------------
    # Access
    # -----------------------------------------------------------------

    def get(self, name: str) -> Any:
        """Return a parameter value."""
        return self.weights.get(name)

    def require(self, name: str) -> Any:
        """Return a parameter or raise KeyError."""
        if name not in self.weights:
            raise KeyError(f"Unknown parameter: {name}")
        return self.weights[name]

    def exists(self, name: str) -> bool:
        """Return True when a parameter exists."""
        return name in self.weights

    def gradient(self, name: str) -> Any:
        """Return a parameter gradient."""
        return self.grads.get(name)

    def require_gradient(self, name: str) -> Any:
        """Return a gradient or raise KeyError."""
        if name not in self.grads:
            raise KeyError(f"Unknown parameter: {name}")
        return self.grads[name]

    # -----------------------------------------------------------------
    # Gradients
    # -----------------------------------------------------------------

    def set_gradient(self, name: str, gradient: Any) -> None:
        """Replace a parameter gradient."""
        if name not in self.weights:
            raise KeyError(f"Unknown parameter: {name}")

        self.grads[name] = deepcopy(gradient)

    def add_gradient(self, name: str, gradient: Any) -> None:
        """Accumulate a gradient into an existing parameter."""
        if name not in self.weights:
            raise KeyError(f"Unknown parameter: {name}")

        current = self.grads.get(name)

        if current is None:
            self.grads[name] = deepcopy(gradient)
            return

        self.grads[name] = self._add(current, gradient)

    def zero_grad(self) -> None:
        """Reset all parameter gradients to zero."""
        for name, value in self.weights.items():
            self.grads[name] = self._zeros_like(value)

    # -----------------------------------------------------------------
    # Updates
    # -----------------------------------------------------------------

    def update(self, name: str, value: Any) -> None:
        """Replace a parameter value."""
        if name not in self.weights:
            raise KeyError(f"Unknown parameter: {name}")

        self.weights[name] = value

    # -----------------------------------------------------------------
    # Namespaces
    # -----------------------------------------------------------------

    @staticmethod
    def join_name(namespace: str, name: str) -> str:
        """Join namespace and parameter name."""
        namespace = namespace.strip(".")
        name = name.strip(".")

        if not name:
            raise ValueError("Parameter name cannot be empty.")

        if not namespace:
            return name

        return f"{namespace}.{name}"

    def names(self) -> list[str]:
        """Return all parameter names."""
        return list(self.weights.keys())

    def names_in_namespace(self, namespace: str) -> list[str]:
        """Return parameters belonging to a namespace."""
        prefix = namespace.strip(".") + "."

        return [
            name
            for name in self.weights
            if name.startswith(prefix)
        ]

    # -----------------------------------------------------------------
    # State
    # -----------------------------------------------------------------

    def state_dict(self) -> dict[str, Any]:
        """Return a deep copy of all parameter values."""
        return deepcopy(self.weights)

    def load_state_dict(self, state: dict[str, Any]) -> None:
        """Restore parameter values from a state dictionary."""
        if not isinstance(state, dict):
            raise TypeError("state must be a dictionary")

        # If the parameter registry is empty, rebuild it from the checkpoint.
        if not self.weights:
            self.weights = deepcopy(state)
            self.grads = {
                name: self._zeros_like(value)
                for name, value in self.weights.items()
            }
            self.count = sum(
                self._num_elements(value)
                for value in self.weights.values()
            )
            return

        state_names = set(state.keys())
        current_names = set(self.weights.keys())

        missing = current_names - state_names
        unexpected = state_names - current_names

        if missing:
            raise ValueError(
                f"Missing parameters: {sorted(missing)}"
            )

        if unexpected:
            raise ValueError(
                f"Unexpected parameters: {sorted(unexpected)}"
            )

        restored = deepcopy(state)

        for name in self.weights:
            self.weights[name] = restored[name]

        self.count = sum(
            self._num_elements(value)
            for value in self.weights.values()
        )

    def gradient_dict(self) -> dict[str, Any]:
        """Return a deep copy of all gradients."""
        return deepcopy(self.grads)

    # -----------------------------------------------------------------
    # Counting
    # -----------------------------------------------------------------

    def parameter_count(self) -> int:
        """Return total scalar parameter count."""
        return self.count

    @staticmethod
    def _num_elements(value: Any) -> int:
        """Count scalar elements recursively."""
        if isinstance(value, Number):
            return 1

        if isinstance(value, (list, tuple)):
            return sum(
                Parameters._num_elements(item)
                for item in value
            )

        try:
            return int(value.numel())
        except AttributeError:
            return 1

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    @staticmethod
    def _validate_name(name: str) -> None:
        if not isinstance(name, str):
            raise TypeError("Parameter name must be a string.")

        if not name.strip():
            raise ValueError("Parameter name cannot be empty.")

    @classmethod
    def _zeros_like(cls, value: Any) -> Any:
        if isinstance(value, list):
            return [
                cls._zeros_like(item)
                for item in value
            ]

        if isinstance(value, tuple):
            return tuple(
                cls._zeros_like(item)
                for item in value
            )

        if isinstance(value, Number):
            return 0.0

        try:
            return value * 0
        except Exception:
            return 0.0

    @classmethod
    def _add(cls, left: Any, right: Any) -> Any:
        if isinstance(left, Number):
            if not isinstance(right, Number):
                raise TypeError(
                    "Gradient structure mismatch."
                )

            return left + right

        if isinstance(left, list):
            if not isinstance(right, (list, tuple)):
                raise TypeError(
                    "Gradient structure mismatch."
                )

            if len(left) != len(right):
                raise ValueError(
                    "Gradient length mismatch."
                )

            return [
                cls._add(a, b)
                for a, b in zip(left, right)
            ]

        if isinstance(left, tuple):
            if not isinstance(right, (list, tuple)):
                raise TypeError(
                    "Gradient structure mismatch."
                )

            if len(left) != len(right):
                raise ValueError(
                    "Gradient length mismatch."
                )

            return tuple(
                cls._add(a, b)
                for a, b in zip(left, right)
            )

        try:
            return left + right
        except Exception as exc:
            raise TypeError(
                f"Unsupported gradient type: "
                f"{type(left).__name__}"
            ) from exc


PARAMETERS = Parameters()


if __name__ == "__main__":
    PARAMETERS.add("test.weight", [1.0, 2.0, 3.0])
    PARAMETERS.set_gradient(
        "test.weight",
        [0.1, 0.2, 0.3],
    )

    print("Parameters OK")
    print("Names:", PARAMETERS.names())
    print("Count:", PARAMETERS.parameter_count())
    print("Gradient:", PARAMETERS.gradient("test.weight"))
