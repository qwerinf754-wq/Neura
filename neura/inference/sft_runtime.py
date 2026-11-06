"""
Neura SFT Runtime

Runtime dedicated to loading and running Neura SFT checkpoints.

Responsibilities:
    - Load SFT checkpoint.
    - Restore checkpoint vocabulary.
    - Build NeuraTorchModel.
    - Strictly load model weights.
    - Encode prompts using the restored tokenizer.
    - Perform autoregressive generation.
    - Stop on EOS or context limit.
    - Decode generated tokens.

This runtime intentionally does NOT depend on:
    - InferenceEngine
    - Generator
    - KV cache
    - ChatService

The current NeuraTorchModel does not expose the cache interface
required by the newer inference engine, so generation here uses
full-sequence forward passes for correctness and isolation.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from neura.model.torch_model import NeuraTorchModel
from neura.tokenizer.tokenizer import Tokenizer
from neura.tokenizer.vocabulary import Vocabulary
from neura.training.sft_checkpoint import SFTCheckpoint


@dataclass(slots=True)
class SFTGenerationResult:
    """Result returned by the SFT runtime."""

    prompt_tokens: list[int]
    generated_tokens: list[int]
    finish_reason: str
    tokenizer: Tokenizer | None = None

    @property
    def tokens(self) -> list[int]:
        """Return prompt + generated tokens."""
        return self.prompt_tokens + self.generated_tokens

    @property
    def text(self) -> str:
        """Return decoded generated tokens only."""
        if self.tokenizer is None:
            raise RuntimeError("generation result has no tokenizer")
        return self.tokenizer.decode(self.generated_tokens)


class SFTRuntime:
    """
    Runtime for Neura SFT checkpoints.

    The checkpoint is loaded once during initialization.

    Parameters
    ----------
    checkpoint_path:
        Path to an SFT checkpoint.
    device:
        Torch device. Defaults to CPU.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str | torch.device = "cpu",
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.device = torch.device(device)

        self.checkpoint: dict[str, Any] | None = None
        self.model: NeuraTorchModel | None = None

        self.model_config: dict[str, Any] | None = None
        self.vocabulary: dict[str, Any] | None = None
        self.tokenizer: Tokenizer | None = None

        self._loaded = False

        self.load()

    # ------------------------------------------------------------------
    # LOAD
    # ------------------------------------------------------------------

    def load(self) -> None:
        """Load checkpoint, vocabulary, and model."""

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SFT checkpoint not found: {self.checkpoint_path}"
            )

        checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(checkpoint, dict):
            raise RuntimeError(
                "Invalid SFT checkpoint: expected dictionary."
            )

        required_keys = {
            "model_config",
            "model_state_dict",
            "vocabulary",
        }

        missing = required_keys.difference(checkpoint.keys())

        if missing:
            raise RuntimeError(
                "Invalid SFT checkpoint. Missing keys: "
                + ", ".join(sorted(missing))
            )

        model_config = checkpoint["model_config"]
        vocabulary = checkpoint["vocabulary"]

        if not isinstance(model_config, dict):
            raise RuntimeError(
                "Invalid model_config in SFT checkpoint."
            )

        if not isinstance(vocabulary, dict):
            raise RuntimeError(
                "Invalid vocabulary in SFT checkpoint."
            )

        # --------------------------------------------------------------
        # Restore the exact vocabulary saved with the checkpoint.
        # --------------------------------------------------------------

        # --------------------------------------------------------------
        # Restore the checkpoint vocabulary into an isolated runtime
        # tokenizer. Never mutate the global Neura TOKENIZER.
        # --------------------------------------------------------------

        token_to_id = vocabulary.get("token_to_id")
        id_to_token = vocabulary.get("id_to_token")
        vocabulary_size = vocabulary.get("size")

        if not isinstance(token_to_id, dict):
            raise RuntimeError(
                "Invalid checkpoint vocabulary: token_to_id"
            )

        if not isinstance(id_to_token, dict):
            raise RuntimeError(
                "Invalid checkpoint vocabulary: id_to_token"
            )

        if not isinstance(vocabulary_size, int):
            raise RuntimeError(
                "Invalid checkpoint vocabulary: size"
            )

        if vocabulary_size <= 0:
            raise RuntimeError(
                "Checkpoint vocabulary size must be positive"
            )

        if len(token_to_id) != vocabulary_size:
            raise RuntimeError(
                "Checkpoint token_to_id size mismatch"
            )

        if len(id_to_token) != vocabulary_size:
            raise RuntimeError(
                "Checkpoint id_to_token size mismatch"
            )

        restored_vocab = Vocabulary()

        restored_vocab.token_to_id = {
            str(token): int(token_id)
            for token, token_id in token_to_id.items()
        }

        restored_vocab.id_to_token = {
            int(token_id): str(token)
            for token_id, token in id_to_token.items()
        }

        restored_vocab._next = vocabulary_size
        restored_vocab.capacity = vocabulary_size

        if "special_tokens" in vocabulary:
            restored_vocab.special_tokens = list(
                vocabulary["special_tokens"]
            )

        tokenizer = Tokenizer(
            vocabulary=restored_vocab
        )
        tokenizer.freeze()

        if tokenizer.vocab_size != vocabulary_size:
            raise RuntimeError(
                "Runtime tokenizer vocabulary size mismatch"
            )

        checkpoint_vocab_size = int(
            vocabulary["size"]
        )

        model_vocab_size = int(
            model_config["vocab_size"]
        )

        restored_vocab_size = int(
            restored_vocab.size()
        )

        if checkpoint_vocab_size != model_vocab_size:
            raise RuntimeError(
                "Checkpoint vocabulary/model vocabulary mismatch: "
                f"{checkpoint_vocab_size} != {model_vocab_size}"
            )

        if restored_vocab_size != model_vocab_size:
            raise RuntimeError(
                "Restored vocabulary/model vocabulary mismatch: "
                f"{restored_vocab_size} != {model_vocab_size}"
            )

        # --------------------------------------------------------------
        # Build model from checkpoint configuration.
        # --------------------------------------------------------------

        model = NeuraTorchModel(
            vocab_size=model_config["vocab_size"],
            hidden_size=model_config["hidden_size"],
            layers=model_config["layers"],
            heads=model_config["heads"],
            context_length=model_config["context_length"],
            dropout=model_config.get("dropout", 0.0),
        )

        load_result = model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        if load_result.missing_keys:
            raise RuntimeError(
                "Missing model keys: "
                + ", ".join(load_result.missing_keys)
            )

        if load_result.unexpected_keys:
            raise RuntimeError(
                "Unexpected model keys: "
                + ", ".join(load_result.unexpected_keys)
            )

        model.to(self.device)
        model.eval()

        self.checkpoint = checkpoint
        self.model = model
        self.model_config = model_config
        self.vocabulary = vocabulary
        self.tokenizer = tokenizer
        self._loaded = True

    # ------------------------------------------------------------------
    # STATUS
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        """Return whether the runtime is loaded."""
        return self._loaded

    @property
    def vocab_size(self) -> int:
        """Return model vocabulary size."""
        self._require_loaded()
        assert self.model_config is not None
        return int(self.model_config["vocab_size"])

    @property
    def context_length(self) -> int:
        """Return model context length."""
        self._require_loaded()
        assert self.model_config is not None
        return int(self.model_config["context_length"])

    # ------------------------------------------------------------------
    # TOKENIZATION
    # ------------------------------------------------------------------

    def encode(self, text: str) -> list[int]:
        """Encode text using the checkpoint vocabulary."""

        self._require_loaded()

        if not isinstance(text, str):
            raise TypeError(
                f"text must be str, got {type(text).__name__}"
            )

        if self.tokenizer is None:
            raise RuntimeError("runtime tokenizer is not initialized")
        return list(self.tokenizer.encode(text))

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs using the checkpoint vocabulary."""

        self._require_loaded()

        if self.tokenizer is None:
            raise RuntimeError("runtime tokenizer is not initialized")
        return self.tokenizer.decode(token_ids)

    # ------------------------------------------------------------------
    # GENERATION
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        prompt: str | list[int],
        *,
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        eos_token: int | None = None,
        repetition_penalty: float = 1.18,
        no_repeat_ngram_size: int = 2,
        repeat_token_limit: int = 3,
    ) -> SFTGenerationResult:
        """
        Generate autoregressively with repetition protection.

        Decoding policy
        ---------------
        temperature <= 0:
            Deterministic greedy decoding.

        temperature > 0:
            Temperature sampling.

        repetition_penalty:
            Penalizes tokens that already appeared in the generated
            response. The prompt itself is not penalized.

        no_repeat_ngram_size:
            Prevents generation of an n-gram that already occurred
            inside the generated response.

        repeat_token_limit:
            If the same token is selected this many consecutive times,
            generation stops with ``repetition``.
        """

        self._require_loaded()

        assert self.model is not None

        if max_new_tokens < 0:
            raise ValueError(
                "max_new_tokens must be >= 0"
            )

        if temperature < 0:
            raise ValueError(
                "temperature must be >= 0"
            )

        if repetition_penalty < 1.0:
            raise ValueError(
                "repetition_penalty must be >= 1.0"
            )

        if no_repeat_ngram_size < 0:
            raise ValueError(
                "no_repeat_ngram_size must be >= 0"
            )

        if repeat_token_limit < 2:
            raise ValueError(
                "repeat_token_limit must be >= 2"
            )

        # --------------------------------------------------------------
        # Encode prompt.
        # --------------------------------------------------------------

        if isinstance(prompt, str):
            prompt_tokens = self.encode(prompt)

        elif isinstance(prompt, list):
            prompt_tokens = [
                int(token_id)
                for token_id in prompt
            ]

        else:
            raise TypeError(
                "prompt must be str or list[int]"
            )

        if not prompt_tokens:
            raise ValueError(
                "prompt must contain at least one token"
            )

        # --------------------------------------------------------------
        # Context validation.
        # --------------------------------------------------------------

        if len(prompt_tokens) >= self.context_length:
            raise ValueError(
                "Prompt length exceeds available generation context: "
                f"{len(prompt_tokens)} >= {self.context_length}"
            )

        if eos_token is None:
            if self.tokenizer is None:
                raise RuntimeError("runtime tokenizer is not initialized")
            eos_token = self.tokenizer.vocab.encode("<EOS>")

        # --------------------------------------------------------------
        # Special-token protection.
        #
        # These tokens belong to the prompt/control format and must
        # never be generated as visible assistant content.
        #
        # EOS is intentionally excluded because it is the valid
        # termination token.
        # --------------------------------------------------------------

        blocked_token_ids: set[int] = set()

        for token in (
            "<PAD>",
            "<UNK>",
            "<BOS>",
            "<|system|>",
            "<|end|>",
            "<|user|>",
            "<|assistant|>",
        ):
            try:
                token_id = int(
                    self.tokenizer.vocab.encode(token)
                )
            except Exception:
                continue

            if token_id != eos_token:
                blocked_token_ids.add(token_id)

        generated_tokens: list[int] = []
        sequence = list(prompt_tokens)

        finish_reason = "length"

        # --------------------------------------------------------------
        # Helpers
        # --------------------------------------------------------------

        def get_banned_tokens(
            tokens: list[int],
            ngram_size: int,
        ) -> set[int]:
            """
            Return tokens that would create an n-gram already seen
            in the generated response.
            """

            if ngram_size <= 0:
                return set()

            if len(tokens) < ngram_size - 1:
                return set()

            if ngram_size == 1:
                return set(tokens)

            prefix = tuple(tokens[-(ngram_size - 1):])
            banned: set[int] = set()

            for index in range(
                len(tokens) - ngram_size + 1
            ):
                ngram = tokens[
                    index:index + ngram_size
                ]

                if tuple(
                    ngram[:-1]
                ) == prefix:
                    banned.add(
                        int(ngram[-1])
                    )

            return banned

        # --------------------------------------------------------------
        # Autoregressive loop.
        # --------------------------------------------------------------

        for _ in range(max_new_tokens):

            if len(sequence) >= self.context_length:
                finish_reason = "context_length"
                break

            input_ids = torch.tensor(
                [sequence],
                dtype=torch.long,
                device=self.device,
            )

            logits = self.model(input_ids)

            next_logits = (
                logits[0, -1].clone()
            )

            # ----------------------------------------------------------
            # Block control/special tokens.
            #
            # EOS remains available.
            # ----------------------------------------------------------

            for token_id in blocked_token_ids:
                next_logits[token_id] = float("-inf")

            # ----------------------------------------------------------
            # Repetition penalty
            #
            # Only generated tokens are penalized. Prompt tokens are
            # intentionally excluded.
            # ----------------------------------------------------------

            if (
                repetition_penalty > 1.0
                and generated_tokens
            ):
                seen_tokens = set(
                    generated_tokens
                )

                for token_id in seen_tokens:
                    value = next_logits[token_id]

                    if value > 0:
                        next_logits[token_id] = (
                            value
                            / repetition_penalty
                        )
                    else:
                        next_logits[token_id] = (
                            value
                            * repetition_penalty
                        )

            # ----------------------------------------------------------
            # No-repeat n-gram protection.
            # ----------------------------------------------------------

            banned_tokens = get_banned_tokens(
                generated_tokens,
                no_repeat_ngram_size,
            )

            for token_id in banned_tokens:
                if (
                    eos_token is None
                    or token_id != eos_token
                ):
                    next_logits[token_id] = (
                        float("-inf")
                    )

            # ----------------------------------------------------------
            # Immediate repeated-token protection.
            # ----------------------------------------------------------

            if (
                generated_tokens
                and len(generated_tokens)
                >= repeat_token_limit - 1
            ):
                recent = generated_tokens[
                    -(repeat_token_limit - 1):
                ]

                if (
                    len(set(recent)) == 1
                ):
                    repeated_token = recent[0]

                    if (
                        eos_token is None
                        or repeated_token != eos_token
                    ):
                        next_logits[
                            repeated_token
                        ] = float("-inf")

            # ----------------------------------------------------------
            # Safety fallback.
            #
            # If constraints accidentally eliminate every token,
            # return to the raw logits instead of crashing.
            # ----------------------------------------------------------

            if not torch.isfinite(
                next_logits
            ).any():
                next_logits = (
                    logits[0, -1].clone()
                )

            # ----------------------------------------------------------
            # Decode.
            # ----------------------------------------------------------

            if temperature <= 0.0:
                next_token = int(
                    torch.argmax(
                        next_logits
                    ).item()
                )

            else:
                scaled_logits = (
                    next_logits / temperature
                )

                probabilities = torch.softmax(
                    scaled_logits,
                    dim=-1,
                )

                if not torch.isfinite(
                    probabilities
                ).all():
                    next_token = int(
                        torch.argmax(
                            next_logits
                        ).item()
                    )
                else:
                    next_token = int(
                        torch.multinomial(
                            probabilities,
                            num_samples=1,
                        ).item()
                    )

            # ----------------------------------------------------------
            # Repetition-loop detection.
            # ----------------------------------------------------------

            if (
                generated_tokens
                and next_token
                == generated_tokens[-1]
                and len(generated_tokens)
                >= repeat_token_limit - 1
            ):
                recent = generated_tokens[
                    -(repeat_token_limit - 1):
                ]

                if (
                    len(set(recent)) == 1
                ):
                    finish_reason = "repetition"
                    break

            generated_tokens.append(
                next_token
            )

            sequence.append(
                next_token
            )

            # ----------------------------------------------------------
            # EOS.
            # ----------------------------------------------------------

            if (
                eos_token is not None
                and next_token == eos_token
            ):
                finish_reason = "eos"
                break

        return SFTGenerationResult(
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            finish_reason=finish_reason,
            tokenizer=self.tokenizer,
        )

    # ------------------------------------------------------------------
    # CHAT PROMPT
    # ------------------------------------------------------------------

    def build_chat_prompt(
        self,
        user_text: str,
        *,
        system_text: str = "You are Neura.",
    ) -> str:
        """
        Build a prompt using the SFT chat format.

        This matches the vocabulary/training format used by the current
        SFT checkpoint.
        """

        return (
            "<BOS> "
            "<|system|> "
            f"{system_text} "
            "<|end|> "
            "<|user|> "
            f"{user_text} "
            "<|assistant|>"
        )

    def chat(
        self,
        user_text: str,
        *,
        system_text: str = "You are Neura.",
        max_new_tokens: int = 32,
        temperature: float = 0.0,
        repetition_penalty: float = 1.18,
        no_repeat_ngram_size: int = 2,
        repeat_token_limit: int = 3,
    ) -> SFTGenerationResult:
        """Generate an assistant response using the SFT chat format."""

        prompt = self.build_chat_prompt(
            user_text,
            system_text=system_text,
        )

        return self.generate(
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            repeat_token_limit=repeat_token_limit,
        )

    # ------------------------------------------------------------------
    # INTERNAL
    # ------------------------------------------------------------------

    def _require_loaded(self) -> None:
        """Ensure runtime is loaded."""

        if not self._loaded:
            raise RuntimeError(
                "SFT runtime is not loaded."
            )
