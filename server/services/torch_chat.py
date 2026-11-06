"""
Neura PyTorch Chat Backend

Checkpoint-backed HTTP-facing adapter for NeuraTorchModel.
"""

from __future__ import annotations

from pathlib import Path

import torch

from neura.model.torch_model import NeuraTorchModel
from neura.tokenizer import TOKENIZER


class TorchChatServiceError(Exception):
    """Raised when the PyTorch chat backend fails."""


class TorchChatService:

    CHECKPOINT_PATH = Path(
        "checkpoints/neura_torch_best.pt"
    )

    VOCAB_PATH = Path(
        "checkpoints/neura_vocab.json"
    )

    HIDDEN_SIZE = 256
    LAYERS = 4
    HEADS = 8
    CONTEXT_LENGTH = 1024
    DROPOUT = 0.0

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = None
        self.model_vocab_size = 0

        self.checkpoint_loaded = False
        self.vocabulary_loaded = False

        self._load_vocabulary()
        self._load_checkpoint()

    # ---------------------------------------------------------
    # VOCABULARY
    # ---------------------------------------------------------

    def _load_vocabulary(self):

        if not self.VOCAB_PATH.exists():
            raise TorchChatServiceError(
                f"vocabulary checkpoint not found: "
                f"{self.VOCAB_PATH}"
            )

        try:
            TOKENIZER.vocab.load(
                self.VOCAB_PATH
            )

            vocabulary_size = int(
                TOKENIZER.vocab.size()
            )

            if vocabulary_size <= 0:
                raise TorchChatServiceError(
                    "loaded vocabulary is empty"
                )

            TOKENIZER.vocab.ensure_size(
                vocabulary_size
            )

            TOKENIZER.freeze()

            self.vocabulary_loaded = True

        except TorchChatServiceError:
            raise

        except Exception as exc:
            raise TorchChatServiceError(
                f"failed to load vocabulary: {exc}"
            ) from exc

    # ---------------------------------------------------------
    # CHECKPOINT
    # ---------------------------------------------------------

    def _load_checkpoint(self):

        if not self.CHECKPOINT_PATH.exists():
            raise TorchChatServiceError(
                f"model checkpoint not found: "
                f"{self.CHECKPOINT_PATH}"
            )

        vocab_size = int(
            TOKENIZER.vocab.size()
        )

        self.model = NeuraTorchModel(
            vocab_size=vocab_size,
            hidden_size=self.HIDDEN_SIZE,
            layers=self.LAYERS,
            heads=self.HEADS,
            context_length=self.CONTEXT_LENGTH,
            dropout=self.DROPOUT,
        ).to(self.device)

        try:
            checkpoint = torch.load(
                self.CHECKPOINT_PATH,
                map_location=self.device,
            )

            state_dict = self._extract_state_dict(
                checkpoint
            )

            self.model.load_state_dict(
                state_dict,
                strict=True,
            )

        except Exception as exc:
            self.model = None

            raise TorchChatServiceError(
                f"failed to load model checkpoint: {exc}"
            ) from exc

        self.model.eval()

        self.model_vocab_size = vocab_size
        self.checkpoint_loaded = True

    @staticmethod
    def _extract_state_dict(checkpoint):

        if isinstance(checkpoint, dict):

            if all(
                isinstance(key, str)
                for key in checkpoint.keys()
            ):
                tensor_values = [
                    value
                    for value in checkpoint.values()
                    if torch.is_tensor(value)
                ]

                if len(tensor_values) == len(
                    checkpoint
                ):
                    return checkpoint

            for key in (
                "model_state_dict",
                "state_dict",
                "model",
            ):
                value = checkpoint.get(key)

                if isinstance(value, dict):
                    return value

        raise TorchChatServiceError(
            "unsupported checkpoint format"
        )

    # ---------------------------------------------------------
    # GENERATION
    # ---------------------------------------------------------

    @torch.inference_mode()
    def generate(
        self,
        message: str,
        max_tokens: int = 1,
    ) -> str:

        if not isinstance(message, str):
            raise TorchChatServiceError(
                "message must be a string"
            )

        message = message.strip()

        if not message:
            raise TorchChatServiceError(
                "message cannot be empty"
            )

        if not isinstance(
            max_tokens,
            int,
        ) or isinstance(
            max_tokens,
            bool,
        ):
            raise TorchChatServiceError(
                "max_tokens must be an integer"
            )

        if max_tokens <= 0:
            raise TorchChatServiceError(
                "max_tokens must be greater than zero"
            )

        if self.model is None:
            raise TorchChatServiceError(
                "PyTorch model is not loaded"
            )

        if not self.checkpoint_loaded:
            raise TorchChatServiceError(
                "PyTorch checkpoint is not loaded"
            )

        max_tokens = min(
            max_tokens,
            4,
        )

        token_ids = TOKENIZER.encode(
            message
        )

        if not token_ids:
            raise TorchChatServiceError(
                "tokenizer produced no tokens"
            )

        for token_id in token_ids:

            if not (
                0
                <= int(token_id)
                < self.model_vocab_size
            ):
                raise TorchChatServiceError(
                    "tokenizer/model vocabulary mismatch: "
                    f"token_id={token_id}, "
                    f"model_vocab_size="
                    f"{self.model_vocab_size}"
                )

        if len(token_ids) > self.model.context_length:
            token_ids = token_ids[
                -self.model.context_length:
            ]

        generated = list(
            token_ids
        )

        prompt_length = len(
            generated
        )

        eos_token_id = (
            TOKENIZER.vocab.token_to_id.get(
                "<EOS>"
            )
        )

        for _ in range(max_tokens):

            input_ids = torch.tensor(
                [
                    generated[
                        -self.model.context_length:
                    ]
                ],
                dtype=torch.long,
                device=self.device,
            )

            logits = self.model(
                input_ids
            )

            next_logits = logits[
                0,
                -1,
            ].clone()

            blocked_token_ids = (
                self._blocked_generation_token_ids()
            )

            for token_id in blocked_token_ids:

                if (
                    0
                    <= token_id
                    < next_logits.numel()
                ):
                    next_logits[
                        token_id
                    ] = float("-inf")

            next_token = int(
                torch.argmax(
                    next_logits
                ).item()
            )

            generated.append(
                next_token
            )

            if (
                eos_token_id is not None
                and next_token == int(
                    eos_token_id
                )
            ):
                break

        new_tokens = generated[
            prompt_length:
        ]

        return TOKENIZER.decode(
            new_tokens
        )

    # ---------------------------------------------------------
    # TOKEN BLOCKING
    # ---------------------------------------------------------

    def _blocked_generation_token_ids(self):

        blocked = []

        for token in (
            "<PAD>",
            "<UNK>",
            "<BOS>",
        ):

            token_id = (
                TOKENIZER.vocab.token_to_id.get(
                    token
                )
            )

            if token_id is not None:
                blocked.append(
                    int(token_id)
                )

        return blocked

    # ---------------------------------------------------------
    # STATUS
    # ---------------------------------------------------------

    def status(self):

        return {
            "backend": "pytorch",
            "device": str(
                self.device
            ),
            "model": (
                "NeuraTorchModel"
                if self.model is not None
                else None
            ),
            "model_vocab_size": (
                self.model_vocab_size
            ),
            "tokenizer_vocab_size": int(
                TOKENIZER.vocab.size()
            ),
            "training_checkpoint": (
                self.checkpoint_loaded
            ),
            "checkpoint_path": str(
                self.CHECKPOINT_PATH
            ),
            "vocabulary_path": str(
                self.VOCAB_PATH
            ),
            "production_ready": False,
        }


__all__ = [
    "TorchChatService",
    "TorchChatServiceError",
]
