"""
Neura Chat SFT V2 inference bridge.

Loads the production chat-sft-v2-pytorch checkpoint without modifying
the existing model, tokenizer, training, or legacy inference runtime.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from neura.model.torch_model import NeuraTorchModel
from neura.tokenizer.neura_bpe_tokenizer import NeuraBPETokenizer


DEFAULT_CHECKPOINT = Path(
    "checkpoints/chat_sft_v2/neura_chat_sft_last.pt"
)

EXPECTED_VERSION = "chat-sft-v2-pytorch"
DEFAULT_MAX_NEW_TOKENS = 128
ASSISTANT_PREFIX = "<|assistant|>\n"
END_MARKER = "\n<|end|>\n"


class NeuraSFTV2Runtime:
    """Inference runtime for the current Chat SFT V2 checkpoint."""

    def __init__(
        self,
        checkpoint_path: str | Path = DEFAULT_CHECKPOINT,
        *,
        device: str | torch.device | None = None,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)

        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"SFT V2 checkpoint not found: {self.checkpoint_path}"
            )

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = torch.device(device)

        state = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        if not isinstance(state, dict):
            raise ValueError("SFT V2 checkpoint must contain a dictionary")

        version = state.get("version")
        if version != EXPECTED_VERSION:
            raise ValueError(
                f"Unsupported checkpoint version: {version!r}; "
                f"expected {EXPECTED_VERSION!r}"
            )

        required = {
            "model_config",
            "model_state_dict",
            "tokenizer",
        }

        missing = required.difference(state)
        if missing:
            raise ValueError(
                f"SFT V2 checkpoint missing keys: {sorted(missing)}"
            )

        self.model_config = dict(state["model_config"])
        tokenizer_meta = dict(state["tokenizer"])

        artifact = tokenizer_meta.get("artifact")
        if not artifact:
            raise ValueError(
                "SFT V2 checkpoint tokenizer metadata has no artifact"
            )

        self.tokenizer_path = self._resolve_artifact(
            str(artifact)
        )

        self.tokenizer = NeuraBPETokenizer(
            self.tokenizer_path
        )

        checkpoint_vocab = int(
            tokenizer_meta["vocab_size"]
        )

        model_vocab = int(
            self.model_config["vocab_size"]
        )

        if self.tokenizer.vocab_size != checkpoint_vocab:
            raise ValueError(
                "Tokenizer/checkpoint vocab mismatch: "
                f"{self.tokenizer.vocab_size} != {checkpoint_vocab}"
            )

        if self.tokenizer.vocab_size != model_vocab:
            raise ValueError(
                "Tokenizer/model vocab mismatch: "
                f"{self.tokenizer.vocab_size} != {model_vocab}"
            )

        self.model = NeuraTorchModel(
            vocab_size=model_vocab,
            hidden_size=int(self.model_config["hidden_size"]),
            layers=int(self.model_config["layers"]),
            heads=int(self.model_config["heads"]),
            context_length=int(self.model_config["context_length"]),
            dropout=float(self.model_config.get("dropout", 0.0)),
        ).to(self.device)

        self.model.load_state_dict(
            state["model_state_dict"],
            strict=True,
        )

        self.model.eval()

        self.context_length = int(
            self.model_config["context_length"]
        )

        self.step = int(state.get("step", 0))

    def _resolve_artifact(self, artifact: str) -> str:
        path = Path(artifact)

        if path.exists():
            return str(path)

        project_root = self.checkpoint_path.resolve().parents[2]

        candidate = project_root / path
        if candidate.exists():
            return str(candidate)

        raise FileNotFoundError(
            "Tokenizer artifact not found. "
            f"Saved path={artifact!r}"
        )

    def build_prompt(
        self,
        text: str,
        *,
        context: dict[str, Any] | None = None,
    ) -> str:
        if not isinstance(text, str):
            raise TypeError("text must be a string")

        text = text.strip()

        if not text:
            raise ValueError("text cannot be empty")

        context = context or {}

        messages: list[tuple[str, str]] = []

        system = context.get("system")
        if system:
            messages.append(("system", str(system)))

        history = context.get("history", [])

        for item in history:
            if not isinstance(item, dict):
                continue

            role = str(item.get("role", "")).strip()
            content = str(item.get("content", "")).strip()

            if role in {"system", "user", "assistant"} and content:
                messages.append((role, content))

        messages.append(("user", text))

        parts = ["<|assistant|>\n"]

        # If history/context is supplied, construct the full training format.
        if messages:
            parts = []

            for role, content in messages:
                parts.append(
                    f"<|{role}|>\n"
                    f"{content}"
                    f"\n<|end|>\n"
                )

            parts.append(ASSISTANT_PREFIX)

        return "".join(parts)

    def generate(
        self,
        text: str,
        *,
        context: dict[str, Any] | None = None,
        route: str = "chat",
        max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
        **_: Any,
    ) -> str:
        del route

        if not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")

        if max_new_tokens <= 0:
            raise ValueError(
                "max_new_tokens must be positive"
            )

        prompt = self.build_prompt(
            text,
            context=context,
        )

        input_ids = self.tokenizer.encode(
            prompt,
            add_bos=True,
            add_eos=False,
        )

        if not input_ids:
            raise RuntimeError("Prompt produced no tokens")

        if len(input_ids) >= self.context_length:
            raise ValueError(
                "Prompt already reaches the model context length: "
                f"{len(input_ids)} >= {self.context_length}"
            )

        generated: list[int] = []

        with torch.no_grad():
            for _step in range(max_new_tokens):
                if len(input_ids) >= self.context_length:
                    break

                tensor = torch.tensor(
                    [input_ids],
                    dtype=torch.long,
                    device=self.device,
                )

                logits = self.model(tensor)

                next_id = int(
                    torch.argmax(
                        logits[0, -1],
                        dim=-1,
                    ).item()
                )

                generated.append(next_id)
                input_ids.append(next_id)

                # Byte-level BPE may temporarily contain an incomplete
                # UTF-8 sequence. Do not decode after every generated token.
                # Decode only when the accumulated byte sequence is valid.
                try:
                    text_so_far = self.tokenizer.decode(
                        generated,
                        skip_special_tokens=False,
                    )
                except RuntimeError as exc:
                    if "Decoded byte sequence is not valid UTF-8" not in str(exc):
                        raise
                    text_so_far = None

                if text_so_far is not None:
                    if END_MARKER in text_so_far:
                        text_so_far = text_so_far.split(
                            END_MARKER,
                            1,
                        )[0]
                        return text_so_far

                if next_id == self.tokenizer.EOS_ID:
                    break

        return self.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip()


__all__ = [
    "NeuraSFTV2Runtime",
    "DEFAULT_CHECKPOINT",
]
