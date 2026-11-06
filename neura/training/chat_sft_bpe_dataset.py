from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

IGNORE_INDEX = -100

PAD_ID = 0
BOS_ID = 2
EOS_ID = 3


@dataclass
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatSFTExample:
    input_ids: list[int]
    labels: list[int]
    loss_mask: list[int]


class BPETokenizerProtocol:
    """Minimal tokenizer contract required by production SFT."""

    vocab_size: int
    PAD_ID: int
    UNK_ID: int
    BOS_ID: int
    EOS_ID: int

    def encode(
        self,
        text: str,
        *,
        add_bos: bool = False,
        add_eos: bool = False,
    ) -> list[int]:
        raise NotImplementedError


class ChatSFTBuilderBPE:
    """
    Builds token-level SFT examples.

    Only assistant content + assistant EOS contribute to loss.
    """

    VALID_ROLES = {"system", "user", "assistant"}

    def __init__(
        self,
        tokenizer: BPETokenizerProtocol,
        *,
        max_length: int = 4096,
        add_bos: bool = True,
        add_eos: bool = False,
    ) -> None:
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.add_bos = add_bos
        self.add_eos = add_eos

        # Tokenizer is the single source of truth.
        self.vocab_size = int(self.tokenizer.vocab_size)
        self.pad_id = int(self.tokenizer.PAD_ID)
        self.unk_id = int(self.tokenizer.UNK_ID)
        self.bos_id = int(self.tokenizer.BOS_ID)
        self.eos_id = int(self.tokenizer.EOS_ID)

        if self.vocab_size <= 0:
            raise ValueError("tokenizer.vocab_size must be positive")

        for name, token_id in (
            ("PAD_ID", self.pad_id),
            ("UNK_ID", self.unk_id),
            ("BOS_ID", self.bos_id),
            ("EOS_ID", self.eos_id),
        ):
            if not 0 <= token_id < self.vocab_size:
                raise ValueError(
                    f"tokenizer {name}={token_id} is outside "
                    f"vocabulary size {self.vocab_size}"
                )

    def _encode(self, text: str) -> list[int]:
        return self.tokenizer.encode(
            text,
            add_bos=False,
            add_eos=False,
        )

    def build(
        self,
        messages: Iterable[dict[str, Any] | ChatMessage],
    ) -> ChatSFTExample:
        messages = list(messages)

        if not messages:
            raise ValueError("messages cannot be empty")

        normalized: list[ChatMessage] = []

        for message in messages:
            if isinstance(message, ChatMessage):
                item = message
            else:
                item = ChatMessage(
                    role=str(message["role"]),
                    content=str(message["content"]),
                )

            if item.role not in self.VALID_ROLES:
                raise ValueError(f"Invalid role: {item.role!r}")

            if not item.content.strip():
                raise ValueError("Message content cannot be empty")

            normalized.append(item)

        if not any(m.role == "assistant" for m in normalized):
            raise ValueError("Conversation must contain an assistant message")

        input_ids: list[int] = []
        labels: list[int] = []
        loss_mask: list[int] = []

        if self.add_bos:
            input_ids.append(self.bos_id)
            labels.append(IGNORE_INDEX)
            loss_mask.append(0)

        for message in normalized:
            role_text = f"<|{message.role}|>\n"
            end_text = "\n<|end|>\n"

            role_ids = self._encode(role_text)
            content_ids = self._encode(message.content)
            end_ids = self._encode(end_text)

            input_ids.extend(role_ids)
            labels.extend([IGNORE_INDEX] * len(role_ids))
            loss_mask.extend([0] * len(role_ids))

            input_ids.extend(content_ids)

            if message.role == "assistant":
                labels.extend(content_ids)
                loss_mask.extend([1] * len(content_ids))
            else:
                labels.extend([IGNORE_INDEX] * len(content_ids))
                loss_mask.extend([0] * len(content_ids))

            input_ids.extend(end_ids)

            if message.role == "assistant":
                labels.extend(end_ids)
                loss_mask.extend([1] * len(end_ids))
            else:
                labels.extend([IGNORE_INDEX] * len(end_ids))
                loss_mask.extend([0] * len(end_ids))

        if self.add_eos:
            input_ids.append(self.eos_id)
            labels.append(self.eos_id)
            loss_mask.append(1 if normalized[-1].role == "assistant" else 0)

        if len(input_ids) > self.max_length:
            # Preserve BOS while keeping the most recent context.
            # The first token is BOS because add_bos=True was requested.
            if self.add_bos and input_ids[0] == self.bos_id:
                keep = self.max_length - 1

                input_ids = (
                    [input_ids[0]]
                    + input_ids[-keep:]
                )
                labels = (
                    [labels[0]]
                    + labels[-keep:]
                )
                loss_mask = (
                    [loss_mask[0]]
                    + loss_mask[-keep:]
                )
            else:
                # No BOS contract: keep the most recent context.
                input_ids = input_ids[-self.max_length:]
                labels = labels[-self.max_length:]
                loss_mask = loss_mask[-self.max_length:]

        if len(input_ids) != len(labels):
            raise RuntimeError("input_ids/labels length mismatch")

        if len(input_ids) != len(loss_mask):
            raise RuntimeError("input_ids/loss_mask length mismatch")

        if not any(x == 1 for x in loss_mask):
            raise ValueError("No assistant tokens available for training")

        return ChatSFTExample(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
        )


def validate_assistant_only_mask(
    example: ChatSFTExample,
) -> None:
    """
    Strict invariant:

        loss_mask == 1  <=>  label != IGNORE_INDEX
    """

    if len(example.input_ids) != len(example.labels):
        raise AssertionError("input_ids/labels mismatch")

    if len(example.input_ids) != len(example.loss_mask):
        raise AssertionError("input_ids/loss_mask mismatch")

    for index, (label, mask) in enumerate(
        zip(example.labels, example.loss_mask)
    ):
        if mask == 1 and label == IGNORE_INDEX:
            raise AssertionError(
                f"Assistant mask has IGNORE_INDEX at token {index}"
            )

        if mask == 0 and label != IGNORE_INDEX:
            raise AssertionError(
                f"Non-assistant token contributes to loss at token {index}"
            )


__all__ = [
    "IGNORE_INDEX",
    "ChatMessage",
    "ChatSFTExample",
    "ChatSFTBuilderBPE",
    "validate_assistant_only_mask",
]
