"""
Neura Chat SFT Dataset

Converts JSONL conversations into token-level supervised fine-tuning
examples.

Important:
- User/system tokens are masked from the loss.
- Assistant tokens are the learning targets.
- Supports multi-turn conversations.
- Supports truncation.
- Does not perform training itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable

import torch
from torch.utils.data import Dataset


IGNORE_INDEX = -100

SYSTEM_ROLE = "system"
USER_ROLE = "user"
ASSISTANT_ROLE = "assistant"


@dataclass(frozen=True)
class ChatMessage:
    role: str
    content: str


@dataclass
class ChatSFTExample:
    input_ids: list[int]
    labels: list[int]
    loss_mask: list[int]

    @property
    def length(self) -> int:
        return len(self.input_ids)


def validate_messages(
    messages: Iterable[dict[str, Any]],
) -> list[ChatMessage]:
    result: list[ChatMessage] = []

    for message in messages:
        role = message.get("role")
        content = message.get("content")

        if role not in {
            SYSTEM_ROLE,
            USER_ROLE,
            ASSISTANT_ROLE,
        }:
            raise ValueError(
                f"Unsupported chat role: {role!r}"
            )

        if not isinstance(content, str):
            raise TypeError(
                "Message content must be a string."
            )

        if not content:
            raise ValueError(
                "Message content cannot be empty."
            )

        result.append(
            ChatMessage(
                role=role,
                content=content,
            )
        )

    if not result:
        raise ValueError(
            "Conversation contains no messages."
        )

    return result


class ChatSFTBuilder:
    """
    Builds token-level SFT examples.

    The tokenizer must provide:
        encode(text) -> list[int]

    The builder intentionally does not depend on a specific tokenizer
    implementation, allowing the future 32K BPE tokenizer to replace
    the current tokenizer without changing the dataset API.
    """

    def __init__(
        self,
        tokenizer: Any,
        *,
        max_length: int = 4096,
        system_prefix: str = "<|system|>\n",
        user_prefix: str = "<|user|>\n",
        assistant_prefix: str = "<|assistant|>\n",
        message_end: str = "\n<|end|>\n",
        add_bos: bool = True,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
    ) -> None:
        if max_length <= 0:
            raise ValueError(
                "max_length must be positive."
            )

        self.tokenizer = tokenizer
        self.max_length = max_length

        self.system_prefix = system_prefix
        self.user_prefix = user_prefix
        self.assistant_prefix = assistant_prefix
        self.message_end = message_end

        self.add_bos = add_bos
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id

    def _encode(
        self,
        text: str,
    ) -> list[int]:
        ids = self.tokenizer.encode(text)

        if not isinstance(ids, list):
            ids = list(ids)

        return [
            int(token_id)
            for token_id in ids
        ]

    def _prefix_for_role(
        self,
        role: str,
    ) -> str:
        if role == SYSTEM_ROLE:
            return self.system_prefix

        if role == USER_ROLE:
            return self.user_prefix

        if role == ASSISTANT_ROLE:
            return self.assistant_prefix

        raise ValueError(
            f"Unsupported role: {role!r}"
        )

    def build(
        self,
        messages: Iterable[dict[str, Any] | ChatMessage],
    ) -> ChatSFTExample:
        normalized: list[ChatMessage] = []

        for message in messages:
            if isinstance(message, ChatMessage):
                normalized.append(message)
            else:
                normalized.extend(
                    validate_messages([message])
                )

        if not normalized:
            raise ValueError(
                "Conversation contains no messages."
            )

        input_ids: list[int] = []
        labels: list[int] = []
        loss_mask: list[int] = []

        if (
            self.add_bos
            and self.bos_token_id is not None
        ):
            input_ids.append(
                int(self.bos_token_id)
            )
            labels.append(IGNORE_INDEX)
            loss_mask.append(0)

        for message in normalized:
            prefix_ids = self._encode(
                self._prefix_for_role(
                    message.role
                )
            )

            content_ids = self._encode(
                message.content
            )

            end_ids = self._encode(
                self.message_end
            )

            input_ids.extend(prefix_ids)
            labels.extend(
                [IGNORE_INDEX] * len(prefix_ids)
            )
            loss_mask.extend(
                [0] * len(prefix_ids)
            )

            input_ids.extend(content_ids)

            if message.role == ASSISTANT_ROLE:
                labels.extend(content_ids)
                loss_mask.extend(
                    [1] * len(content_ids)
                )
            else:
                labels.extend(
                    [IGNORE_INDEX]
                    * len(content_ids)
                )
                loss_mask.extend(
                    [0] * len(content_ids)
                )

            input_ids.extend(end_ids)
            labels.extend(
                [IGNORE_INDEX] * len(end_ids)
            )
            loss_mask.extend(
                [0] * len(end_ids)
            )

        if (
            self.eos_token_id is not None
            and len(input_ids) < self.max_length
        ):
            input_ids.append(
                int(self.eos_token_id)
            )
            labels.append(
                IGNORE_INDEX
            )
            loss_mask.append(0)

        # ----------------------------------------------------------
        # Truncation
        #
        # Keep the most recent context because it is usually the
        # most useful part of a long conversation.
        # ----------------------------------------------------------

        if len(input_ids) > self.max_length:
            input_ids = input_ids[
                -self.max_length:
            ]
            labels = labels[
                -self.max_length:
            ]
            loss_mask = loss_mask[
                -self.max_length:
            ]

        if not any(loss_mask):
            raise ValueError(
                "Conversation contains no assistant "
                "tokens available for training."
            )

        return ChatSFTExample(
            input_ids=input_ids,
            labels=labels,
            loss_mask=loss_mask,
        )


class ChatSFTDataset(Dataset):
    """
    JSONL-backed chat SFT dataset.

    Each line must contain:

    {
        "messages": [
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """

    def __init__(
        self,
        path: str | Path,
        builder: ChatSFTBuilder,
    ) -> None:
        self.path = Path(path)
        self.builder = builder

        if not self.path.exists():
            raise FileNotFoundError(
                self.path
            )

        self.examples: list[ChatSFTExample] = []

        with self.path.open(
            "r",
            encoding="utf-8",
        ) as f:
            for line_no, line in enumerate(
                f,
                1,
            ):
                line = line.strip()

                if not line:
                    continue

                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(
                        f"Invalid JSON at line {line_no}"
                    ) from exc

                messages = record.get(
                    "messages"
                )

                if not isinstance(
                    messages,
                    list,
                ):
                    raise ValueError(
                        f"Missing messages at line "
                        f"{line_no}"
                    )

                self.examples.append(
                    builder.build(messages)
                )

        if not self.examples:
            raise ValueError(
                "SFT dataset is empty."
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(
        self,
        index: int,
    ) -> ChatSFTExample:
        return self.examples[index]


class ChatSFTCollator:
    """
    Pads variable-length examples into PyTorch tensors.
    """

    def __init__(
        self,
        *,
        pad_token_id: int = 0,
        ignore_index: int = IGNORE_INDEX,
    ) -> None:
        self.pad_token_id = int(
            pad_token_id
        )
        self.ignore_index = int(
            ignore_index
        )

    def __call__(
        self,
        examples: list[ChatSFTExample],
    ) -> dict[str, torch.Tensor]:
        if not examples:
            raise ValueError(
                "Cannot collate an empty batch."
            )

        max_length = max(
            example.length
            for example in examples
        )

        batch_size = len(examples)

        input_ids = torch.full(
            (
                batch_size,
                max_length,
            ),
            self.pad_token_id,
            dtype=torch.long,
        )

        labels = torch.full(
            (
                batch_size,
                max_length,
            ),
            self.ignore_index,
            dtype=torch.long,
        )

        attention_mask = torch.zeros(
            (
                batch_size,
                max_length,
            ),
            dtype=torch.long,
        )

        loss_mask = torch.zeros(
            (
                batch_size,
                max_length,
            ),
            dtype=torch.long,
        )

        for row, example in enumerate(
            examples
        ):
            length = example.length

            input_ids[
                row,
                :length,
            ] = torch.tensor(
                example.input_ids,
                dtype=torch.long,
            )

            labels[
                row,
                :length,
            ] = torch.tensor(
                example.labels,
                dtype=torch.long,
            )

            attention_mask[
                row,
                :length,
            ] = 1

            loss_mask[
                row,
                :length,
            ] = torch.tensor(
                example.loss_mask,
                dtype=torch.long,
            )

        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "loss_mask": loss_mask,
        }


def load_chat_sft_dataset(
    path: str | Path,
    tokenizer: Any,
    *,
    max_length: int = 4096,
    pad_token_id: int = 0,
    bos_token_id: int | None = None,
    eos_token_id: int | None = None,
) -> tuple[ChatSFTDataset, ChatSFTCollator]:
    """
    Convenience loader.
    """

    builder = ChatSFTBuilder(
        tokenizer,
        max_length=max_length,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
    )

    dataset = ChatSFTDataset(
        path,
        builder,
    )

    collator = ChatSFTCollator(
        pad_token_id=pad_token_id,
    )

    return dataset, collator


__all__ = [
    "IGNORE_INDEX",
    "ChatMessage",
    "ChatSFTExample",
    "ChatSFTBuilder",
    "ChatSFTDataset",
    "ChatSFTCollator",
    "load_chat_sft_dataset",
]
