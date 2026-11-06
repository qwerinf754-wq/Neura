"""
Neura Chat SFT Dataset

Converts structured chat messages into token-level
training examples with assistant-only loss masking.
"""

from __future__ import annotations

from dataclasses import dataclass

from neura.chat import ChatTemplate
from neura.tokenizer import TOKENIZER


IGNORE_INDEX = -100


@dataclass
class ChatSFTExample:
    input_ids: list[int]
    labels: list[int]


class ChatSFTDatasetBuilder:

    def __init__(
        self,
        tokenizer=None,
        template=None,
    ):
        self.tokenizer = (
            tokenizer
            if tokenizer is not None
            else TOKENIZER
        )

        self.template = (
            template
            if template is not None
            else ChatTemplate()
        )

    # ---------------------------------------------------------
    # PUBLIC API
    # ---------------------------------------------------------

    def build(self, messages):
        """
        Build one SFT example.

        Assistant content is trainable.
        Everything else is masked with IGNORE_INDEX.
        """

        if not isinstance(messages, list):
            raise TypeError(
                "messages must be a list"
            )

        rendered = self.template.render(
            messages
        )

        input_ids = self.tokenizer.encode(
            rendered
        )

        labels = [
            IGNORE_INDEX
            for _ in input_ids
        ]

        self._mask_assistant_content(
            rendered=rendered,
            input_ids=input_ids,
            labels=labels,
            messages=messages,
        )

        return ChatSFTExample(
            input_ids=input_ids,
            labels=labels,
        )

    # ---------------------------------------------------------
    # ASSISTANT MASKING
    # ---------------------------------------------------------

    def _mask_assistant_content(
        self,
        rendered,
        input_ids,
        labels,
        messages,
    ):
        """
        Mark assistant content as trainable.

        Because the initial tokenizer is whitespace based,
        token boundaries are reconstructed using the same
        tokenizer encoding operation.
        """

        cursor = 0

        for message in messages:

            block = self.template.render(
                [message]
            )

            start = rendered.find(
                block,
                cursor,
            )

            if start < 0:
                raise RuntimeError(
                    "unable to locate message block"
                )

            end = start + len(block)

            if message["role"] == "assistant":

                marker = (
                    self.template.ASSISTANT
                    + "\n"
                )

                content_start = (
                    start
                    + len(marker)
                )

                # Include the assistant EOS marker in the
                # supervised span.
                #
                # Rendered assistant block:
                #
                #   <|assistant|>\n
                #   content\n
                #   <|end|>\n
                #
                # We exclude only the final newline after EOS.
                # Therefore <|end|> itself remains trainable.
                content_end = (
                    end
                    - len("\n")
                )

                self._train_character_span(
                    rendered=rendered,
                    input_ids=input_ids,
                    labels=labels,
                    start=content_start,
                    end=content_end,
                )

            cursor = end

            if (
                cursor < len(rendered)
                and rendered[cursor] == "\n"
            ):
                cursor += 1

    # ---------------------------------------------------------
    # CHARACTER -> TOKEN SPAN
    # ---------------------------------------------------------

    def _train_character_span(
        self,
        rendered,
        input_ids,
        labels,
        start,
        end,
    ):
        """
        Convert a character span into tokenizer-token
        supervision.

        The current tokenizer is whitespace based, so token
        offsets are reconstructed deterministically.
        """

        prefix = rendered[:start]
        target = rendered[start:end]

        prefix_ids = self.tokenizer.encode(
            prefix
        )

        target_ids = self.tokenizer.encode(
            target
        )

        token_start = len(prefix_ids)
        token_end = (
            token_start
            + len(target_ids)
        )

        if token_start < 0:
            token_start = 0

        if token_end > len(input_ids):
            token_end = len(input_ids)

        for index in range(
            token_start,
            token_end,
        ):
            labels[index] = input_ids[index]


__all__ = [
    "IGNORE_INDEX",
    "ChatSFTExample",
    "ChatSFTDatasetBuilder",
]
