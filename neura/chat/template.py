"""
Neura Chat Template

Deterministic serialization of structured chat messages.

The exact same serialization contract is used by:
    - chat inference
    - SFT dataset preparation
    - assistant-loss masking
    - runtime tests
"""

from __future__ import annotations


class ChatTemplate:
    """
    Serialize structured messages into the Neura chat format.

    Format:

        <|system|>
        system content
        <|end|>
        <|user|>
        user content
        <|end|>
        <|assistant|>
        assistant content
        <|end|>
    """

    SYSTEM = "<|system|>"
    USER = "<|user|>"
    ASSISTANT = "<|assistant|>"
    END = "<|end|>"

    SUPPORTED_ROLES = {
        "system",
        "user",
        "assistant",
    }

    def validate_messages(self, messages):
        if not isinstance(messages, list):
            raise TypeError("messages must be a list")

        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                raise TypeError(
                    f"message at index {index} must be a dict"
                )

            role = message.get("role")
            content = message.get("content")

            if role not in self.SUPPORTED_ROLES:
                raise ValueError(
                    f"unsupported role at index {index}: {role}"
                )

            if not isinstance(content, str):
                raise TypeError(
                    f"message content at index {index} "
                    "must be a string"
                )

        return True

    def _marker(self, role: str) -> str:
        if role == "system":
            return self.SYSTEM

        if role == "user":
            return self.USER

        if role == "assistant":
            return self.ASSISTANT

        raise ValueError(
            f"unsupported role: {role}"
        )

    def render_message(self, message) -> str:
        if not isinstance(message, dict):
            raise TypeError("message must be a dict")

        role = message.get("role")
        content = message.get("content")

        if role not in self.SUPPORTED_ROLES:
            raise ValueError(
                f"unsupported role: {role}"
            )

        if not isinstance(content, str):
            raise TypeError(
                "message content must be a string"
            )

        marker = self._marker(role)

        return (
            f"{marker}\n"
            f"{content}\n"
            f"{self.END}"
        )

    def render(self, messages) -> str:
        self.validate_messages(messages)

        return "\n".join(
            self.render_message(message)
            for message in messages
        )

    def render_for_generation(self, messages) -> str:
        """
        Render a conversation and append an empty assistant turn.

        This is the canonical inference prompt.

        Example:

            <|user|>
            Hello
            <|end|>
            <|assistant|>
        """

        self.validate_messages(messages)

        parts = [
            self.render_message(message)
            for message in messages
        ]

        parts.append(self.ASSISTANT)

        return "\n".join(parts) + "\n"

    def assistant_spans(self, messages):
        """
        Return character spans containing complete assistant blocks.

        Each span is:
            (start, end)

        The span includes:
            <|assistant|>
            assistant content
            <|end|>
        """

        self.validate_messages(messages)

        rendered = self.render(messages)
        spans = []

        cursor = 0

        for message in messages:
            block = self.render_message(message)

            start = rendered.find(
                block,
                cursor,
            )

            if start == -1:
                raise RuntimeError(
                    "failed to locate rendered message"
                )

            end = start + len(block)

            if message["role"] == "assistant":
                spans.append((start, end))

            cursor = end

            if (
                cursor < len(rendered)
                and rendered[cursor] == "\n"
            ):
                cursor += 1

        return spans

    def assistant_content_spans(self, messages):
        """
        Return character spans containing only assistant content.

        The role marker and <|end|> marker are excluded.

        Each span is:
            (start, end)
        """

        self.validate_messages(messages)

        rendered = self.render(messages)
        spans = []

        cursor = 0

        for message in messages:
            block = self.render_message(message)

            start = rendered.find(
                block,
                cursor,
            )

            if start == -1:
                raise RuntimeError(
                    "failed to locate rendered message"
                )

            if message["role"] == "assistant":
                marker = self.ASSISTANT + "\n"
                prefix_start = start + len(marker)

                content_start = prefix_start
                content_end = (
                    content_start
                    + len(message["content"])
                )

                spans.append(
                    (
                        content_start,
                        content_end,
                    )
                )

            cursor = start + len(block)

            if (
                cursor < len(rendered)
                and rendered[cursor] == "\n"
            ):
                cursor += 1

        return spans


__all__ = ["ChatTemplate"]
