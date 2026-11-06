"""
Neura Chat Engine
"""

from .session import ChatSession
from .prompt import PromptBuilder


class ChatEngine:

    def __init__(
        self,
        generator,
        tokenizer
    ):
        self.generator = generator
        self.tokenizer = tokenizer

        self.session = ChatSession()
        self.prompt_builder = PromptBuilder()

    def system(self, text):
        self.session.add_system(text)

    def chat(
        self,
        text,
        max_tokens=64
    ):

        self.session.add_user(text)

        prompt = self.prompt_builder.build(
            self.session.history()
        )

        tokens = self.tokenizer.encode(
            prompt
        )

        output = self.generator.generate(
            tokens,
            max_tokens=max_tokens
        )

        response_ids = output[len(tokens):]

        eos_token = self.tokenizer.vocab.encode(
            "<EOS>"
        )

        if eos_token in response_ids:
            response_ids = response_ids[
                :response_ids.index(eos_token)
            ]

        response = self.tokenizer.decode(
            response_ids
        )

        self.session.add_assistant(
            response
        )

        return response
