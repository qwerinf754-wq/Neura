"""
Neura CLI Chat
"""

from neura.chat.engine import ChatEngine
from neura.tokenizer.tokenizer import TOKENIZER
from neura.generation.generator import Generator
from neura.model.model import NeuraModel
from neura.model.config import NeuraTiny


def build_chat():

    model = NeuraModel(config=NeuraTiny())

    generator = Generator(model)

    return ChatEngine(
        generator=generator,
        tokenizer=TOKENIZER
    )


def main():

    chat = build_chat()

    print("=" * 50)
    print("Neura Chat")
    print("=" * 50)
    print("Type /exit to quit.")
    print()

    while True:

        try:
            text = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not text.strip():
            continue

        if text.strip().lower() == "/exit":
            break

        try:
            reply = chat.chat(
                text,
                max_tokens=32
            )
        except Exception as exc:
            print("Neura: [ERROR]", exc)
            print()
            continue

        print("Neura:", reply)
        print()


if __name__ == "__main__":
    main()
