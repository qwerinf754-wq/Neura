from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from neura.rag import (
    ContextBuilder,
    Document,
    NativeRAGStore,
)


def test_context_builder() -> None:
    with TemporaryDirectory() as tmp:
        store = NativeRAGStore(
            Path(tmp) / "context.db"
        )

        store.add_documents(
            [
                Document(
                    "python",
                    "Python programming language and software development",
                    "language",
                ),
                Document(
                    "neura",
                    "Neura is an artificial intelligence project",
                    "project",
                ),
                Document(
                    "ai",
                    "Artificial intelligence uses machine learning",
                    "technology",
                ),
            ]
        )

        builder = ContextBuilder(
            max_characters=300
        )

        context = builder.build_from_store(
            store,
            "Python programming",
            top_k=3,
        )

        assert context.query == "Python programming"
        assert not context.is_empty
        assert context.chunks
        assert context.chunks[0].doc_id == "python"
        assert context.total_characters <= 300
        assert "[Source: python]" in context.text
        assert "language" in context.text

        # Verify strict bounding.
        tiny = ContextBuilder(
            max_characters=50
        )

        tiny_context = tiny.build(
            "test",
            context.chunks and [
                type(
                    "Result",
                    (),
                    {
                        "doc_id": "long",
                        "text": "x" * 500,
                        "score": 1.0,
                        "metadata": "",
                    },
                )()
            ],
        )

        assert tiny_context.total_characters <= 50

        print("CONTEXT BUILDER: PASS")
        print("CHUNKS:", len(context.chunks))
        print("CHARACTERS:", context.total_characters)
        print("TOP SOURCE:", context.chunks[0].doc_id)


if __name__ == "__main__":
    test_context_builder()
