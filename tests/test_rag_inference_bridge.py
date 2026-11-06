from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from neura.inference import InferenceResponse
from neura.rag import (
    ContextBuilder,
    Document,
    NativeRAGStore,
    RAGInferenceBridge,
)


class MockEngine:
    """Minimal engine implementing the contract used by the bridge."""

    def __init__(self):
        self.requests = []

    def generate(self, request):
        self.requests.append(request)

        # Return a deterministic token sequence from the prompt tail.
        generated = list(request.prompt_tokens[-2:])

        return InferenceResponse(
            prompt_tokens=list(request.prompt_tokens),
            generated_tokens=generated,
            finish_reason="length",
        )


def main():
    with TemporaryDirectory() as tmp:
        db = Path(tmp) / "rag.sqlite"

        store = NativeRAGStore(db)

        store.add_documents([
            Document(
                doc_id="python",
                text="Python is a programming language used for AI and machine learning.",
                metadata="documentation",
            ),
            Document(
                doc_id="rag",
                text="Retrieval augmented generation uses retrieved documents as context.",
                metadata="documentation",
            ),
        ])

        engine = MockEngine()

        bridge = RAGInferenceBridge(
            engine=engine,
            store=store,
            context_builder=ContextBuilder(max_characters=4000),
        )

        result = bridge.generate(
            "What is Python?",
            max_tokens=4,
            retrieval_top_k=2,
        )

        assert result.query == "What is Python?"
        assert result.prompt
        assert result.context is not None
        assert not result.context.is_empty

        assert result.generated_tokens
        assert result.generated_text is not None
        assert result.finish_reason == "length"

        assert len(engine.requests) == 1

        request = engine.requests[0]

        assert request.prompt_tokens
        assert request.max_tokens == 4
        assert request.prompt_tokens == list(
            request.prompt_tokens
        )

        assert result.sources

        print("RAG → INFERENCE MOCK: PASS")
        print("PROMPT TOKENS:", len(request.prompt_tokens))
        print("GENERATED TOKENS:", len(result.generated_tokens))
        print("SOURCES:", result.sources)
        print("FINISH:", result.finish_reason)


if __name__ == "__main__":
    main()
