from __future__ import annotations

from dataclasses import dataclass

from .native_store import SearchResult


@dataclass(frozen=True)
class ContextChunk:
    doc_id: str
    text: str
    score: float
    metadata: str
    position: int


@dataclass(frozen=True)
class RAGContext:
    query: str
    chunks: list[ContextChunk]
    text: str
    total_characters: int

    @property
    def is_empty(self) -> bool:
        return not self.chunks


class ContextBuilder:
    """
    Converts retrieved RAG results into bounded context for Neura.

    The builder is intentionally independent from any tokenizer so it can
    run on Android before a model-specific tokenizer is available.
    """

    def __init__(
        self,
        max_characters: int = 12000,
        separator: str = "\n\n",
    ):
        if max_characters <= 0:
            raise ValueError("max_characters must be positive")

        if not separator:
            raise ValueError("separator must not be empty")

        self.max_characters = max_characters
        self.separator = separator

    @staticmethod
    def _clean_text(text: str) -> str:
        return " ".join(text.split())

    @staticmethod
    def _clean_metadata(metadata: str) -> str:
        return " ".join(metadata.split())

    def build(
        self,
        query: str,
        results: list[SearchResult],
    ) -> RAGContext:
        if not query:
            return RAGContext(
                query="",
                chunks=[],
                text="",
                total_characters=0,
            )

        chunks: list[ContextChunk] = []
        parts: list[str] = []
        total = 0

        for result in results:
            text = self._clean_text(result.text)

            if not text:
                continue

            metadata = self._clean_metadata(result.metadata)

            header = f"[Source: {result.doc_id}]"

            if metadata:
                header += f" [Metadata: {metadata}]"

            block = f"{header}\n{text}"

            additional = len(block)

            if parts:
                additional += len(self.separator)

            if total + additional > self.max_characters:
                remaining = (
                    self.max_characters
                    - total
                    - (
                        len(self.separator)
                        if parts
                        else 0
                    )
                )

                if remaining <= 0:
                    break

                text_prefix = text[:remaining]

                if not text_prefix:
                    break

                block = f"{header}\n{text_prefix}"

                additional = len(block)

                if parts:
                    additional += len(self.separator)

                if total + additional > self.max_characters:
                    break

                text = text_prefix

            position = len(chunks) + 1

            chunks.append(
                ContextChunk(
                    doc_id=result.doc_id,
                    text=text,
                    score=result.score,
                    metadata=metadata,
                    position=position,
                )
            )

            parts.append(block)
            total += additional

            if total >= self.max_characters:
                break

        return RAGContext(
            query=query,
            chunks=chunks,
            text=self.separator.join(parts),
            total_characters=total,
        )

    def build_from_store(
        self,
        store,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.45,
        bm25_weight: float = 0.55,
    ) -> RAGContext:
        results = store.search_hybrid(
            query=query,
            top_k=top_k,
            vector_weight=vector_weight,
            bm25_weight=bm25_weight,
        )

        return self.build(
            query=query,
            results=results,
        )
