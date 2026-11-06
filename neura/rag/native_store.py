from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Document:
    doc_id: str
    text: str
    metadata: str = ""


@dataclass(frozen=True)
class SearchResult:
    doc_id: str
    text: str
    score: float
    metadata: str


class NativeRAGStore:
    """
    Lightweight Android-friendly RAG store.

    Uses SQLite for persistence and pure Python cosine similarity
    over sparse token-frequency vectors. No FAISS or sentence-transformers
    dependency is required.
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA foreign_keys = ON")
        return conn

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()

    @classmethod
    def _vector(cls, text: str) -> dict[str, float]:
        counts: dict[str, float] = {}

        for token in cls._tokenize(text):
            counts[token] = counts.get(token, 0.0) + 1.0

        norm = math.sqrt(sum(value * value for value in counts.values()))

        if norm == 0:
            return {}

        return {
            token: value / norm
            for token, value in counts.items()
        }

    @staticmethod
    def _cosine(
        query: dict[str, float],
        document: dict[str, float],
    ) -> float:
        if not query or not document:
            return 0.0

        return sum(
            query[token] * document.get(token, 0.0)
            for token in query
        )

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS rag_documents (
                    doc_id TEXT PRIMARY KEY,
                    text TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '',
                    vector TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def add_document(
        self,
        doc_id: str,
        text: str,
        metadata: str = "",
    ) -> None:
        if not doc_id:
            raise ValueError("doc_id must not be empty")

        if not text:
            raise ValueError("text must not be empty")

        vector = self._vector(text)

        # Compact JSON-like storage without an external dependency.
        import json

        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO rag_documents(doc_id, text, metadata, vector)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(doc_id) DO UPDATE SET
                    text = excluded.text,
                    metadata = excluded.metadata,
                    vector = excluded.vector
                """,
                (
                    doc_id,
                    text,
                    metadata,
                    json.dumps(vector, ensure_ascii=False),
                ),
            )
            conn.commit()

    def add_documents(self, documents: list[Document]) -> None:
        for document in documents:
            self.add_document(
                document.doc_id,
                document.text,
                document.metadata,
            )


    @staticmethod
    def _bm25_score(
        query_tokens: list[str],
        document_tokens: list[str],
        document_frequency: dict[str, int],
        document_count: int,
        avg_document_length: float,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> float:
        """
        Pure-Python BM25 score.

        This implementation is intentionally lightweight and Android-friendly.
        """
        if not query_tokens or not document_tokens or document_count <= 0:
            return 0.0

        term_frequency: dict[str, int] = {}

        for token in document_tokens:
            term_frequency[token] = term_frequency.get(token, 0) + 1

        document_length = len(document_tokens)

        if avg_document_length <= 0:
            avg_document_length = 1.0

        score = 0.0

        for token in query_tokens:
            tf = term_frequency.get(token, 0)

            if tf == 0:
                continue

            df = document_frequency.get(token, 0)

            # BM25 inverse document frequency.
            idf = math.log(
                1.0
                + (
                    (document_count - df + 0.5)
                    / (df + 0.5)
                )
            )

            numerator = tf * (k1 + 1.0)

            denominator = (
                tf
                + k1
                * (
                    1.0
                    - b
                    + b
                    * (
                        document_length
                        / avg_document_length
                    )
                )
            )

            score += idf * (numerator / denominator)

        return score

    def search_bm25(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        """
        Search documents using BM25.
        """
        if not query or top_k <= 0:
            return []

        import json

        query_tokens = self._tokenize(query)

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, text, metadata, vector
                FROM rag_documents
                """
            ).fetchall()

        if not rows:
            return []

        tokenized_documents = []

        for doc_id, text, metadata, vector_json in rows:
            tokens = self._tokenize(text)
            tokenized_documents.append(
                (
                    doc_id,
                    text,
                    metadata,
                    vector_json,
                    tokens,
                )
            )

        document_count = len(tokenized_documents)

        document_frequency: dict[str, int] = {}

        total_length = 0

        for _, _, _, _, tokens in tokenized_documents:
            total_length += len(tokens)

            for token in set(tokens):
                document_frequency[token] = (
                    document_frequency.get(token, 0) + 1
                )

        avg_document_length = (
            total_length / document_count
            if document_count
            else 1.0
        )

        results: list[SearchResult] = []

        for (
            doc_id,
            text,
            metadata,
            vector_json,
            tokens,
        ) in tokenized_documents:
            score = self._bm25_score(
                query_tokens=query_tokens,
                document_tokens=tokens,
                document_frequency=document_frequency,
                document_count=document_count,
                avg_document_length=avg_document_length,
            )

            if score > 0:
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        text=text,
                        score=score,
                        metadata=metadata,
                    )
                )

        results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return results[:top_k]

    def search_hybrid(
        self,
        query: str,
        top_k: int = 5,
        vector_weight: float = 0.45,
        bm25_weight: float = 0.55,
    ) -> list[SearchResult]:
        """
        Combine native cosine retrieval and BM25.

        Scores are min-max normalized independently before fusion.
        """
        if not query or top_k <= 0:
            return []

        if vector_weight < 0 or bm25_weight < 0:
            raise ValueError("weights must be non-negative")

        weight_sum = vector_weight + bm25_weight

        if weight_sum <= 0:
            raise ValueError("at least one weight must be positive")

        vector_weight /= weight_sum
        bm25_weight /= weight_sum

        vector_results = self.search(
            query,
            top_k=max(top_k, self.count()),
        )

        bm25_results = self.search_bm25(
            query,
            top_k=max(top_k, self.count()),
        )

        vector_scores = {
            item.doc_id: item.score
            for item in vector_results
        }

        bm25_scores = {
            item.doc_id: item.score
            for item in bm25_results
        }

        def normalize(scores: dict[str, float]) -> dict[str, float]:
            if not scores:
                return {}

            values = list(scores.values())
            minimum = min(values)
            maximum = max(values)

            if math.isclose(minimum, maximum):
                return {
                    key: 1.0
                    for key in scores
                }

            span = maximum - minimum

            return {
                key: (value - minimum) / span
                for key, value in scores.items()
            }

        vector_normalized = normalize(vector_scores)
        bm25_normalized = normalize(bm25_scores)

        documents = {}

        for item in vector_results:
            documents[item.doc_id] = item

        for item in bm25_results:
            documents[item.doc_id] = item

        fused: list[SearchResult] = []

        for doc_id, item in documents.items():
            score = (
                vector_weight
                * vector_normalized.get(doc_id, 0.0)
                + bm25_weight
                * bm25_normalized.get(doc_id, 0.0)
            )

            fused.append(
                SearchResult(
                    doc_id=item.doc_id,
                    text=item.text,
                    score=score,
                    metadata=item.metadata,
                )
            )

        fused.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return fused[:top_k]

    def delete_document(self, doc_id: str) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "DELETE FROM rag_documents WHERE doc_id = ?",
                (doc_id,),
            )
            conn.commit()
            return cursor.rowcount > 0

    def count(self) -> int:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM rag_documents"
            ).fetchone()

        return int(row[0])

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:
        if not query:
            return []

        if top_k <= 0:
            return []

        import json

        query_vector = self._vector(query)
        results: list[SearchResult] = []

        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT doc_id, text, metadata, vector
                FROM rag_documents
                """
            ).fetchall()

        for doc_id, text, metadata, vector_json in rows:
            vector = json.loads(vector_json)

            score = self._cosine(
                query_vector,
                vector,
            )

            if score > 0:
                results.append(
                    SearchResult(
                        doc_id=doc_id,
                        text=text,
                        score=score,
                        metadata=metadata,
                    )
                )

        results.sort(
            key=lambda item: item.score,
            reverse=True,
        )

        return results[:top_k]
