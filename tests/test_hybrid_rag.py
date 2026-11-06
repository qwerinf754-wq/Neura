from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from neura.rag import Document, NativeRAGStore


def test_hybrid_rag() -> None:
    with TemporaryDirectory() as tmp:
        store = NativeRAGStore(Path(tmp) / "hybrid.db")

        store.add_documents(
            [
                Document(
                    "python",
                    "Python programming language software development",
                ),
                Document(
                    "python-ai",
                    "Python artificial intelligence machine learning",
                ),
                Document(
                    "iraq",
                    "Iraq history culture heritage",
                ),
                Document(
                    "neura",
                    "Neura artificial intelligence project",
                ),
            ]
        )

        assert store.count() == 4

        bm25 = store.search_bm25(
            "Python programming",
            top_k=3,
        )

        assert bm25
        assert bm25[0].doc_id == "python"

        hybrid = store.search_hybrid(
            "Python programming",
            top_k=3,
        )

        assert hybrid
        assert hybrid[0].doc_id == "python"
        assert 0.0 <= hybrid[0].score <= 1.0

        print("BM25 SEARCH: PASS")
        print("HYBRID SEARCH: PASS")
        print("DOCUMENTS:", store.count())
        print("BM25 TOP:", bm25[0].doc_id)
        print("HYBRID TOP:", hybrid[0].doc_id)
        print("HYBRID SCORE:", round(hybrid[0].score, 4))


if __name__ == "__main__":
    test_hybrid_rag()
