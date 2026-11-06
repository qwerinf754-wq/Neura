from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from neura.rag import Document, NativeRAGStore


def test_native_rag() -> None:
    with TemporaryDirectory() as tmp:
        store = NativeRAGStore(Path(tmp) / "rag.db")

        store.add_documents(
            [
                Document(
                    "python",
                    "Python programming language and software development",
                ),
                Document(
                    "iraq",
                    "Iraq has a rich history and cultural heritage",
                ),
                Document(
                    "neura",
                    "Neura is an artificial intelligence project",
                ),
            ]
        )

        assert store.count() == 3

        results = store.search("Python programming", top_k=2)

        assert results
        assert results[0].doc_id == "python"
        assert results[0].score > 0

        assert store.delete_document("iraq") is True
        assert store.count() == 2

        print("NATIVE RAG TEST: PASS")
        print("DOCUMENTS:", store.count())
        print("TOP RESULT:", results[0].doc_id)
        print("SCORE:", round(results[0].score, 4))


if __name__ == "__main__":
    test_native_rag()
