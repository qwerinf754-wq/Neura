from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

from neura.inference.sft_runtime import SFTRuntime
from neura.rag import ContextBuilder, Document, NativeRAGStore


CHECKPOINT = Path("checkpoints/neura_chat_sft_best.pt")

print("=== RAG → REAL SFT INTEGRATION ===")

assert CHECKPOINT.exists(), f"Missing checkpoint: {CHECKPOINT}"

with TemporaryDirectory() as directory:
    store = NativeRAGStore(
        db_path=Path(directory) / "rag.sqlite3"
    )

    store.add_documents([
        Document(
            doc_id="python",
            text="Python is a programming language used for software development and AI.",
            metadata="documentation",
        ),
        Document(
            doc_id="neura",
            text="Neura is an AI project with native retrieval and inference components.",
            metadata="project",
        ),
        Document(
            doc_id="database",
            text="SQLite provides lightweight persistent storage for local applications.",
            metadata="documentation",
        ),
    ])

    results = store.search_hybrid(
        "Python programming AI",
        top_k=2,
        vector_weight=0.45,
        bm25_weight=0.55,
    )

    assert results
    print("HYBRID RETRIEVAL: PASS")
    print("TOP SOURCE:", results[0].doc_id)

    builder = ContextBuilder(max_characters=4000)

    context = builder.build(
        "Python programming AI",
        results,
    )

    assert not context.is_empty
    assert context.chunks
    assert context.total_characters > 0

    print("CONTEXT BUILD: PASS")
    print("CHUNKS:", len(context.chunks))
    print("CHARACTERS:", context.total_characters)

    prompt = (
        "Use the following retrieved context to answer the user query.\n\n"
        f"Context:\n{context.text}\n\n"
        "User query:\nPython programming AI\n\n"
        "Answer:"
    )

    assert context.text in prompt
    print("PROMPT CONSTRUCTION: PASS")
    print("PROMPT CHARACTERS:", len(prompt))

    runtime = SFTRuntime(
        checkpoint_path=CHECKPOINT,
        device="cpu",
    )

    assert runtime.loaded
    assert runtime.model is not None
    assert runtime.tokenizer is not None

    print("REAL RUNTIME LOAD: PASS")
    print("VOCAB SIZE:", runtime.vocab_size)
    print("CONTEXT LENGTH:", runtime.context_length)

    prompt_tokens = runtime.encode(prompt)

    assert prompt_tokens
    assert len(prompt_tokens) < runtime.context_length

    print("RUNTIME TOKENIZATION: PASS")
    print("PROMPT TOKENS:", len(prompt_tokens))

    result = runtime.generate(
        prompt,
        max_new_tokens=8,
        temperature=0.0,
    )

    assert result.prompt_tokens == prompt_tokens
    assert len(result.generated_tokens) <= 8
    assert result.finish_reason in {
        "eos",
        "length",
        "repetition",
    }

    generated_text = result.text
    assert isinstance(generated_text, str)

    print("REAL GENERATION: PASS")
    print("GENERATED TOKENS:", len(result.generated_tokens))
    print("GENERATED TEXT:", repr(generated_text))
    print("FINISH:", result.finish_reason)

    sources = [chunk.doc_id for chunk in context.chunks]

    assert sources
    assert "python" in sources

    print("SOURCE PRESERVATION: PASS")
    print("SOURCES:", sources)

print()
print("=== RAG → REAL SFT INTEGRATION: PASS ===")
