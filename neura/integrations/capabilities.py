"""Capability definitions for optional Neura integrations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CapabilityInfo:
    """Describes an optional capability and its availability."""

    name: str
    package: str
    available: bool
    required: bool = False
    description: str = ""


def _package_available(package: str) -> bool:
    """Return True when a Python package can be imported."""
    try:
        __import__(package)
    except (ImportError, ModuleNotFoundError):
        return False
    except Exception:
        # A broken optional package should not crash Neura's registry.
        return False
    return True


_CAPABILITIES: tuple[CapabilityInfo, ...] = (
    CapabilityInfo(
        name="pytorch",
        package="torch",
        available=_package_available("torch"),
        required=True,
        description="Neura model execution and training backend.",
    ),
    CapabilityInfo(
        name="tokenizers",
        package="tokenizers",
        available=_package_available("tokenizers"),
        description=(
            "Optional Hugging Face tokenizer backend. "
            "Not currently available on this Android/aarch64 "
            "Python 3.13 environment; Neura Native BPE remains "
            "the primary Android tokenizer."
        ),
    ),
    CapabilityInfo(
        name="transformers",
        package="transformers",
        available=_package_available("transformers"),
        description="Optional transformer model interoperability.",
    ),
    CapabilityInfo(
        name="faiss",
        package="faiss",
        available=_package_available("faiss"),
        description="Optional vector similarity search backend.",
    ),
    CapabilityInfo(
        name="sentence_transformers",
        package="sentence_transformers",
        available=_package_available("sentence_transformers"),
        description="Optional embedding backend for RAG.",
    ),
    CapabilityInfo(
        name="fastapi",
        package="fastapi",
        available=_package_available("fastapi"),
        description="Optional FastAPI application backend.",
    ),
    CapabilityInfo(
        name="pydantic",
        package="pydantic",
        available=_package_available("pydantic"),
        description="Optional structured data validation.",
    ),
    CapabilityInfo(
        name="httpx",
        package="httpx",
        available=_package_available("httpx"),
        description="Optional async HTTP client.",
    ),
    CapabilityInfo(
        name="fitz",
        package="fitz",
        available=_package_available("fitz"),
        description="Optional PDF processing through PyMuPDF.",
    ),
    CapabilityInfo(
        name="docx",
        package="docx",
        available=_package_available("docx"),
        description="Optional DOCX processing.",
    ),
    CapabilityInfo(
        name="openpyxl",
        package="openpyxl",
        available=_package_available("openpyxl"),
        description="Optional XLSX processing.",
    ),
    CapabilityInfo(
        name="llama_cpp",
        package="llama_cpp",
        available=_package_available("llama_cpp"),
        description="Optional CPU-oriented inference backend.",
    ),
)


def list_capabilities() -> tuple[CapabilityInfo, ...]:
    """Return the current capability snapshot."""
    return _CAPABILITIES


def get_capability(name: str) -> CapabilityInfo | None:
    """Return a capability by name."""
    for capability in _CAPABILITIES:
        if capability.name == name:
            return capability
    return None
