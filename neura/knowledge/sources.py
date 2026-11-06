"""
Neura Knowledge Sources
=======================

Trusted learning and development sources used by Neura.
Neura stores source metadata and fetches content on demand.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeSource:
    name: str
    url: str
    category: str
    priority: int = 50
    languages: tuple[str, ...] = ("en",)


GLOBAL_KNOWLEDGE_SOURCES = (
    KnowledgeSource(
        "Python Documentation",
        "https://docs.python.org/3/",
        "programming",
        100,
        ("en",),
    ),
    KnowledgeSource(
        "PyTorch Documentation",
        "https://docs.pytorch.org/docs/main/",
        "machine_learning",
        100,
        ("en",),
    ),
    KnowledgeSource(
        "PyTorch Tutorials",
        "https://docs.pytorch.org/tutorials/",
        "machine_learning",
        98,
        ("en",),
    ),
    KnowledgeSource(
        "Hugging Face Documentation",
        "https://huggingface.co/docs",
        "artificial_intelligence",
        100,
        ("en",),
    ),
    KnowledgeSource(
        "Hugging Face Hub",
        "https://huggingface.co/docs/hub/",
        "models_datasets",
        98,
        ("en",),
    ),
    KnowledgeSource(
        "MDN Web Docs",
        "https://developer.mozilla.org/en-US/",
        "web_development",
        100,
        ("en",),
    ),
    KnowledgeSource(
        "MDN Learn Web Development",
        "https://developer.mozilla.org/en-US/docs/Learn_web_development",
        "web_development",
        98,
        ("en",),
    ),
    KnowledgeSource(
        "GitHub",
        "https://github.com/",
        "software",
        100,
        ("en",),
    ),
    KnowledgeSource(
        "arXiv",
        "https://arxiv.org/",
        "research",
        95,
        ("en",),
    ),
    KnowledgeSource(
        "Kaggle",
        "https://www.kaggle.com/",
        "data_science",
        90,
        ("en",),
    ),
    KnowledgeSource(
        "Stack Overflow",
        "https://stackoverflow.com/",
        "programming",
        85,
        ("en",),
    ),
    KnowledgeSource(
        "Wikipedia",
        "https://www.wikipedia.org/",
        "general_knowledge",
        70,
        ("ar", "en", "fr", "de", "es", "it", "tr", "fa"),
    ),
)


def list_sources() -> list[KnowledgeSource]:
    return sorted(
        GLOBAL_KNOWLEDGE_SOURCES,
        key=lambda source: source.priority,
        reverse=True,
    )


def find_sources(category: str | None = None) -> list[KnowledgeSource]:
    if not category:
        return list_sources()

    category = category.strip().lower()

    return [
        source
        for source in list_sources()
        if source.category == category
    ]


class KnowledgeSourceResolver:
    """
    Resolves user queries to curated global knowledge sources.

    This layer only selects trusted source metadata.
    Content is fetched on demand by the web/tool layer.
    """

    CATEGORY_KEYWORDS = {
        "programming": (
            "python",
            "javascript",
            "typescript",
            "java",
            "rust",
            "code",
            "coding",
            "programming",
            "برمجة",
            "بايثون",
            "كود",
            "برمج",
        ),
        "machine_learning": (
            "pytorch",
            "deep learning",
            "machine learning",
            "neural network",
            "transformer",
            "ذكاء اصطناعي",
            "تعلم آلي",
            "تعلم عميق",
            "شبكة عصبية",
            "transformer",
        ),
        "artificial_intelligence": (
            "artificial intelligence",
            "ai",
            "llm",
            "language model",
            "generative ai",
            "ذكاء اصطناعي",
            "نموذج لغوي",
            "نموذج ذكاء",
        ),
        "models_datasets": (
            "hugging face",
            "huggingface",
            "dataset",
            "datasets",
            "model hub",
            "نموذج",
            "نماذج",
            "بيانات",
        ),
        "web_development": (
            "html",
            "css",
            "javascript",
            "web",
            "website",
            "frontend",
            "backend",
            "موقع",
            "مواقع",
            "ويب",
        ),
        "software": (
            "github",
            "repository",
            "repo",
            "software",
            "source code",
            "مستودع",
            "كود المصدر",
        ),
        "research": (
            "paper",
            "papers",
            "research",
            "arxiv",
            "study",
            "بحث علمي",
            "ورقة بحثية",
            "دراسة",
        ),
        "data_science": (
            "data science",
            "dataset",
            "kaggle",
            "statistics",
            "تحليل البيانات",
            "علم البيانات",
            "إحصاء",
        ),
        "general_knowledge": (
            "wikipedia",
            "history",
            "geography",
            "science",
            "who is",
            "what is",
            "تاريخ",
            "جغرافيا",
            "علوم",
            "من هو",
            "ما هو",
            "ما هي",
        ),
    }

    def resolve(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 5,
    ) -> list[KnowledgeSource]:
        """
        Return the most relevant curated sources for a query.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        query = query.strip().lower()

        if limit < 1:
            return []

        if category:
            candidates = find_sources(category)
        else:
            candidates = list_sources()

        if not query:
            return candidates[:limit]

        scored = []

        for source in candidates:
            score = source.priority

            keywords = self.CATEGORY_KEYWORDS.get(
                source.category,
                (),
            )

            for keyword in keywords:
                if keyword in query:
                    score += 20
                    break

            if source.name.lower() in query:
                score += 50

            if any(
                language in query
                for language in ("arabic", "العربية")
            ) and "ar" in source.languages:
                score += 10

            scored.append((score, source))

        scored.sort(
            key=lambda item: (
                item[0],
                item[1].priority,
                item[1].name,
            ),
            reverse=True,
        )

        return [
            source
            for _, source in scored[:limit]
        ]

    def get_urls(
        self,
        query: str,
        *,
        category: str | None = None,
        limit: int = 5,
    ) -> list[str]:
        """
        Return URLs for sources relevant to the query.
        """

        return [
            source.url
            for source in self.resolve(
                query,
                category=category,
                limit=limit,
            )
        ]


KNOWLEDGE_SOURCE_RESOLVER = KnowledgeSourceResolver()
