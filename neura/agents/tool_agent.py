"""
Neura Tool Agent

Routes user requests to the appropriate external tool.
"""

import re

from neura.tools import ToolRouter
from neura.agents.web_context import WebContextBuilder
from neura.rag.wikipedia_retriever import WikipediaRetriever


class ToolAgent:

    def __init__(
        self,
        router=None,
        wikipedia_retriever=None,
    ):
        self.router = (
            router
            if router is not None
            else ToolRouter()
        )

        self.wikipedia_retriever = (
            wikipedia_retriever
            if wikipedia_retriever is not None
            else WikipediaRetriever()
        )

    def should_search(self, text):
        text = text.lower().strip()

        triggers = [
            # Explicit web requests
            "search",
            "web",
            "internet",
            "online",
            "look up",
            "find online",

            # Fresh information
            "latest",
            "current",
            "today",
            "recent",
            "news",
            "now",

            # Arabic
            "ابحث",
            "الإنترنت",
            "الويب",
            "بحث",
            "أحدث",
            "آخر",
            "اليوم",
            "أخبار",
            "حالي",
            "حديث",

            # Question patterns
            "what is",
            "who is",
            "who was",
            "where is",
            "when did",
            "when was",
            "how does",
            "how do",
            "what are",

            "ما هو",
            "ما هي",
            "من هو",
            "من هي",
            "أين",
            "متى",
            "كيف",
        ]

        return any(
            trigger in text
            for trigger in triggers
        )

    def should_use_github(self, text):
        text = text.lower().strip()

        triggers = [
            "github",
            "git hub",
            "repository",
            "repo",
            "repos",
            "pull request",
            "issue",
            "release",
            "commit",
            "branch",
            "readme",
            "source code",
            "source-code",
            "github.com",

            "مستودع",
            "مستودعات",
            "كود المصدر",
            "شفرة المصدر",
            "فرع",
            "إصدار",
            "كوميت",
        ]

        if any(
            trigger in text
            for trigger in triggers
        ):
            return True

        # owner/repository pattern
        if re.search(
            r"\b[\w.-]+/[\w.-]+\b",
            text,
        ):
            return True

        return False

    def should_use_official_site(self, text):
        text = text.lower().strip()

        triggers = [
            "official documentation",
            "official docs",
            "official website",
            "official site",
            "documentation",
            "docs",
            "documentation site",

            "التوثيق الرسمي",
            "التوثيق",
            "الموقع الرسمي",
            "الموقع الرسمي لـ",
        ]

        if any(
            trigger in text
            for trigger in triggers
        ):
            return True

        known_sites = [
            "python",
            "pytorch",
            "hugging face",
            "huggingface",
            "mdn",
            "numpy",
            "scipy",
            "rust",
            "docker",
            "wikipedia",
            "pypi",
            "linux",
        ]

        # Only classify a known technology/site as
        # official-site when the request also looks
        # documentation-oriented.
        doc_triggers = [
            "docs",
            "documentation",
            "official",
            "api",
            "reference",
            "manual",
            "توثيق",
            "رسمي",
            "مرجع",
            "دليل",
        ]

        return (
            any(
                site in text
                for site in known_sites
            )
            and any(
                trigger in text
                for trigger in doc_triggers
            )
        )

    def _official_site_name(self, text):
        text = text.lower()

        aliases = {
            "python": "python",
            "pytorch": "pytorch",
            "hugging face": "huggingface",
            "huggingface": "huggingface",
            "github": "github",
            "mdn": "mdn",
            "numpy": "numpy",
            "scipy": "scipy",
            "rust": "rust",
            "docker": "docker",
            "wikipedia": "wikipedia",
            "pypi": "python_pypi",
            "linux": "linux",
        }

        for alias, site_name in aliases.items():
            if alias in text:
                return site_name

        return None

    def search(
        self,
        query,
        max_results=5,
    ):
        """
        Preserve the original public search API.
        """
        return self.router.call(
            "web_search",
            query=query,
            max_results=max_results,
        )

    def github_query(self, text):
        """Extract a useful GitHub search query from user text."""
        query = text.strip()

        patterns = [
            r"^ابحث\s+(?:في\s+)?github\s+(?:عن\s+)?",
            r"^ابحث\s+(?:في\s+)?GitHub\s+(?:عن\s+)?",
            r"^search\s+(?:github\s+)?(?:for\s+)?",
            r"^find\s+(?:on\s+)?github\s+(?:for\s+)?",
            r"^look\s+up\s+(?:on\s+)?github\s+(?:for\s+)?",
        ]

        for pattern in patterns:
            query = re.sub(
                pattern,
                "",
                query,
                flags=re.IGNORECASE,
            )

        query = re.sub(
            r"\bgithub\.com\b",
            "",
            query,
            flags=re.IGNORECASE,
        )

        query = re.sub(
            r"\s+",
            " ",
            query,
        ).strip()

        query = re.sub(
            r"^(?:عن|for)\s+",
            "",
            query,
            flags=re.IGNORECASE,
        ).strip()

        return query or text.strip()

    def github_search(
        self,
        query,
        max_results=5,
    ):
        github_query = self.github_query(query)

        return self.router.call(
            "github",
            operation="search",
            query=github_query,
            max_results=max_results,
        )

    def official_site(
        self,
        name,
        page="url",
    ):
        return self.router.call(
            "official_site",
            operation="fetch",
            name=name,
            page=page,
        )

    def should_use_knowledge_source(self, text):
        """
        Detect explicit requests for curated learning resources.
        """
        text = text.lower().strip()

        learning_triggers = (
            "learn",
            "learning",
            "tutorial",
            "tutorials",
            "course",
            "courses",
            "study",
            "how to learn",
            "resources to learn",
            "مصادر للتعلم",
            "مصادر تعلم",
            "مصادر للتعلّم",
            "اريد تعلم",
            "أريد تعلم",
            "اريد أن أتعلم",
            "أريد أن أتعلم",
            "اريد ان اتعلم",
            "أريد ان أتعلم",
            "تعلم",
            "تعلّم",
            "اتعلم",
            "أتعلم",
            "دروس",
            "دورة",
            "دورات",
            "كورس",
            "كورسات",
            "شرح",
            "دليل",
        )

        if not any(
            trigger in text
            for trigger in learning_triggers
        ):
            return False

        source_keywords = (
            "python",
            "pytorch",
            "hugging face",
            "huggingface",
            "javascript",
            "typescript",
            "html",
            "css",
            "web development",
            "machine learning",
            "deep learning",
            "artificial intelligence",
            "ai",
            "llm",
            "transformer",
            "github",
            "rust",
            "docker",
            "numpy",
            "scipy",
            "kaggle",
            "arxiv",
            "data science",
            "برمجة",
            "بايثون",
            "بايتورتش",
            "ذكاء اصطناعي",
            "الذكاء الاصطناعي",
            "تعلم آلي",
            "تعلم عميق",
            "تطوير الويب",
            "تطوير المواقع",
            "نماذج لغوية",
            "محولات",
            "علم البيانات",
            "بحث علمي",
        )

        return any(
            keyword in text
            for keyword in source_keywords
        )

    def knowledge_source_category(self, text):
        """
        Determine the most specific curated source category.
        """
        text = text.lower().strip()

        category_map = {
            "programming": (
                "python",
                "javascript",
                "typescript",
                "rust",
                "برمجة",
                "بايثون",
            ),
            "machine_learning": (
                "pytorch",
                "machine learning",
                "deep learning",
                "neural network",
                "transformer",
                "تعلم آلي",
                "تعلم عميق",
                "بايتورتش",
                "شبكة عصبية",
                "محولات",
            ),
            "artificial_intelligence": (
                "artificial intelligence",
                "ai",
                "llm",
                "language model",
                "generative ai",
                "ذكاء اصطناعي",
                "الذكاء الاصطناعي",
                "نموذج لغوي",
            ),
            "models_datasets": (
                "hugging face",
                "huggingface",
                "dataset",
                "datasets",
                "model hub",
                "نماذج",
                "بيانات",
            ),
            "web_development": (
                "web development",
                "website",
                "frontend",
                "backend",
                "html",
                "css",
                "javascript",
                "تطوير الويب",
                "تطوير المواقع",
                "موقع",
            ),
            "research": (
                "arxiv",
                "research",
                "paper",
                "papers",
                "scientific",
                "بحث علمي",
                "ورقة بحثية",
            ),
            "data_science": (
                "data science",
                "kaggle",
                "statistics",
                "علم البيانات",
                "تحليل البيانات",
                "إحصاء",
            ),
        }

        for category, keywords in category_map.items():
            if any(
                keyword in text
                for keyword in keywords
            ):
                return category

        return None

    def should_use_wikipedia(self, text):
        """
        Detect stable factual/knowledge questions suitable for
        Wikipedia retrieval.

        Fresh/current/news requests remain on the normal web path.
        Explicit Wikipedia requests always use Wikipedia.
        """
        text = text.lower().strip()

        explicit_triggers = [
            "wikipedia",
            "ويكيبيديا",
        ]

        if any(trigger in text for trigger in explicit_triggers):
            return True

        fresh_triggers = [
            "latest",
            "current",
            "today",
            "recent",
            "news",
            "now",
            "أحدث",
            "آخر الأخبار",
            "آخر",
            "اليوم",
            "أخبار",
            "حالي",
            "حاليا",
            "حديث",
            "الآن",
        ]

        if any(trigger in text for trigger in fresh_triggers):
            return False

        factual_triggers = [
            "what is",
            "who is",
            "who was",
            "where is",
            "when was",
            "how does",
            "what are",
            "ما هو",
            "ما هي",
            "من هو",
            "من هي",
            "أين",
            "متى",
            "كيف",
        ]

        return any(trigger in text for trigger in factual_triggers)

    def wikipedia_query(self, text):
        """Extract a clean query for Wikipedia retrieval."""
        query = text.strip()

        query = re.sub(
            r"\b(?:wikipedia)\b",
            "",
            query,
            flags=re.IGNORECASE,
        )

        query = re.sub(
            r"\b(?:في|على)\s+ويكيبيديا\b",
            "",
            query,
            flags=re.IGNORECASE,
        )

        query = re.sub(
            r"(?:في|على)\s+ويكيبيديا",
            "",
            query,
            flags=re.IGNORECASE,
        )

        query = re.sub(r"\s+", " ", query).strip()

        return query or text.strip()

    def wikipedia_retrieve(
        self,
        query,
        language="ar",
        top_k=3,
    ):
        """
        Retrieve ranked Wikipedia passages and convert them into
        the common document contract consumed by the orchestrator.
        """
        chunks = self.wikipedia_retriever.retrieve(
            query,
            language=language,
            max_results=max(top_k, 5),
            top_k=top_k,
        )

        documents = []

        for chunk in chunks:
            documents.append({
                "title": chunk.title,
                "url": chunk.url,
                "content": chunk.text,
                "status": 200,
                "source_type": "wikipedia",
                "page_id": chunk.page_id,
                "language": chunk.language,
                "score": chunk.score,
            })

        return documents

    def decide(self, text):
        if not isinstance(text, str):
            raise TypeError(
                "text must be a string"
            )

        if self.should_use_github(text):
            return {
                "action": "github",
                "query": text,
            }

        if self.should_use_official_site(text):
            return {
                "action": "official_site",
                "query": text,
                "site": self._official_site_name(
                    text
                ),
            }

        if self.should_use_knowledge_source(text):
            return {
                "action": "knowledge_source",
                "query": text,
                "category": self.knowledge_source_category(
                    text
                ),
            }

        if self.should_use_wikipedia(text):
            return {
                "action": "wikipedia",
                "query": self.wikipedia_query(text),
            }

        if self.should_search(text):
            return {
                "action": "web_search",
                "query": text,
            }

        return {
            "action": "none",
            "query": text,
        }

    def run_context(
        self,
        text,
        max_results=5,
    ):
        """
        Route a user request and return a unified
        WebContextBuilder payload.

        This method does not modify the existing
        run() API.
        """

        decision = self.decide(text)

        action = decision["action"]

        builder = WebContextBuilder(
            router=self.router,
            max_results=max_results,
        )

        if action == "github":
            return builder.build_from_tool(
                self.github_query(decision["query"]),
                tool="github",
            )

        if action == "official_site":
            site = decision.get("site")

            if site is None:
                return {
                    "query": text,
                    "results": [],
                    "documents": [],
                    "tool": "official_site",
                }

            return builder.build_from_tool(
                decision["query"],
                tool="official_site",
                name=site,
                page="url",
            )

        if action == "knowledge_source":
            return builder.build_from_tool(
                decision["query"],
                tool="knowledge_source",
                category=decision.get(
                    "category"
                ),
            )

        if action == "wikipedia":
            documents = self.wikipedia_retrieve(
                decision["query"],
                language="ar",
                top_k=max_results,
            )

            return {
                "query": decision["query"],
                "results": [],
                "documents": documents,
                "tool": "wikipedia",
            }

        if action == "web_search":
            return builder.build(
                decision["query"]
            )

        return {
            "query": text,
            "results": [],
            "documents": [],
            "tool": "none",
        }

    def run(
        self,
        text,
        max_results=5,
    ):
        decision = self.decide(text)

        action = decision["action"]

        if action == "github":
            results = self.github_search(
                decision["query"],
                max_results=max_results,
            )

            return {
                "action": "github",
                "results": results,
            }

        if action == "official_site":
            site = decision.get("site")

            if site is None:
                return {
                    "action": "official_site",
                    "results": [],
                }

            result = self.official_site(
                site
            )

            return {
                "action": "official_site",
                "results": [result],
            }

        if action == "wikipedia":
            documents = self.wikipedia_retrieve(
                decision["query"],
                language="ar",
                top_k=max_results,
            )

            return {
                "action": "wikipedia",
                "results": documents,
            }

        if action == "web_search":
            results = self.search(
                decision["query"],
                max_results=max_results,
            )

            return {
                "action": "web_search",
                "results": results,
            }

        return {
            "action": "none",
            "results": [],
        }
