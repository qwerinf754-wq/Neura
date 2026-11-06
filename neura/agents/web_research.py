"""
Neura Web Research Engine
"""

import re
from html import unescape
from html.parser import HTMLParser
from urllib.parse import urlparse

from neura.tools import ToolRouter
from neura.agents.tool_documents import ToolDocumentAdapter
from neura.knowledge.sources import (
    KNOWLEDGE_SOURCE_RESOLVER,
)


class _HTMLTextExtractor(HTMLParser):
    SKIP_TAGS = {
        "script",
        "style",
        "noscript",
        "svg",
        "canvas",
        "template",
        "head",
    }

    BLOCK_TAGS = {
        "p",
        "div",
        "article",
        "section",
        "main",
        "header",
        "footer",
        "li",
        "ul",
        "ol",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "br",
        "tr",
    }

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):
        tag = tag.lower()

        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return

        if self.skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_endtag(self, tag):
        tag = tag.lower()

        if tag in self.SKIP_TAGS:
            if self.skip_depth > 0:
                self.skip_depth -= 1
            return

        if self.skip_depth == 0 and tag in self.BLOCK_TAGS:
            self.parts.append("\n")

    def handle_data(self, data):
        if self.skip_depth != 0:
            return

        if data.strip():
            self.parts.append(data)

    def get_text(self):
        text = " ".join(self.parts)
        text = unescape(text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()


class WebResearchEngine:

    def __init__(
        self,
        router=None,
        max_results=5,
        max_chars_per_page=5000,
    ):
        self.router = (
            router
            if router is not None
            else ToolRouter()
        )

        self.max_results = max_results
        self.max_chars_per_page = max_chars_per_page

    # ---------------------------------------------------------
    # URL normalization
    # ---------------------------------------------------------

    def normalize_url(self, url):
        if not url:
            return ""

        parsed = urlparse(url)

        if not parsed.scheme:
            return url

        normalized = (
            f"{parsed.scheme.lower()}://"
            f"{parsed.netloc.lower()}"
            f"{parsed.path}"
        )

        return normalized.rstrip("/")

    # ---------------------------------------------------------
    # Deduplicate search results
    # ---------------------------------------------------------

    def deduplicate(self, results):

        unique = []
        seen_urls = set()

        for result in results:

            url = self.normalize_url(
                result.get("url", "")
            )

            if not url:
                continue

            if url in seen_urls:
                continue

            seen_urls.add(url)

            item = dict(result)
            item["url"] = url

            unique.append(item)

        return unique

    # ---------------------------------------------------------
    # Simple source ranking
    # ---------------------------------------------------------

    def rank_sources(self, results):

        ranked = []

        preferred_domains = {
            "reuters.com": 10,
            "apnews.com": 10,
            "bbc.com": 9,
            "bbc.co.uk": 9,
            "nature.com": 9,
            "science.org": 9,
            "ibm.com": 8,
            "microsoft.com": 8,
            "google.com": 8,
            "openai.com": 8,
            "wikipedia.org": 5,
            "britannica.com": 6,
            "coursera.org": 4,
        }

        for result in results:

            url = result.get("url", "")

            parsed = urlparse(url)

            domain = parsed.netloc.lower()

            if domain.startswith("www."):
                domain = domain[4:]

            score = 0

            for preferred, value in preferred_domains.items():

                if (
                    domain == preferred
                    or domain.endswith("." + preferred)
                ):
                    score = value
                    break

            title = result.get(
                "title",
                ""
            ).lower()

            query_words = result.get(
                "_query_words",
                []
            )

            for word in query_words:

                if word and word in title:
                    score += 1

            item = dict(result)
            item["_score"] = score

            ranked.append(item)

        ranked.sort(
            key=lambda item: item.get(
                "_score",
                0
            ),
            reverse=True
        )

        return ranked

    # ---------------------------------------------------------
    # Fetch source
    # ---------------------------------------------------------

    def clean_text(self, page):
        if isinstance(page, dict):
            raw = page.get(
                "body",
                page.get(
                    "text",
                    page.get("content", "")
                )
            )
        elif isinstance(page, str):
            raw = page
        else:
            raw = ""

        raw = str(raw)

        if not raw:
            return ""

        if "<html" in raw.lower() or "<!doctype" in raw.lower():
            parser = _HTMLTextExtractor()
            parser.feed(raw)
            parser.close()
            return parser.get_text()

        text = unescape(raw)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def fetch_source(self, result):
        url = result.get("url")

        if not url:
            return None

        try:
            page = self.router.call(
                "web_fetch",
                url=url
            )

            content = self.clean_text(page)

            return {
                "title": result.get("title", ""),
                "url": url,
                "content": content[:self.max_chars_per_page],
                "status": (
                    page.get("status")
                    if isinstance(page, dict)
                    else None
                ),
            }

        except Exception as exc:
            return {
                "title": result.get("title", ""),
                "url": url,
                "content": "",
                "error": str(exc),
            }

    # ---------------------------------------------------------
    # Build research
    # ---------------------------------------------------------

    def research(self, query):

        raw_results = self.router.call(
            "web_search",
            query=query,
            max_results=self.max_results
        )

        query_words = [
            word.lower()
            for word in query.split()
            if len(word) > 2
        ]

        prepared = []

        for result in raw_results:

            item = dict(result)
            item["_query_words"] = query_words

            prepared.append(item)

        unique = self.deduplicate(
            prepared
        )

        ranked = self.rank_sources(
            unique
        )

        documents = []

        for result in ranked:

            document = self.fetch_source(
                result
            )

            if document is None:
                continue

            if document.get("content"):

                documents.append(
                    document
                )

        return {
            "query": query,
            "results": ranked,
            "documents": documents,
        }

    # ---------------------------------------------------------
    # Tool research
    # ---------------------------------------------------------

    def research_tools(
        self,
        query,
        tool="github",
        max_results=None,
        **kwargs,
    ):
        """
        Research through specialized tools while preserving
        the WebResearchEngine document contract.
        """

        limit = (
            self.max_results
            if max_results is None
            else max_results
        )

        adapter = ToolDocumentAdapter(
            max_chars=self.max_chars_per_page
        )

        if tool == "github":
            results = self.router.call(
                "github",
                operation="search",
                query=query,
                max_results=limit,
            )

            documents = adapter.github_search_documents(
                results
            )

        elif tool == "official_site":
            name = kwargs.get("name")

            if not name:
                raise ValueError(
                    "official_site research requires "
                    "'name'"
                )

            page = kwargs.get(
                "page",
                "url",
            )

            result = self.router.call(
                "official_site",
                operation="fetch",
                name=name,
                page=page,
            )

            documents = []

            if isinstance(result, dict):
                document = adapter.official_site_document(
                    result
                )

                if document is not None:
                    if document.get("content"):
                        documents.append(
                            document
                        )

            elif isinstance(result, list):
                documents = (
                    adapter.official_site_documents(
                        result
                    )
                )

        else:
            raise ValueError(
                f"Unknown research tool: {tool}"
            )

        return {
            "query": query,
            "results": results if tool == "github" else [],
            "documents": documents,
            "tool": tool,
        }

    # ---------------------------------------------------------
    # Curated global knowledge sources
    # ---------------------------------------------------------

    def research_knowledge_sources(
        self,
        query,
        max_results=None,
        category=None,
    ):
        """
        Fetch curated global knowledge sources relevant to a query.

        Sources are selected by KnowledgeSourceResolver and fetched
        only when requested. Results use the same document contract
        as normal web research.
        """

        if not isinstance(query, str):
            raise TypeError("query must be a string")

        limit = (
            self.max_results
            if max_results is None
            else max_results
        )

        if limit < 1:
            return {
                "query": query,
                "results": [],
                "documents": [],
                "tool": "knowledge_source",
            }

        sources = KNOWLEDGE_SOURCE_RESOLVER.resolve(
            query,
            category=category,
            limit=limit,
        )

        documents = []
        results = []

        for source in sources:
            try:
                fetched = self.router.call(
                    "web_fetch",
                    url=source.url,
                )
            except Exception as exc:
                print(
                    "[KnowledgeSource] "
                    f"failed={source.name!r} "
                    f"error={exc!r}"
                )
                continue

            if not isinstance(fetched, dict):
                continue

            body = fetched.get("body", "")
            if not body:
                continue

            document = {
                "title": source.name,
                "url": fetched.get(
                    "url",
                    source.url,
                ),
                "content": self.clean_text(
                    body
                )[:self.max_chars_per_page],
                "status": fetched.get(
                    "status",
                ),
                "source_type": "knowledge_source",
                "category": source.category,
                "priority": source.priority,
                "languages": source.languages,
            }

            if not document["content"]:
                continue

            documents.append(document)

            results.append({
                "title": source.name,
                "url": source.url,
                "category": source.category,
                "priority": source.priority,
            })

        return {
            "query": query,
            "results": results,
            "documents": documents,
            "tool": "knowledge_source",
        }

    # ---------------------------------------------------------
    # Context formatter
    # ---------------------------------------------------------

    def format_context(self, data):

        blocks = []

        for index, document in enumerate(
            data.get("documents", []),
            start=1
        ):

            blocks.append(
                f"SOURCE {index}\n"
                f"TITLE: "
                f"{document.get('title', '')}\n"
                f"URL: "
                f"{document.get('url', '')}\n"
                f"CONTENT:\n"
                f"{document.get('content', '')}"
            )

        return "\n\n".join(
            blocks
        )

    # ---------------------------------------------------------
    # Citation source block
    # ---------------------------------------------------------

    def format_citations(self, data):

        lines = [
            "SOURCES:"
        ]

        for index, document in enumerate(
            data.get("documents", []),
            start=1
        ):

            lines.append(
                f"\n[{index}] "
                f"{document.get('title', '')}\n"
                f"URL: "
                f"{document.get('url', '')}"
            )

        return "\n".join(lines)
