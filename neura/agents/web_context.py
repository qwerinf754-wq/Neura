"""
Neura Web Context Builder
"""

import re
from html import unescape
from html.parser import HTMLParser

from neura.tools import ToolRouter
from neura.agents.web_research import WebResearchEngine


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
        super().__init__(
            convert_charrefs=True
        )

        self.parts = []
        self.skip_depth = 0

    def handle_starttag(self, tag, attrs):

        tag = tag.lower()

        if tag in self.SKIP_TAGS:
            self.skip_depth += 1
            return

        if (
            self.skip_depth == 0
            and tag in self.BLOCK_TAGS
        ):
            self.parts.append("\n")

    def handle_endtag(self, tag):

        tag = tag.lower()

        if tag in self.SKIP_TAGS:

            if self.skip_depth > 0:
                self.skip_depth -= 1

            return

        if (
            self.skip_depth == 0
            and tag in self.BLOCK_TAGS
        ):
            self.parts.append("\n")

    def handle_data(self, data):

        if self.skip_depth != 0:
            return

        text = data.strip()

        if text:
            self.parts.append(text)

    def get_text(self):

        text = " ".join(
            self.parts
        )

        text = unescape(text)

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()


class WebContextBuilder:

    def __init__(
        self,
        router=None,
        max_results=3,
        max_chars_per_page=4000
    ):
        self.router = (
            router
            if router is not None
            else ToolRouter()
        )

        self.max_results = max_results
        self.max_chars_per_page = max_chars_per_page

        self.research_engine = WebResearchEngine(
            router=self.router,
            max_results=self.max_results,
            max_chars_per_page=self.max_chars_per_page,
        )

    def _extract_text(self, page):

        if isinstance(page, str):

            raw = page

        elif isinstance(page, dict):

            raw = page.get(
                "text"
            )

            if not raw:
                raw = page.get(
                    "content"
                )

            if not raw:
                raw = page.get(
                    "body"
                )

            if not raw:
                raw = ""

        else:

            raw = getattr(
                page,
                "text",
                ""
            )

        raw = str(raw)

        if not raw:
            return ""

        # --------------------------------------------------
        # HTML document
        # --------------------------------------------------

        if "<html" in raw.lower():

            parser = _HTMLTextExtractor()

            try:

                parser.feed(raw)
                parser.close()

                text = parser.get_text()

            except Exception:

                text = raw

        else:

            text = raw

        # --------------------------------------------------
        # Cleanup
        # --------------------------------------------------

        text = unescape(
            text
        )

        text = re.sub(
            r"\s+",
            " ",
            text
        )

        return text.strip()

    def build(self, query):
        data = self.research_engine.research(query)

        return {
            "query": data.get("query", query),
            "results": data.get("results", []),
            "documents": data.get("documents", []),
        }

    def build_from_tool(
        self,
        query,
        tool="github",
        **kwargs,
    ):
        """
        Build a unified context payload from a specialized tool.

        Supported tools:
        - github
        - official_site
        - knowledge_source
        """

        if tool == "knowledge_source":
            data = (
                self.research_engine
                .research_knowledge_sources(
                    query,
                    max_results=self.max_results,
                    category=kwargs.get(
                        "category"
                    ),
                )
            )
        else:
            data = self.research_engine.research_tools(
                query,
                tool=tool,
                max_results=self.max_results,
                **kwargs,
            )

        return {
            "query": data.get(
                "query",
                query,
            ),
            "results": data.get(
                "results",
                [],
            ),
            "documents": data.get(
                "documents",
                [],
            ),
            "tool": data.get(
                "tool",
                tool,
            ),
        }

    def format_context(self, data):

        blocks = []

        for index, document in enumerate(
            data["documents"],
            start=1
        ):

            content = document.get(
                "content",
                ""
            )

            blocks.append(
                f"SOURCE {index}\n"
                f"TITLE: "
                f"{document.get('title', '')}\n"
                f"URL: "
                f"{document.get('url', '')}\n"
                f"CONTENT:\n"
                f"{content}"
            )

        return "\n\n".join(
            blocks
        )

    def format_citations(self, data):
        lines = ["SOURCES:"]

        for index, document in enumerate(
            data.get("documents", []),
            start=1
        ):
            lines.append(
                f"\n[{index}] "
                f"{document.get('title', '')}\n"
                f"URL: {document.get('url', '')}"
            )

        return "\n".join(lines)
