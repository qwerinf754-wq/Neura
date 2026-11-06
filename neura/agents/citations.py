"""
Neura Source Attribution / Citation Formatter
"""

import re


class CitationFormatter:

    def __init__(self):
        pass

    def format_sources(self, documents):
        sources = []

        for index, document in enumerate(
            documents,
            start=1
        ):
            title = document.get(
                "title",
                ""
            )

            url = document.get(
                "url",
                ""
            )

            if not url:
                continue

            sources.append({
                "id": index,
                "title": title,
                "url": url
            })

        return sources

    def build_source_block(self, documents):
        sources = self.format_sources(
            documents
        )

        if not sources:
            return ""

        lines = [
            "SOURCES:"
        ]

        for source in sources:
            lines.append(
                f"[{source['id']}] "
                f"{source['title']}\n"
                f"URL: {source['url']}"
            )

        return "\n\n".join(lines)

    def build_instruction(self):
        return (
            "CITATION RULES:\n"
            "When using information from the web context, "
            "cite the relevant source using [N].\n"
            "Only cite sources that actually support the claim.\n"
            "Do not invent citations.\n"
            "If the web context does not support a claim, "
            "do not attach a citation to it."
        )

    def extract_citations(self, text):
        if not text:
            return []

        matches = re.findall(
            r"\[(\d+)\]",
            text
        )

        seen = set()
        citations = []

        for match in matches:
            number = int(match)

            if number not in seen:
                seen.add(number)
                citations.append(number)

        return citations

    def validate_citations(
        self,
        text,
        documents
    ):
        citations = self.extract_citations(
            text
        )

        valid_max = len(documents)

        invalid = [
            number
            for number in citations
            if number < 1
            or number > valid_max
        ]

        return {
            "citations": citations,
            "invalid": invalid,
            "valid": len(invalid) == 0
        }
