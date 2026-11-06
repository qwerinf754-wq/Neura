"""
Neura Tool Documents Adapter

Converts GitHub and official-site tool results
into the document contract used by WebResearchEngine.
"""

from urllib.parse import urlparse


class ToolDocumentAdapter:

    def __init__(
        self,
        max_chars=5000,
    ):
        self.max_chars = max_chars

    def _domain_title(self, url):
        if not url:
            return ""

        domain = urlparse(url).netloc

        if domain.startswith("www."):
            domain = domain[4:]

        return domain

    def github_search_documents(self, results):
        documents = []

        for result in results or []:
            url = result.get("url", "")

            if not url:
                continue

            full_name = result.get(
                "full_name",
                result.get("name", ""),
            )

            description = result.get(
                "description",
                "",
            )

            language = result.get(
                "language",
                "",
            )

            stars = result.get(
                "stars",
                0,
            )

            forks = result.get(
                "forks",
                0,
            )

            content_parts = []

            if description:
                content_parts.append(
                    description
                )

            if language:
                content_parts.append(
                    f"Language: {language}"
                )

            if stars is not None:
                content_parts.append(
                    f"Stars: {stars}"
                )

            if forks is not None:
                content_parts.append(
                    f"Forks: {forks}"
                )

            documents.append({
                "title": full_name,
                "url": url,
                "content": "\n".join(
                    content_parts
                )[:self.max_chars],
                "status": 200,
                "source_type": "github",
            })

        return documents

    def github_repository_document(self, result):
        if not isinstance(result, dict):
            return None

        url = result.get(
            "html_url",
            result.get("url", ""),
        )

        if not url:
            return None

        name = result.get(
            "full_name",
            result.get("name", ""),
        )

        content_parts = []

        description = result.get(
            "description"
        )

        if description:
            content_parts.append(
                description
            )

        language = result.get(
            "language"
        )

        if language:
            content_parts.append(
                f"Language: {language}"
            )

        stars = result.get(
            "stargazers_count"
        )

        if stars is not None:
            content_parts.append(
                f"Stars: {stars}"
            )

        forks = result.get(
            "forks_count"
        )

        if forks is not None:
            content_parts.append(
                f"Forks: {forks}"
            )

        default_branch = result.get(
            "default_branch"
        )

        if default_branch:
            content_parts.append(
                f"Default branch: {default_branch}"
            )

        return {
            "title": name,
            "url": url,
            "content": "\n".join(
                content_parts
            )[:self.max_chars],
            "status": 200,
            "source_type": "github",
        }

    def official_site_document(self, result):
        if not isinstance(result, dict):
            return None

        url = result.get("url", "")

        if not url:
            return None

        content = result.get(
            "body",
            result.get(
                "content",
                result.get("text", ""),
            ),
        )

        if content is None:
            content = ""

        title = result.get(
            "name"
        ) or self._domain_title(url)

        return {
            "title": title,
            "url": url,
            "content": str(content)[:self.max_chars],
            "status": result.get(
                "status"
            ),
            "source_type": "official_site",
        }

    def official_site_documents(self, results):
        documents = []

        for result in results or []:
            document = self.official_site_document(
                result
            )

            if document is not None:
                if document.get("content"):
                    documents.append(
                        document
                    )

        return documents
