"""
Neura Wikipedia Tool

Provides deterministic access to Wikipedia through the
MediaWiki API.

Supports:
- article search
- article summary
- article URL
- Arabic and English Wikipedia
"""

import json
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class WikipediaError(RuntimeError):
    """Base error for Wikipedia operations."""


class WikipediaTool:

    API_BASES = {
        "ar": "https://ar.wikipedia.org/w/api.php",
        "en": "https://en.wikipedia.org/w/api.php",
    }

    USER_AGENT = (
        "Neura/0.1 "
        "(Wikipedia knowledge tool; "
        "+https://github.com/neura-ai/neura)"
    )

    def __init__(self, timeout=15):
        self.timeout = timeout

    def _api_request(self, language, params):
        language = (language or "ar").strip().lower()

        if language not in self.API_BASES:
            raise ValueError(
                f"Unsupported Wikipedia language: {language}"
            )

        query = dict(params)
        query["format"] = "json"
        query["formatversion"] = "2"

        url = (
            self.API_BASES[language]
            + "?"
            + urlencode(query)
        )

        request = Request(
            url,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "application/json",
            },
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout,
            ) as response:

                raw = response.read()

                return json.loads(
                    raw.decode(
                        "utf-8",
                        errors="replace",
                    )
                )

        except HTTPError as exc:
            raise WikipediaError(
                f"Wikipedia HTTP {exc.code}: {exc.reason}"
            ) from exc

        except URLError as exc:
            raise WikipediaError(
                f"Wikipedia network error: {exc.reason}"
            ) from exc

        except json.JSONDecodeError as exc:
            raise WikipediaError(
                "Wikipedia returned invalid JSON"
            ) from exc

    def search(
        self,
        query,
        language="ar",
        max_results=5,
    ):
        if not isinstance(query, str):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            return []

        max_results = max(
            1,
            min(int(max_results), 20),
        )

        data = self._api_request(
            language,
            {
                "action": "query",
                "list": "search",
                "srsearch": query,
                "srlimit": max_results,
            },
        )

        results = []

        for item in data.get(
            "query",
            {},
        ).get(
            "search",
            [],
        ):

            title = item.get(
                "title",
                "",
            ).strip()

            if not title:
                continue

            results.append(
                {
                    "title": title,
                    "page_id": item.get(
                        "pageid"
                    ),
                    "snippet": item.get(
                        "snippet",
                        "",
                    ),
                    "url": self.article_url(
                        title,
                        language,
                    ),
                    "language": language,
                }
            )

        return results

    def article(
        self,
        title,
        language="ar",
    ):
        if not isinstance(title, str):
            raise TypeError(
                "title must be a string"
            )

        title = title.strip()

        if not title:
            raise ValueError(
                "title must not be empty"
            )

        data = self._api_request(
            language,
            {
                "action": "query",
                "prop": (
                    "extracts|info"
                ),
                "explaintext": "1",
                "exsectionformat": "plain",
                "inprop": "url",
                "titles": title,
            },
        )

        pages = data.get(
            "query",
            {},
        ).get(
            "pages",
            [],
        )

        if not pages:
            return None

        page = pages[0]

        if page.get("missing"):
            return None

        actual_title = page.get(
            "title",
            title,
        )

        return {
            "title": actual_title,
            "page_id": page.get(
                "pageid"
            ),
            "extract": page.get(
                "extract",
                "",
            ),
            "url": page.get(
                "fullurl"
            ) or self.article_url(
                actual_title,
                language,
            ),
            "language": language,
        }

    def article_url(
        self,
        title,
        language="ar",
    ):
        language = (
            language or "ar"
        ).strip().lower()

        if language not in self.API_BASES:
            raise ValueError(
                f"Unsupported Wikipedia language: {language}"
            )

        return (
            f"https://{language}.wikipedia.org/wiki/"
            + quote(
                title.replace(
                    " ",
                    "_",
                ),
                safe="()",
            )
        )

    def get(
        self,
        title,
        language="ar",
    ):
        return self.article(
            title,
            language,
        )

    def search_and_fetch(
        self,
        query,
        language="ar",
        max_results=5,
    ):
        results = self.search(
            query,
            language,
            max_results,
        )

        articles = []

        for result in results:
            article = self.article(
                result["title"],
                language,
            )

            if article is not None:
                articles.append(article)

        return articles
