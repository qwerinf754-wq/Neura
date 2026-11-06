"""
Neura Web Search Tool
"""

from urllib.parse import quote_plus, unquote, urlparse, parse_qs
from urllib.request import Request, urlopen
from html.parser import HTMLParser


class SearchResultParser(HTMLParser):

    def __init__(self):
        super().__init__()

        self.results = []

        self.current = None
        self.in_title = False

    def handle_starttag(self, tag, attrs):

        attrs = dict(attrs)

        if tag != "a":
            return

        href = attrs.get("href", "")

        if not href:
            return

        classes = attrs.get("class", "").split()

        # DuckDuckGo result links
        if (
            "result__a" in classes
            or "result-link" in classes
        ):

            self.current = {
                "title": "",
                "url": href
            }

            self.in_title = True

    def handle_data(self, data):

        if (
            self.current is not None
            and self.in_title
        ):
            self.current["title"] += data

    def handle_endtag(self, tag):

        if (
            tag == "a"
            and self.current is not None
        ):

            title = (
                self.current["title"]
                .strip()
            )

            url = self.current["url"]

            if title and url:

                self.results.append(
                    {
                        "title": title,
                        "url": url
                    }
                )

            self.current = None
            self.in_title = False


class WebSearch:

    def __init__(
        self,
        timeout=15
    ):

        self.timeout = timeout

    def _normalize_url(self, url):

        if not url:
            return url

        # DuckDuckGo redirect URL
        if "uddg=" in url:

            parsed = urlparse(url)

            params = parse_qs(
                parsed.query
            )

            if "uddg" in params:

                return unquote(
                    params["uddg"][0]
                )

        return url

    def search(
        self,
        query,
        max_results=5
    ):

        if not isinstance(
            query,
            str
        ):
            raise TypeError(
                "query must be a string"
            )

        query = query.strip()

        if not query:
            return []

        url = (
            "https://html.duckduckgo.com/html/?q="
            + quote_plus(query)
        )

        request = Request(
            url,
            headers={
                "User-Agent":
                    "Mozilla/5.0 "
                    "(Linux; Android 13) "
                    "AppleWebKit/537.36 "
                    "(KHTML, like Gecko) "
                    "Chrome/120 Safari/537.36",
                "Accept":
                    "text/html,application/xhtml+xml",
                "Accept-Language":
                    "en-US,en;q=0.9"
            }
        )

        # DuckDuckGo may occasionally return a temporary
        # HTTP 202 response. Retry transient responses before
        # treating the search request as failed.
        max_attempts = 3
        transient_statuses = {
            202,
            429,
            500,
            502,
            503,
            504,
        }

        html = None
        status = None
        last_error = None

        for attempt in range(1, max_attempts + 1):
            try:
                with urlopen(
                    request,
                    timeout=self.timeout
                ) as response:

                    status = response.status

                    html = response.read().decode(
                        "utf-8",
                        errors="replace"
                    )

                if status == 200:
                    break

                if status not in transient_statuses:
                    raise RuntimeError(
                        f"Search request failed: "
                        f"HTTP {status}"
                    )

            except Exception as exc:
                last_error = exc

                if attempt >= max_attempts:
                    raise RuntimeError(
                        f"Search request failed "
                        f"after {max_attempts} attempts: "
                        f"{exc}"
                    ) from exc

                continue

        if status != 200 or html is None:
            if last_error is not None:
                raise RuntimeError(
                    f"Search request failed "
                    f"after {max_attempts} attempts: "
                    f"{last_error}"
                ) from last_error

            raise RuntimeError(
                f"Search request failed: "
                f"HTTP {status}"
            )

        parser = SearchResultParser()

        parser.feed(html)

        results = []

        seen = set()

        for result in parser.results:

            result["url"] = (
                self._normalize_url(
                    result["url"]
                )
            )

            url = result["url"]

            if not url:
                continue

            if url in seen:
                continue

            seen.add(url)

            results.append(result)

            if len(results) >= max_results:
                break

        return results
