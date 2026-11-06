"""
Neura Web Tool

Provides basic HTTP fetching for the Neura agent layer.
"""

from dataclasses import dataclass
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


class WebError(RuntimeError):
    """Base error for web operations."""


class WebFetchError(WebError):
    """Raised when a web request cannot be completed."""


@dataclass(slots=True)
class WebResponse:
    url: str
    status: int
    content_type: str
    text: str


class WebTool:
    """
    Minimal deterministic HTTP fetcher.

    This layer is intentionally independent from the model.
    """

    USER_AGENT = "Neura/0.1 (+https://github.com/neura-ai/neura)"

    def fetch(self, url: str, timeout: float = 15.0) -> WebResponse:
        if not isinstance(url, str) or not url.strip():
            raise WebFetchError("url must be a non-empty string")

        url = url.strip()

        if not (
            url.startswith("http://")
            or url.startswith("https://")
        ):
            raise WebFetchError(
                "only http:// and https:// URLs are supported"
            )

        request = Request(
            url,
            headers={
                "User-Agent": self.USER_AGENT,
                "Accept": "text/html,text/plain,application/json,*/*",
            },
        )

        try:
            with urlopen(request, timeout=timeout) as response:
                raw = response.read()

                charset = response.headers.get_content_charset()
                encoding = charset or "utf-8"

                text = raw.decode(
                    encoding,
                    errors="replace",
                )

                return WebResponse(
                    url=response.geturl(),
                    status=response.status,
                    content_type=(
                        response.headers.get_content_type()
                    ),
                    text=text,
                )

        except HTTPError as exc:
            raise WebFetchError(
                f"HTTP {exc.code}: {exc.reason}"
            ) from exc

        except URLError as exc:
            raise WebFetchError(
                f"network error: {exc.reason}"
            ) from exc


WEB_TOOL = WebTool()
