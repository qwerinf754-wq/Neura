"""
Neura Web Fetch Tool
"""

import gzip
import zlib

from urllib.parse import urlparse
from urllib.request import Request, urlopen


class WebFetch:

    def __init__(self, timeout=10):
        self.timeout = timeout

    def _decode_body(self, data, response):
        encoding = (
            response.headers.get("Content-Encoding", "")
            .lower()
            .strip()
        )

        if encoding == "gzip":
            data = gzip.decompress(data)

        elif encoding == "deflate":
            try:
                data = zlib.decompress(data)
            except zlib.error:
                data = zlib.decompress(
                    data,
                    -zlib.MAX_WBITS,
                )

        content_type = (
            response.headers.get("Content-Type", "")
        )

        charset = "utf-8"

        for part in content_type.split(";"):
            part = part.strip()

            if part.lower().startswith("charset="):
                charset = part.split(
                    "=",
                    1,
                )[1].strip().strip('"')

        return data.decode(
            charset or "utf-8",
            errors="replace",
        )

    def fetch(self, url):
        parsed = urlparse(url)

        if parsed.scheme not in (
            "http",
            "https",
        ):
            raise ValueError(
                "Only HTTP and HTTPS URLs are supported."
            )

        request = Request(
            url,
            headers={
                "User-Agent": "Neura/0.1 Web Agent",
                "Accept": (
                    "text/html,"
                    "application/xhtml+xml,"
                    "application/json,"
                    "text/plain,"
                    "*/*"
                ),
                "Accept-Encoding": (
                    "gzip, deflate"
                ),
            },
        )

        with urlopen(
            request,
            timeout=self.timeout,
        ) as response:

            data = response.read()

            body = self._decode_body(
                data,
                response,
            )

            return {
                "url": url,
                "status": response.status,
                "content_type": response.headers.get(
                    "Content-Type"
                ),
                "content_encoding": response.headers.get(
                    "Content-Encoding"
                ),
                "body": body,
            }
