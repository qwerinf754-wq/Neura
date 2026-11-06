"""
Neura GitHub Tool

Public GitHub REST API integration.

Supports:
- repository search
- repository metadata
- README retrieval
- file retrieval
"""

import base64
import json
import os
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen


class GitHubError(RuntimeError):
    """Base GitHub tool error."""


class GitHubTool:
    API_BASE = "https://api.github.com"

    USER_AGENT = (
        "Neura/0.1 "
        "(+https://github.com/neura-ai/neura)"
    )

    API_VERSION = "2026-03-10"

    def __init__(self, timeout=15):
        self.timeout = timeout
        self.token = os.environ.get("GITHUB_TOKEN")

    def _request(self, path, params=None, accept=None):
        url = self.API_BASE + path

        if params:
            url += "?" + urlencode(params)

        headers = {
            "User-Agent": self.USER_AGENT,
            "Accept": accept or "application/vnd.github+json",
            "X-GitHub-Api-Version": self.API_VERSION,
        }

        if self.token:
            headers["Authorization"] = (
                f"Bearer {self.token}"
            )

        request = Request(
            url,
            headers=headers,
        )

        try:
            with urlopen(
                request,
                timeout=self.timeout
            ) as response:

                raw = response.read()

                charset = (
                    response.headers.get_content_charset()
                    or "utf-8"
                )

                text = raw.decode(
                    charset,
                    errors="replace"
                )

                return response.status, text

        except HTTPError as exc:
            body = ""

            try:
                body = exc.read().decode(
                    "utf-8",
                    errors="replace"
                )
            except Exception:
                pass

            raise GitHubError(
                f"GitHub HTTP {exc.code}: "
                f"{body[:500]}"
            ) from exc

        except URLError as exc:
            raise GitHubError(
                f"GitHub network error: {exc.reason}"
            ) from exc

    def search_repositories(
        self,
        query,
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
            min(int(max_results), 30)
        )

        status, text = self._request(
            "/search/repositories",
            params={
                "q": query,
                "per_page": max_results,
            },
        )

        data = json.loads(text)

        results = []

        for item in data.get("items", []):
            results.append({
                "name": item.get("name"),
                "full_name": item.get("full_name"),
                "description": item.get(
                    "description"
                ),
                "url": item.get("html_url"),
                "api_url": item.get("url"),
                "language": item.get(
                    "language"
                ),
                "stars": item.get(
                    "stargazers_count"
                ),
                "forks": item.get(
                    "forks_count"
                ),
                "open_issues": item.get(
                    "open_issues_count"
                ),
                "default_branch": item.get(
                    "default_branch"
                ),
                "archived": item.get(
                    "archived"
                ),
            })

        return results

    def repository(self, full_name):
        if not isinstance(full_name, str):
            raise TypeError(
                "full_name must be a string"
            )

        full_name = full_name.strip().strip("/")

        if full_name.endswith(".git"):
            full_name = full_name[:-4]

        parts = full_name.split("/")

        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "repository must be owner/name"
            )

        owner, repo = parts

        _, text = self._request(
            f"/repos/"
            f"{quote(owner, safe='')}/"
            f"{quote(repo, safe='')}"
        )

        return json.loads(text)

    def readme(self, full_name):
        if not isinstance(full_name, str):
            raise TypeError(
                "full_name must be a string"
            )

        full_name = full_name.strip().strip("/")

        if full_name.endswith(".git"):
            full_name = full_name[:-4]

        parts = full_name.split("/")

        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "repository must be owner/name"
            )

        owner, repo = parts

        _, text = self._request(
            f"/repos/"
            f"{quote(owner, safe='')}/"
            f"{quote(repo, safe='')}/readme"
        )

        data = json.loads(text)

        content = data.get("content", "")
        encoding = data.get("encoding")

        if encoding == "base64":
            content = base64.b64decode(
                content
            ).decode(
                "utf-8",
                errors="replace"
            )

        return {
            "repository": full_name,
            "name": data.get("name"),
            "path": data.get("path"),
            "url": data.get("html_url"),
            "content": content,
        }

    def file(
        self,
        full_name,
        path,
        ref=None,
    ):
        if not isinstance(full_name, str):
            raise TypeError(
                "full_name must be a string"
            )

        if not isinstance(path, str):
            raise TypeError(
                "path must be a string"
            )

        full_name = full_name.strip().strip("/")

        if full_name.endswith(".git"):
            full_name = full_name[:-4]

        parts = full_name.split("/")

        if len(parts) != 2 or not all(parts):
            raise ValueError(
                "repository must be owner/name"
            )

        owner, repo = parts

        api_path = (
            f"/repos/"
            f"{quote(owner, safe='')}/"
            f"{quote(repo, safe='')}/"
            f"contents/"
            f"{quote(path.strip('/'), safe='/')}"
        )

        params = {}

        if ref:
            params["ref"] = ref

        _, text = self._request(
            api_path,
            params=params or None,
        )

        data = json.loads(text)

        if isinstance(data, list):
            return {
                "repository": full_name,
                "path": path,
                "type": "directory",
                "entries": data,
            }

        content = data.get("content", "")
        encoding = data.get("encoding")

        if encoding == "base64" and content:
            content = base64.b64decode(
                content
            ).decode(
                "utf-8",
                errors="replace"
            )

        return {
            "repository": full_name,
            "path": data.get(
                "path",
                path
            ),
            "type": data.get("type"),
            "name": data.get("name"),
            "url": data.get("html_url"),
            "content": content,
            "size": data.get("size"),
            "sha": data.get("sha"),
        }

    def search(self, query, max_results=5):
        return self.search_repositories(
            query,
            max_results=max_results,
        )


GITHUB_TOOL = GitHubTool()
