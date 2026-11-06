"""
Neura Tool Router
"""

from .github import GitHubTool
from .official_sites import OfficialSites
from .web_fetch import WebFetch
from .web_search import WebSearch
from .wikipedia import WikipediaTool


class ToolRouter:

    def __init__(self):
        self.web_search = WebSearch()
        self.web_fetch = WebFetch()
        self.github = GitHubTool()
        self.official_sites = OfficialSites(
            fetcher=self.web_fetch
        )
        self.wikipedia = WikipediaTool()

        self.tools = {
            "web_search": self.web_search,
            "web_fetch": self.web_fetch,
            "github": self.github,
            "official_site": self.official_sites,
            "wikipedia": self.wikipedia,
        }

    def list_tools(self):
        return list(self.tools.keys())

    def has_tool(self, name):
        return name in self.tools

    def call(self, tool_name, **kwargs):

        if not self.has_tool(tool_name):
            raise KeyError(
                f"Unknown tool: {tool_name}"
            )

        if tool_name == "web_search":
            return self.web_search.search(
                kwargs.get("query", ""),
                kwargs.get(
                    "max_results",
                    5
                ),
            )

        if tool_name == "web_fetch":
            return self.web_fetch.fetch(
                kwargs["url"]
            )

        if tool_name == "github":
            operation = kwargs.get(
                "operation",
                "search"
            )

            if operation == "search":
                return self.github.search_repositories(
                    kwargs["query"],
                    kwargs.get(
                        "max_results",
                        5
                    ),
                )

            if operation == "repository":
                return self.github.repository(
                    kwargs["full_name"]
                )

            if operation == "readme":
                return self.github.readme(
                    kwargs["full_name"]
                )

            if operation == "file":
                return self.github.file(
                    kwargs["full_name"],
                    kwargs["path"],
                    kwargs.get("ref"),
                )

            raise ValueError(
                f"Unknown GitHub operation: "
                f"{operation}"
            )

        if tool_name == "wikipedia":
            operation = kwargs.get(
                "operation",
                "search",
            )

            if operation == "search":
                return self.wikipedia.search(
                    kwargs.get("query", ""),
                    kwargs.get("language", "ar"),
                    kwargs.get("max_results", 5),
                )

            if operation == "get":
                return self.wikipedia.get(
                    kwargs["title"],
                    kwargs.get("language", "ar"),
                )

            if operation == "search_and_fetch":
                return self.wikipedia.search_and_fetch(
                    kwargs.get("query", ""),
                    kwargs.get("language", "ar"),
                    kwargs.get("max_results", 5),
                )

            raise ValueError(
                f"Unknown Wikipedia operation: {operation}"
            )

        if tool_name == "official_site":
            operation = kwargs.get(
                "operation",
                "list"
            )

            if operation == "list":
                return self.official_sites.list_sites()

            if operation == "get":
                return self.official_sites.get(
                    kwargs["name"]
                )

            if operation == "fetch":
                return self.official_sites.fetch(
                    kwargs["name"],
                    kwargs.get("page", "url"),
                )

            raise ValueError(
                f"Unknown official_site "
                f"operation: {operation}"
            )

        raise RuntimeError(
            f"Unsupported tool: {tool_name}"
        )
