"""
Neura Official Sites Tool

Curated official documentation/site registry.
"""

from .web_fetch import WebFetch


class OfficialSitesError(RuntimeError):
    """Official sites tool error."""


class OfficialSites:
    SITES = {
        "python": {
            "name": "Python",
            "url": "https://www.python.org/",
            "docs": "https://docs.python.org/3/",
        },
        "pytorch": {
            "name": "PyTorch",
            "url": "https://pytorch.org/",
            "docs": "https://docs.pytorch.org/",
        },
        "huggingface": {
            "name": "Hugging Face",
            "url": "https://huggingface.co/",
            "docs": "https://huggingface.co/docs",
        },
        "github": {
            "name": "GitHub",
            "url": "https://github.com/",
            "docs": "https://docs.github.com/",
        },
        "mdn": {
            "name": "MDN Web Docs",
            "url": "https://developer.mozilla.org/",
            "docs": "https://developer.mozilla.org/en-US/docs",
        },
        "wikipedia": {
            "name": "Wikipedia",
            "url": "https://www.wikipedia.org/",
        },
        "python_pypi": {
            "name": "Python Package Index",
            "url": "https://pypi.org/",
        },
        "numpy": {
            "name": "NumPy",
            "url": "https://numpy.org/",
            "docs": "https://numpy.org/doc/",
        },
        "scipy": {
            "name": "SciPy",
            "url": "https://scipy.org/",
            "docs": "https://docs.scipy.org/",
        },
        "rust": {
            "name": "Rust",
            "url": "https://www.rust-lang.org/",
            "docs": "https://doc.rust-lang.org/",
        },
        "javascript": {
            "name": "JavaScript",
            "url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript",
        },
        "docker": {
            "name": "Docker",
            "url": "https://www.docker.com/",
            "docs": "https://docs.docker.com/",
        },
        "linux": {
            "name": "Linux",
            "url": "https://www.linux.org/",
        },
    }

    def __init__(self, fetcher=None):
        self.fetcher = fetcher or WebFetch()

    def list_sites(self):
        return {
            key: dict(value)
            for key, value in self.SITES.items()
        }

    def get(self, name):
        if not isinstance(name, str):
            raise TypeError(
                "name must be a string"
            )

        key = name.strip().lower()

        if key not in self.SITES:
            raise KeyError(
                f"Unknown official site: {name}"
            )

        return dict(self.SITES[key])

    def fetch(self, name, page="url"):
        site = self.get(name)

        if page not in site:
            raise KeyError(
                f"Page '{page}' is not available "
                f"for site '{name}'"
            )

        url = site[page]

        result = self.fetcher.fetch(url)

        return {
            "site": site["name"],
            "key": name,
            "page": page,
            "url": result["url"],
            "status": result["status"],
            "content_type": result[
                "content_type"
            ],
            "body": result["body"],
        }


OFFICIAL_SITES = OfficialSites()
