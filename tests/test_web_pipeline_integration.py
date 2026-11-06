"""
Neura Web Pipeline Integration Test

ToolAgent
    ↓
ToolRouter
    ↓
WebResearchEngine
    ↓
WebContextBuilder
    ↓
Results + Context + Citations
"""

from neura.agents.tool_agent import ToolAgent
from neura.agents.web_research import WebResearchEngine
from neura.agents.web_context import WebContextBuilder
from neura.tools.router import ToolRouter


QUERY = "What is artificial intelligence?"
MAX_RESULTS = 3
MAX_CHARS_PER_PAGE = 3000

FORBIDDEN_HTML = (
    "<!DOCTYPE",
    "<HTML",
    "<HEAD",
    "<META",
    "<SCRIPT",
    "<STYLE",
)


def assert_no_raw_html(text, label):
    upper = str(text).upper()

    for marker in FORBIDDEN_HTML:
        assert marker not in upper, (
            f"{label}: raw HTML marker remains: {marker}"
        )


def test_tool_router():
    print("\n===== 1. TOOL ROUTER =====")

    router = ToolRouter()
    tools = router.list_tools()

    print("Tools:", tools)

    assert router.has_tool("web_search")
    assert router.has_tool("web_fetch")

    assert "web_search" in tools
    assert "web_fetch" in tools

    print("ToolRouter: PASS")

    return router


def test_tool_agent(router):
    print("\n===== 2. TOOL AGENT =====")

    agent = ToolAgent(router=router)

    result = agent.search(
        QUERY,
        max_results=MAX_RESULTS,
    )

    print(
        "Result type:",
        type(result).__name__
    )

    assert isinstance(result, list)
    assert len(result) > 0
    assert len(result) <= MAX_RESULTS

    for item in result:
        assert isinstance(item, dict)
        assert item.get("title")
        assert item.get("url")

        assert item["url"].startswith(
            ("http://", "https://")
        )

    print(
        "Search results:",
        len(result)
    )

    print("ToolAgent.search(): PASS")

    return result


def test_web_research(router):
    print("\n===== 3. WEB RESEARCH ENGINE =====")

    engine = WebResearchEngine(
        router=router,
        max_results=MAX_RESULTS,
        max_chars_per_page=MAX_CHARS_PER_PAGE,
    )

    data = engine.research(QUERY)

    assert isinstance(data, dict)
    assert data.get("query") == QUERY
    assert isinstance(data.get("results"), list)
    assert isinstance(data.get("documents"), list)

    results = data["results"]
    documents = data["documents"]

    print("Search results:", len(results))
    print("Documents:", len(documents))

    assert len(results) > 0
    assert len(documents) > 0
    assert len(results) <= MAX_RESULTS

    urls = set()

    for result in results:
        assert result.get("title")
        assert result.get("url")

        url = result["url"]

        assert url.startswith(
            ("http://", "https://")
        )

        assert url not in urls
        urls.add(url)

    for index, document in enumerate(
        documents,
        1
    ):
        assert document.get("title")
        assert document.get("url")

        content = document.get(
            "content",
            ""
        )

        assert isinstance(content, str)
        assert len(content) > 0
        assert len(content) <= MAX_CHARS_PER_PAGE

        assert_no_raw_html(
            content,
            f"WebResearchEngine document {index}",
        )

    print("HTML cleaning: PASS")
    print("Deduplication: PASS")
    print("Documents: PASS")
    print("WebResearchEngine: PASS")

    return data


def test_web_context(router):
    print("\n===== 4. WEB CONTEXT BUILDER =====")

    builder = WebContextBuilder(
        router=router,
        max_results=MAX_RESULTS,
        max_chars_per_page=MAX_CHARS_PER_PAGE,
    )

    data = builder.build(QUERY)

    assert isinstance(data, dict)
    assert data.get("query") == QUERY
    assert isinstance(data.get("results"), list)
    assert isinstance(data.get("documents"), list)

    results = data["results"]
    documents = data["documents"]

    print("Search results:", len(results))
    print("Documents:", len(documents))

    assert len(results) > 0
    assert len(documents) > 0

    for index, document in enumerate(
        documents,
        1
    ):
        content = document.get(
            "content",
            ""
        )

        assert document.get("title")
        assert document.get("url")
        assert content

        assert_no_raw_html(
            content,
            f"WebContextBuilder document {index}",
        )

    print("Context documents: PASS")

    return builder, data


def test_context_formatting(
    builder,
    data,
):
    print("\n===== 5. CONTEXT FORMATTING =====")

    context = builder.format_context(data)

    assert isinstance(context, str)
    assert len(context) > 0

    assert "SOURCE 1" in context
    assert "TITLE:" in context
    assert "URL:" in context
    assert "CONTENT:" in context

    assert_no_raw_html(
        context,
        "Formatted context",
    )

    print("Context length:", len(context))
    print("Required fields: PASS")
    print("Raw HTML: PASS")

    return context


def test_citations(
    builder,
    data,
):
    print("\n===== 6. CITATIONS =====")

    citations = builder.format_citations(data)

    assert isinstance(citations, str)
    assert len(citations) > 0

    assert citations.startswith(
        "SOURCES:"
    )

    documents = data["documents"]

    for index, document in enumerate(
        documents,
        1
    ):
        title = document.get(
            "title",
            ""
        )

        url = document.get(
            "url",
            ""
        )

        assert f"[{index}]" in citations
        assert title in citations
        assert url in citations

    print("Citation length:", len(citations))
    print("Citation header: PASS")
    print("Citation URLs: PASS")
    print("Citation titles: PASS")

    return citations


def test_cross_layer_consistency(
    agent_result,
    research_data,
    context_data,
):
    print(
        "\n===== 7. CROSS-LAYER CONSISTENCY ====="
    )

    agent_results = agent_result["results"]
    research_results = research_data["results"]
    context_results = context_data["results"]

    assert len(agent_results) > 0
    assert len(research_results) > 0
    assert len(context_results) > 0

    agent_urls = {
        item.get("url")
        for item in agent_results
        if item.get("url")
    }

    research_urls = {
        item.get("url")
        for item in research_results
        if item.get("url")
    }

    context_urls = {
        item.get("url")
        for item in context_results
        if item.get("url")
    }

    assert agent_urls
    assert research_urls
    assert context_urls

    assert agent_urls & research_urls
    assert research_urls & context_urls

    print(
        "ToolAgent ↔ Research: PASS"
    )

    print(
        "Research ↔ Context: PASS"
    )

    print(
        "Cross-layer consistency: PASS"
    )


def main():
    print()
    print("=" * 60)
    print(
        "NEURA WEB PIPELINE FULL INTEGRATION TEST"
    )
    print("=" * 60)

    router = test_tool_router()

    agent_result = test_tool_agent(
        router
    )

    research_data = test_web_research(
        router
    )

    builder, context_data = test_web_context(
        router
    )

    context = test_context_formatting(
        builder,
        context_data
    )

    citations = test_citations(
        builder,
        context_data
    )

    test_cross_layer_consistency(
        agent_result,
        research_data,
        context_data
    )

    print(
        "\n===== CONTEXT PREVIEW ====="
    )

    print(
        context[:1000]
    )

    print(
        "\n===== CITATIONS PREVIEW ====="
    )

    print(
        citations[:1000]
    )

    print()
    print("=" * 60)
    print(
        "NEURA WEB PIPELINE INTEGRATION: PASS"
    )
    print("=" * 60)


if __name__ == "__main__":
    main()
