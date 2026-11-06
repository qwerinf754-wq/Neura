from neura.agents import WebResearchEngine


print("=" * 60)
print("Neura Web Research Engine Test")
print("=" * 60)


query = "artificial intelligence"


engine = WebResearchEngine(
    max_results=5,
    max_chars_per_page=3000
)


print()
print("Query:", query)
print()
print("Running research...")


data = engine.research(
    query
)


print()
print("Search results:")
print(len(data["results"]))


print()
print("Fetched documents:")
print(len(data["documents"]))


print()
print("=" * 60)
print("RANKED SOURCES")
print("=" * 60)


for index, result in enumerate(
    data["results"],
    start=1
):

    print()
    print(
        f"[{index}]",
        result.get("title", "")
    )

    print(
        "URL:",
        result.get("url", "")
    )

    print(
        "Score:",
        result.get("_score", 0)
    )


print()
print("=" * 60)
print("RESEARCH DOCUMENTS")
print("=" * 60)


for index, document in enumerate(
    data["documents"],
    start=1
):

    content = document.get(
        "content",
        ""
    )

    print()
    print(
        f"SOURCE {index}:",
        document.get("title", "")
    )

    print(
        "Characters:",
        len(content)
    )

    print(
        "Preview:"
    )

    print(
        content[:250]
    )


print()
print("=" * 60)
print("CITATIONS")
print("=" * 60)

print(
    engine.format_citations(
        data
    )
)


print()
print("=" * 60)
print("CONTEXT CHARACTERS:")

context = engine.format_context(
    data
)

print(
    len(context)
)


print()
print("=" * 60)


if (
    len(data["results"]) > 0
    and len(data["documents"]) > 0
    and len(context) > 0
):

    print(
        "RESULT: WEB RESEARCH ENGINE PASSED"
    )

else:

    print(
        "RESULT: WEB RESEARCH ENGINE FAILED"
    )

print("=" * 60)
