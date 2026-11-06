from neura.agents.web_context import (
    WebContextBuilder
)


print("=" * 60)
print("Neura Web Context Builder Test")
print("=" * 60)


builder = WebContextBuilder(
    max_results=2,
    max_chars_per_page=2000
)


query = "artificial intelligence"


print()
print("Query:", query)
print()
print("Searching and fetching...")


data = builder.build(
    query
)


print()
print(
    "Search results:",
    len(data["results"])
)

print(
    "Fetched documents:",
    len(data["documents"])
)


for index, document in enumerate(
    data["documents"],
    start=1
):

    print()
    print(
        f"SOURCE {index}:",
        document["title"]
    )

    print(
        "URL:",
        document["url"]
    )

    content = document.get(
        "content",
        ""
    )

    print(
        "Content characters:",
        len(content)
    )

    print(
        "Preview:"
    )

    print(
        content[:300]
    )


print()
print("=" * 60)


if (
    len(data["results"]) > 0
    and len(data["documents"]) > 0
):

    print(
        "RESULT: WEB CONTEXT WORKING"
    )

else:

    print(
        "RESULT: WEB CONTEXT FAILED"
    )

print("=" * 60)
