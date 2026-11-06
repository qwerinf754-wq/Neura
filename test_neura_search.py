from neura.tools.web_search import WebSearch


print("=" * 60)
print("Neura Web Search Test")
print("=" * 60)

query = "artificial intelligence"

print("Query:", query)
print()
print("Searching...")


try:

    search = WebSearch()

    results = search.search(
        query,
        max_results=5
    )

    print()

    print(
        "Results found:",
        len(results)
    )

    print()

    for index, result in enumerate(
        results,
        start=1
    ):

        print(
            f"[{index}] "
            f"{result['title']}"
        )

        print(
            "URL:",
            result["url"]
        )

        print()

    print("=" * 60)

    if results:

        print(
            "RESULT: WEB SEARCH WORKING"
        )

    else:

        print(
            "RESULT: SEARCH RETURNED NO RESULTS"
        )

    print("=" * 60)


except Exception as e:

    print("=" * 60)

    print(
        "RESULT: WEB SEARCH FAILED"
    )

    print(
        "Error:",
        type(e).__name__
    )

    print(
        "Message:",
        e
    )

    print("=" * 60)
