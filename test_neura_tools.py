from neura.tools import ToolRouter


print("=" * 60)
print("Neura Tool Router Test")
print("=" * 60)


router = ToolRouter()


print("Available tools:")
print(router.list_tools())


print()
print("Testing web_search...")


results = router.call(
    "web_search",
    query="latest artificial intelligence",
    max_results=3
)


print(
    "Results:",
    len(results)
)


for index, result in enumerate(
    results,
    start=1
):

    print()
    print(
        f"[{index}]",
        result["title"]
    )

    print(
        result["url"]
    )


print()
print("=" * 60)


if (
    router.has_tool("web_search")
    and router.has_tool("web_fetch")
    and results
):

    print(
        "RESULT: TOOL ROUTER WORKING"
    )

else:

    print(
        "RESULT: TOOL ROUTER FAILED"
    )


print("=" * 60)
