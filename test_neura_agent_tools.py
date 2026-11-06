from neura.agents import ToolAgent


print("=" * 60)
print("Neura Agent Tool Calling Test")
print("=" * 60)


agent = ToolAgent()


queries = [
    "What is artificial intelligence?",
    "latest artificial intelligence news",
]


for query in queries:

    print()
    print("Query:", query)

    decision = agent.decide(
        query
    )

    print(
        "Decision:",
        decision["action"]
    )

    if decision["action"] == "web_search":

        result = agent.run(
            query,
            max_results=3
        )

        print(
            "Search results:",
            len(result["results"])
        )

        for index, item in enumerate(
            result["results"],
            start=1
        ):

            print(
                f"[{index}]",
                item["title"]
            )

            print(
                item["url"]
            )


print()
print("=" * 60)

print(
    "RESULT: AGENT TOOL CALLING WORKING"
)

print("=" * 60)
