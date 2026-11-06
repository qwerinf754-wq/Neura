from neura.agents.orchestrator import (
    NeuraOrchestrator
)


print("=" * 60)
print("Neura Web-Augmented Generation Test")
print("=" * 60)


def model_callback(prompt):

    print()
    print("MODEL RECEIVED PROMPT")
    print("-" * 60)

    print(prompt[:2500])

    print("-" * 60)

    if "WEB CONTEXT:" in prompt:
        return "MODEL_USED_WEB_CONTEXT"

    return "MODEL_DID_NOT_USE_WEB_CONTEXT"


orchestrator = NeuraOrchestrator(
    model_callback=model_callback
)


query = "What is artificial intelligence?"

print()
print("Query:", query)
print()
print("Running orchestrator...")


result = orchestrator.run(
    query,
    max_results=2
)


print()
print("=" * 60)
print("ORCHESTRATOR RESULT")
print("=" * 60)

print("Action:", result["action"])

print(
    "Search results:",
    len(result["results"])
)

print(
    "Documents:",
    len(result["documents"])
)

print(
    "Web context characters:",
    len(result["web_context"])
)

print(
    "Model response:",
    result["response"]
)

print()
print("=" * 60)


if (
    result["action"] == "web_search"
    and len(result["results"]) > 0
    and len(result["documents"]) > 0
    and len(result["web_context"]) > 0
    and result["response"] == "MODEL_USED_WEB_CONTEXT"
):

    print(
        "RESULT: WEB-AUGMENTED GENERATION PASSED"
    )

else:

    print(
        "RESULT: WEB-AUGMENTED GENERATION FAILED"
    )

print("=" * 60)
