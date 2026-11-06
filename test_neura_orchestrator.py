from neura.agents import (
    NeuraOrchestrator
)


print("=" * 60)
print("Neura Agent Orchestrator Test")
print("=" * 60)


def fake_model(prompt):

    print()
    print("MODEL RECEIVED PROMPT:")
    print("-" * 40)
    print(prompt[:1000])
    print("-" * 40)

    return "MODEL_RESPONSE_OK"


agent = NeuraOrchestrator(
    model_callback=fake_model
)


print()
print("Test 1: normal question")

result = agent.run(
    "What is artificial intelligence?"
)

print(
    "Action:",
    result["action"]
)

print(
    "Response:",
    result["response"]
)


print()
print("Test 2: web question")

result = agent.run(
    "latest artificial intelligence news",
    max_results=3
)

print(
    "Action:",
    result["action"]
)

print(
    "Results:",
    len(result["results"])
)

print(
    "Response:",
    result["response"]
)


print()
print("=" * 60)

if (
    result["action"] == "web_search"
    and len(result["results"]) > 0
    and result["response"]
    == "MODEL_RESPONSE_OK"
):

    print(
        "RESULT: ORCHESTRATOR WORKING"
    )

else:

    print(
        "RESULT: ORCHESTRATOR FAILED"
    )

print("=" * 60)
