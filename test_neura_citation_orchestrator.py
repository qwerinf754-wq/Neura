from neura.agents import (
    NeuraOrchestrator
)


print("=" * 60)
print("Neura Citation-Aware Orchestrator Test")
print("=" * 60)


captured_prompt = {}


def fake_model(prompt):

    captured_prompt["value"] = prompt

    print()
    print("MODEL RECEIVED CITATION-AWARE PROMPT")
    print("-" * 60)

    print(prompt[:5000])

    print("-" * 60)

    return (
        "Artificial intelligence is a field "
        "of computer science concerned with "
        "creating systems capable of tasks "
        "associated with human intelligence [1]. "
        "These tasks include reasoning, "
        "problem solving, perception and "
        "language processing [2]."
    )


orchestrator = NeuraOrchestrator(
    model_callback=fake_model
)


query = "What is artificial intelligence?"


print()
print("Query:", query)
print()
print("Running citation-aware orchestrator...")


result = orchestrator.run(
    query,
    max_results=2
)


print()
print("=" * 60)
print("ORCHESTRATOR RESULT")
print("=" * 60)

print(
    "Action:",
    result["action"]
)

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
    "Model response:"
)

print(
    result["response"]
)

print()
print(
    "Detected citations:",
    result["citations"]
)

print(
    "Citation validation:",
    result["citation_validation"]
)


prompt = captured_prompt.get(
    "value",
    ""
)


has_citation_rules = (
    "CITATION RULES:" in prompt
)

has_source_block = (
    "SOURCES:" in prompt
)

has_web_context = (
    "WEB CONTEXT:" in prompt
)

has_question = (
    "USER QUESTION:" in prompt
)


print()
print("=" * 60)


if (
    result["action"] == "web_search"
    and len(result["results"]) > 0
    and len(result["documents"]) > 0
    and len(result["web_context"]) > 0
    and has_citation_rules
    and has_source_block
    and has_web_context
    and has_question
    and result["citations"] == [1, 2]
    and result["citation_validation"]["valid"]
):

    print(
        "RESULT: CITATION-AWARE ORCHESTRATOR PASSED"
    )

else:

    print(
        "RESULT: CITATION-AWARE ORCHESTRATOR FAILED"
    )

print("=" * 60)
