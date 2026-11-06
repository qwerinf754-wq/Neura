from neura.agents import WebResearchEngine


class MockModel:

    def __call__(self, prompt):

        print()
        print("MODEL RECEIVED RESEARCH PROMPT")
        print("-" * 60)

        print(
            prompt[:6000]
        )

        print("-" * 60)

        return (
            "Artificial intelligence is a field of "
            "computer science focused on systems that "
            "perform tasks associated with human "
            "intelligence [1]."
        )


class ResearchCitationOrchestrator:

    def __init__(self, model_callback):

        self.model_callback = model_callback

        self.research = (
            WebResearchEngine(
                max_results=5,
                max_chars_per_page=3000
            )
        )

    def run(self, question):

        data = self.research.research(
            question
        )

        context = self.research.format_context(
            data
        )

        sources = self.research.format_citations(
            data
        )

        prompt = (
            "You are Neura, an AI assistant.\n\n"
            "Answer the user's question using "
            "the research context when relevant.\n\n"
            "CITATION RULES:\n"
            "When using information from the web "
            "context, cite the relevant source "
            "using [N].\n"
            "Only cite sources that support the claim.\n"
            "Do not invent citations.\n\n"
            "USER QUESTION:\n"
            f"{question}\n\n"
            "WEB RESEARCH:\n"
            f"{context}\n\n"
            f"{sources}"
        )

        response = self.model_callback(
            prompt
        )

        return {
            "question": question,
            "results": data["results"],
            "documents": data["documents"],
            "web_context": context,
            "sources": sources,
            "response": response,
        }


print("=" * 60)
print("Neura Research + Citation Integration Test")
print("=" * 60)


question = "What is artificial intelligence?"


orchestrator = ResearchCitationOrchestrator(
    MockModel()
)


result = orchestrator.run(
    question
)


print()
print("=" * 60)
print("RESULT")
print("=" * 60)

print(
    "Question:",
    result["question"]
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
    "Context characters:",
    len(result["web_context"])
)

print(
    "Response:",
    result["response"]
)


print()
print("=" * 60)


if (
    len(result["results"]) > 0
    and len(result["documents"]) > 0
    and len(result["web_context"]) > 0
    and "[1]" in result["response"]
):

    print(
        "RESULT: RESEARCH + CITATION PASSED"
    )

else:

    print(
        "RESULT: RESEARCH + CITATION FAILED"
    )

print("=" * 60)
