from neura.agents import CitationFormatter


print("=" * 60)
print("Neura Citation Formatter Test")
print("=" * 60)


documents = [
    {
        "title": "Artificial Intelligence - Wikipedia",
        "url": "https://en.wikipedia.org/wiki/Artificial_intelligence",
        "content": "Artificial intelligence is a field of computer science."
    },
    {
        "title": "Artificial Intelligence - Britannica",
        "url": "https://www.britannica.com/technology/artificial-intelligence",
        "content": "AI includes reasoning, problem solving and perception."
    }
]


formatter = CitationFormatter()


print()
print("Source block:")
print(
    formatter.build_source_block(
        documents
    )
)

print()
print("Instruction:")
print(
    formatter.build_instruction()
)


response = (
    "Artificial intelligence is a field "
    "of computer science [1]. "
    "It includes reasoning and problem solving [2]."
)


validation = formatter.validate_citations(
    response,
    documents
)


print()
print("Response:")
print(response)

print()
print("Citation validation:")
print(validation)

print()
print("=" * 60)


if (
    validation["valid"]
    and validation["citations"] == [1, 2]
):

    print(
        "RESULT: CITATION SYSTEM PASSED"
    )

else:

    print(
        "RESULT: CITATION SYSTEM FAILED"
    )

print("=" * 60)
