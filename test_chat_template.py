from neura.chat import ChatTemplate


template = ChatTemplate()

messages = [
    {
        "role": "system",
        "content": "You are Neura.",
    },
    {
        "role": "user",
        "content": "Hello",
    },
    {
        "role": "assistant",
        "content": "Hello. I'm Neura.",
    },
    {
        "role": "user",
        "content": "What are you?",
    },
    {
        "role": "assistant",
        "content": "I am Neura.",
    },
]


print("=" * 60)
print("NEURA CHAT TEMPLATE TEST")
print("=" * 60)

rendered = template.render(messages)

print()
print("RENDERED:")
print(repr(rendered))

spans = template.assistant_spans(messages)

print()
print("ASSISTANT SPANS:")
print(spans)

for start, end in spans:
    print(
        "SPAN:",
        repr(rendered[start:end]),
    )


assert len(spans) == 2

assert rendered[
    spans[0][0]:
    spans[0][1]
] == (
    "<|assistant|>\n"
    "Hello. I'm Neura.\n"
    "<|end|>"
)

assert rendered[
    spans[1][0]:
    spans[1][1]
] == (
    "<|assistant|>\n"
    "I am Neura.\n"
    "<|end|>"
)


print()
print("=" * 60)
print("RESULT: CHAT TEMPLATE PASS")
print("=" * 60)
