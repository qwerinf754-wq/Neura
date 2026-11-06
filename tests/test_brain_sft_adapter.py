from neura.brain import NeuraOrchestrator
from neura.brain.sft_adapter import NeuraSFTAdapter


class FakeSFTRuntime:
    def __init__(self):
        self.calls = []

    def generate(self, *, text, context=None, route="chat", **kwargs):
        self.calls.append(
            {
                "text": text,
                "context": context,
                "route": route,
            }
        )
        return f"SFT:{route}:{text}"


def main():
    runtime = FakeSFTRuntime()
    adapter = NeuraSFTAdapter(runtime)

    orchestrator = NeuraOrchestrator()
    adapter.attach(orchestrator)

    response = orchestrator.handle(
        "مرحبا يا نيورا",
        conversation_id="adapter-test",
    )

    assert response.text == "SFT:chat:مرحبا يا نيورا"
    assert response.route == "chat"

    assert len(runtime.calls) == 1
    assert runtime.calls[0]["text"] == "مرحبا يا نيورا"
    assert runtime.calls[0]["route"] == "chat"

    context = orchestrator.get_context("adapter-test")

    assert len(context.messages) == 2
    assert context.messages[0]["role"] == "user"
    assert context.messages[1]["role"] == "assistant"

    print("BRAIN SFT ADAPTER: PASS")


if __name__ == "__main__":
    main()
