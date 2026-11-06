from neura.brain import NeuraOrchestrator, NeuraRouter
from neura.neura_core_engine import NeuraCoreEngine


def main():
    router = NeuraRouter()

    assert router.route("اكتب كود بايثون").name == "coding"
    assert router.route("لماذا يحدث هذا؟").name == "reasoning"
    assert router.route("لخّص هذا الملف").name == "documents"
    assert router.route("مرحبا يا نيورا").name == "chat"

    calls = []

    def generator(*, text, context, route):
        calls.append((text, route))
        return f"OK:{route}:{text}"

    engine = NeuraCoreEngine(generator=generator)

    first = engine.handle("مرحبا", conversation_id="test")
    second = engine.handle("اكتب كود بايثون", conversation_id="test")

    assert first["text"] == "OK:chat:مرحبا"
    assert second["route"] == "coding"
    assert len(calls) == 2

    history = engine.orchestrator.get_context("test").messages
    assert len(history) == 4
    assert history[0]["role"] == "user"
    assert history[1]["role"] == "assistant"

    print("BRAIN INTEGRATION: PASS")


if __name__ == "__main__":
    main()
