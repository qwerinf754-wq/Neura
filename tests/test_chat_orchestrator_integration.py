import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(
        0,
        str(PROJECT_ROOT),
    )

from server.services.chat import ChatService


class FakeStore:
    def __init__(self):
        self.messages = {}
        self.counter = 0

    def create_conversation(self, user_id=None):
        self.counter += 1
        conversation_id = (
            f"test-conversation-{self.counter}"
        )
        self.messages[conversation_id] = []
        return conversation_id

    def get_messages(self, conversation_id):
        return list(
            self.messages.get(
                conversation_id,
                [],
            )
        )

    def add_message(
        self,
        conversation_id,
        role,
        content,
    ):
        self.messages.setdefault(
            conversation_id,
            [],
        ).append(
            {
                "role": role,
                "content": content,
            }
        )

    def message_count(self, conversation_id):
        return len(
            self.messages.get(
                conversation_id,
                [],
            )
        )

    def clear_messages(self, conversation_id):
        self.messages[conversation_id] = []


class FakeToolAgent:
    def __init__(self, action):
        self.action = action

    def decide(self, text):
        return {
            "action": self.action,
            "query": text,
        }


class FakeOrchestrator:
    def __init__(self, action):
        self.tool_agent = FakeToolAgent(action)
        self.calls = []

    def run(self, text, max_results=5):
        self.calls.append(
            {
                "text": text,
                "max_results": max_results,
            }
        )

        return {
            "response": "ORCHESTRATOR_RESPONSE",
            "documents": [
                {
                    "title": "test-source",
                    "url": "https://example.com/",
                }
            ],
        }


class FakeSFTResult:
    text = "SFT_RESPONSE"
    finish_reason = "test"
    generated_tokens = [1, 2, 3]


def test_normal_chat_uses_sft():
    store = FakeStore()

    service = ChatService(
        user_id="test-user",
        conversation_store=store,
    )

    fake_orchestrator = FakeOrchestrator(
        "none"
    )

    service.orchestrator = fake_orchestrator

    calls = []

    def fake_generate(prompt, **kwargs):
        calls.append(
            {
                "prompt": prompt,
                "kwargs": kwargs,
            }
        )
        return FakeSFTResult()

    service.sft_runtime.generate = fake_generate

    result = service.handle_chat(
        {
            "message": "مرحبا Neura"
        }
    )

    assert result["status"] == "ok"
    assert result["response"] == "SFT_RESPONSE"
    assert len(calls) == 1
    assert fake_orchestrator.calls == []

    messages = store.get_messages(
        result["conversation_id"]
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_github_chat_uses_orchestrator():
    store = FakeStore()

    service = ChatService(
        user_id="test-user",
        conversation_store=store,
    )

    fake_orchestrator = FakeOrchestrator(
        "github"
    )

    service.orchestrator = fake_orchestrator

    def forbidden_generate(prompt, **kwargs):
        raise AssertionError(
            "Normal SFT generation must not run "
            "for a GitHub tool request"
        )

    service.sft_runtime.generate = (
        forbidden_generate
    )

    result = service.handle_chat(
        {
            "message": (
                "ابحث في GitHub عن pytorch"
            )
        }
    )

    assert result["status"] == "ok"
    assert result["response"] == \
        "ORCHESTRATOR_RESPONSE"

    assert len(fake_orchestrator.calls) == 1

    messages = store.get_messages(
        result["conversation_id"]
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def test_official_site_chat_uses_orchestrator():
    store = FakeStore()

    service = ChatService(
        user_id="test-user",
        conversation_store=store,
    )

    fake_orchestrator = FakeOrchestrator(
        "official_site"
    )

    service.orchestrator = fake_orchestrator

    def forbidden_generate(prompt, **kwargs):
        raise AssertionError(
            "Normal SFT generation must not run "
            "for an official-site request"
        )

    service.sft_runtime.generate = (
        forbidden_generate
    )

    result = service.handle_chat(
        {
            "message": (
                "ما هي توثيقات Python الرسمية؟"
            )
        }
    )

    assert result["status"] == "ok"
    assert result["response"] == \
        "ORCHESTRATOR_RESPONSE"

    assert len(fake_orchestrator.calls) == 1

    messages = store.get_messages(
        result["conversation_id"]
    )

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"


def main():
    print(
        "===== CHAT ORCHESTRATOR REGRESSION ====="
    )

    tests = [
        (
            "Normal chat → SFT",
            test_normal_chat_uses_sft,
        ),
        (
            "GitHub → Orchestrator",
            test_github_chat_uses_orchestrator,
        ),
        (
            "Official site → Orchestrator",
            test_official_site_chat_uses_orchestrator,
        ),
    ]

    for name, test in tests:
        test()
        print(f"{name}: PASS")

    print(
        "===== CHAT ORCHESTRATOR REGRESSION: PASS ====="
    )


if __name__ == "__main__":
    main()
