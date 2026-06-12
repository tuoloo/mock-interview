import pytest


class FakeLLM:
    """测试用：按 FIFO 返回预设响应，记录调用。"""
    def __init__(self):
        self.structured_responses: list[dict] = []
        self.text_responses: list[str] = []
        self.calls: list[dict] = []

    def queue_structured(self, obj: dict):
        self.structured_responses.append(obj)

    def queue_text(self, text: str):
        self.text_responses.append(text)

    def structured(self, *, prompt, schema, system=None):
        self.calls.append({"kind": "structured", "prompt": prompt, "system": system})
        return self.structured_responses.pop(0)

    def stream_text(self, *, prompt, system=None):
        self.calls.append({"kind": "text", "prompt": prompt, "system": system})
        text = self.text_responses.pop(0)
        for chunk in text.split(" "):
            yield chunk + " "


@pytest.fixture
def fake_llm():
    return FakeLLM()
