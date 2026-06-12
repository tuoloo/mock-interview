from mockinterview.llm import LLMClient


def test_protocol_is_satisfied_by_fake(fake_llm):
    assert isinstance(fake_llm, LLMClient)


def test_anthropic_client_importable():
    from mockinterview.llm import AnthropicClient
    assert AnthropicClient is not None
