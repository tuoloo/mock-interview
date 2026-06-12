from mockinterview.config import Settings


def test_defaults():
    s = Settings(anthropic_api_key="sk-test")
    assert s.model_id == "claude-opus-4-8"
    assert s.max_followups_per_question == 2
    assert s.max_questions_per_stage == 4


def test_env_override(monkeypatch):
    monkeypatch.setenv("MOCKINTERVIEW_MODEL_ID", "claude-sonnet-4-6")
    monkeypatch.setenv("MOCKINTERVIEW_ANTHROPIC_API_KEY", "sk-test")
    s = Settings()
    assert s.model_id == "claude-sonnet-4-6"
