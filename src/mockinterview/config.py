from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="MOCKINTERVIEW_", env_file=".env")

    anthropic_api_key: str
    model_id: str = "claude-opus-4-8"
    max_followups_per_question: int = 2
    max_questions_per_stage: int = 4
    effort: str = "high"
