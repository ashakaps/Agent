from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Только секрет — без default, контейнер падает если не задан.
    llm_api_key: str

    # LLM provider. Дефолты для OpenRouter; override через .env при смене провайдера.
    llm_base_url: str = "https://openrouter.ai/api/v1"
    llm_model: str = "meta-llama/llama-3.3-70b-instruct"
    llm_temperature: float = 0.0
    # Лимит токенов ответа. Без него OpenRouter резервирует под ответ до 32768
    # токенов и отвечает 402, если на ключе меньше кредитов.
    llm_max_tokens: int = 1024

    # Vector store
    qdrant_url: str = "http://qdrant:6333"
    collection_name: str = "sklearn_docs"
    top_k: int = 4

    # Embeddings (e5 — мультиязычный, нужно для русского)
    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    normalize_embeddings: bool = True

    # agent
    max_iterations: int = 5
    agent_temperature: float = 0.0
    enable_web_search: bool = True
    agent_max_output_chars: int = 2000

    # agent tools
    web_search_max_results: int = 3
    web_search_retries: int = 3
    python_repl_timeout_s: int = 10


settings = Settings()
