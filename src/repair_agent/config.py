from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

CvBackend = Literal["heuristic", "template_diff", "yolo"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # DeepSeek API (OpenAI-compatible). Empty is allowed so tests/eval can import.
    DEEPSEEK_API_KEY: str = ""
    DEEPSEEK_BASE_URL: str = "https://api.deepseek.com/v1"
    DEEPSEEK_LLM_MODEL: str = "deepseek-chat"

    # Local embeddings
    LOCAL_EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"

    # Database (optional — only needed for pgvector + PostgreSQL checkpointing)
    DATABASE_URL: str = ""
    SYNC_DATABASE_URL: str = ""

    # File paths
    CORPUS_DIR: str = "./docs/corpus"
    DATA_DIR: str = "./data"
    YOLO_WEIGHTS: str = ""
    TESSERACT_CMD: str = ""

    # Agent tuning
    CV_BACKEND: CvBackend = "heuristic"
    CONFIDENCE_THRESHOLD: float = 0.75
    MAX_CORRECTION_RETRIES: int = 3
    LOG_LEVEL: str = "INFO"


settings = Settings()
