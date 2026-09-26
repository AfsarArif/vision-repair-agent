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
    # Cross-encoder that reranks FAISS candidates; empty string disables reranking.
    RAG_RERANKER: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    RAG_FETCH_K: int = 20

    # Database (optional — only needed for pgvector + PostgreSQL checkpointing)
    DATABASE_URL: str = ""
    SYNC_DATABASE_URL: str = ""
    # Diagnose-session persistence: "memory" (default, lost on restart) or "postgres"
    # (LangGraph AsyncPostgresSaver + diagnostic_sessions rows; needs DATABASE_URL).
    PERSISTENCE_BACKEND: Literal["memory", "postgres"] = "memory"

    # File paths
    CORPUS_DIR: str = "./docs/corpus"
    DATA_DIR: str = "./data"
    YOLO_WEIGHTS: str = ""
    TESSERACT_CMD: str = ""

    # Agent tuning
    CV_BACKEND: CvBackend = "heuristic"
    # Frozen from `run_eval.py --stage sweep` on DeepPCB val (never test):
    # YOLO_CONF maximizes box micro-F1; boxes in [YOLO_CANDIDATE_CONF, CONFIDENCE_THRESHOLD)
    # are the uncertain band that template verification can recover or drop.
    YOLO_CONF: float = 0.55
    YOLO_CANDIDATE_CONF: float = 0.4
    # Self-correct when the least confident candidate box is below this.
    CONFIDENCE_THRESHOLD: float = 0.8
    TEMPLATE_MIN_COVERAGE: float = 0.05
    MAX_CORRECTION_RETRIES: int = 3
    LOG_LEVEL: str = "INFO"


settings = Settings()
