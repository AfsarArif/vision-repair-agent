from sqlalchemy import JSON, Boolean, Column, DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class DiagnosticSession(Base):
    """One row per POST /diagnose. ``id`` is also the LangGraph ``thread_id``."""

    __tablename__ = "diagnostic_sessions"

    id = Column(String, primary_key=True)
    status = Column(String, nullable=False, default="completed")  # completed | failed
    error = Column(Text, nullable=True)
    defect_type = Column(String, nullable=True)
    defect_confidence = Column(Float, nullable=True)
    serial_number = Column(String, nullable=True)
    designator = Column(String, nullable=True)
    diagnosis = Column(Text, nullable=True)
    correction_mode = Column(String, nullable=True)
    cv_backend = Column(String, nullable=True)
    image_sha256 = Column(String(64), nullable=True, index=True)
    template_sha256 = Column(String(64), nullable=True)
    detections = Column(JSON().with_variant(JSONB(), "postgresql"), nullable=True)
    rag_documents_used = Column(Integer, default=0)
    self_correction_triggered = Column(Boolean, default=False)
    correction_attempts = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source_file = Column(String, nullable=False)
    chunk_index = Column(Integer, nullable=False)
    content = Column(Text, nullable=False)
    defect_type = Column(String, nullable=True)
    ingested_at = Column(DateTime, server_default=func.now())
