from datetime import datetime

from sqlalchemy import Column, String, Float, Boolean, Integer, Text, DateTime, func
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class DiagnosticSession(Base):
    __tablename__ = "diagnostic_sessions"

    id = Column(String, primary_key=True)
    defect_type = Column(String, nullable=True)
    defect_confidence = Column(Float, nullable=True)
    serial_number = Column(String, nullable=True)
    diagnosis = Column(Text, nullable=True)
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
