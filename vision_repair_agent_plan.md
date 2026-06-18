# Vision-Driven Self-Correcting Repair Agent — Project Plan

> A stateful, agentic AI system for autonomous hardware diagnostic workflows using LangGraph, Computer Vision, OCR, and RAG.

---

## Project Overview

**Goal:** Build a production-ready agentic AI system that autonomously diagnoses hardware defects by chaining Computer Vision (CV), OCR, and Retrieval-Augmented Generation (RAG) in a self-correcting feedback loop — with stateful memory across multi-step tool calls.

**Key Outcomes:**
- 90%+ diagnostic accuracy over a 500+ document RAG corpus
- Autonomous self-correction: crop → OCR → re-query RAG without human intervention
- Multi-step tool chain: CV → RAG → OCR → RAG with stateful memory
- 70% reduction in diagnostic resolution time

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                    LangGraph Agent Graph                    │
│                                                             │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌───────┐ │
│  │ CV Node  │───▶│RAG Query │───▶│ OCR Node │───▶│  RAG  │ │
│  │(defect   │    │ (initial │    │(serial # │    │(re-   │ │
│  │ detect)  │    │ lookup)  │    │ extract) │    │query) │ │
│  └──────────┘    └──────────┘    └──────────┘    └───────┘ │
│        │                │               │              │    │
│        └────────────────┴───────────────┴──────────────┘   │
│                     Shared Agent State                      │
│              (PostgreSQL-backed checkpointing)              │
└─────────────────────────────────────────────────────────────┘
```

**Flow:**
1. Image input → CV node detects defect region and type
2. Initial RAG query with defect classification → retrieves candidate docs
3. If confidence < threshold → self-correction triggers
4. OCR node crops image region → extracts serial number
5. Re-queries RAG with serial number + defect type for precision results
6. Final diagnosis output with evidence citations from corpus

---

## Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| Agent Orchestration | LangGraph | Stateful graph-based agent with conditional edges |
| LLM / Embeddings | OpenAI API (GPT-4o + text-embedding-3-small) | Reasoning, summarization, vector embeddings |
| RAG Framework | LangChain + FAISS / pgvector | Document retrieval pipeline |
| Computer Vision | OpenCV + PIL (Pillow) | Defect detection, image cropping, region localization |
| OCR | Tesseract (pytesseract) + OpenCV preprocessing | Serial number extraction from hardware labels |
| Vector Store | PostgreSQL + pgvector | Persistent vector embeddings for 500+ document corpus |
| State Persistence | PostgreSQL (LangGraph checkpointer) | Stateful memory across agent steps |
| API Layer | FastAPI | REST API for submitting images and retrieving diagnoses |
| Environment | Python 3.11+, Poetry | Dependency management |
| Testing | pytest + pytest-asyncio | Unit + integration tests |
| Containerization | Docker + docker-compose | Local dev parity |

---

## Project Directory Structure

```
vision-repair-agent/
├── README.md
├── pyproject.toml                  # Poetry dependencies
├── .env.example                    # Environment variable template
├── docker-compose.yml              # PostgreSQL + pgvector service
├── Makefile                        # Dev shortcuts
│
├── docs/
│   └── corpus/                     # Place 500+ hardware diagnostic PDFs/docs here
│
├── src/
│   └── repair_agent/
│       ├── __init__.py
│       │
│       ├── config.py               # Settings via pydantic-settings
│       │
│       ├── agent/
│       │   ├── __init__.py
│       │   ├── graph.py            # LangGraph StateGraph definition
│       │   ├── state.py            # AgentState TypedDict
│       │   ├── nodes/
│       │   │   ├── __init__.py
│       │   │   ├── cv_node.py      # Computer Vision defect detection
│       │   │   ├── rag_node.py     # RAG query node
│       │   │   ├── ocr_node.py     # OCR serial number extraction
│       │   │   └── diagnosis_node.py  # Final diagnosis synthesis
│       │   └── edges.py            # Conditional edge logic (self-correction trigger)
│       │
│       ├── tools/
│       │   ├── __init__.py
│       │   ├── cv_tools.py         # OpenCV defect detection utilities
│       │   ├── ocr_tools.py        # pytesseract OCR utilities
│       │   └── rag_tools.py        # LangChain retrieval chain utilities
│       │
│       ├── rag/
│       │   ├── __init__.py
│       │   ├── ingestor.py         # Document loader + chunking + embedding pipeline
│       │   ├── retriever.py        # pgvector retriever setup
│       │   └── prompts.py          # RAG prompt templates
│       │
│       ├── db/
│       │   ├── __init__.py
│       │   ├── connection.py       # SQLAlchemy async engine
│       │   ├── models.py           # ORM models (DiagnosticSession, DocumentChunk)
│       │   └── migrations/         # Alembic migrations
│       │
│       └── api/
│           ├── __init__.py
│           ├── main.py             # FastAPI app entrypoint
│           ├── routes/
│           │   ├── diagnose.py     # POST /diagnose endpoint
│           │   └── health.py       # GET /health endpoint
│           └── schemas.py          # Pydantic request/response models
│
├── scripts/
│   ├── ingest_corpus.py            # One-shot script: load docs → chunk → embed → store
│   └── seed_test_docs.py           # Seed minimal test corpus for CI
│
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_cv_node.py
    │   ├── test_ocr_node.py
    │   └── test_rag_node.py
    └── integration/
        ├── test_agent_graph.py     # Full graph execution test
        └── test_api.py             # FastAPI endpoint tests
```

---

## Implementation Plan

### Phase 1 — Project Scaffolding & Environment (Day 1)

**Objective:** Get a runnable skeleton with all dependencies configured.

#### 1.1 Initialize Project

```bash
mkdir vision-repair-agent && cd vision-repair-agent
poetry init --name repair-agent --python "^3.11"
```

#### 1.2 Install Dependencies

```toml
# pyproject.toml — [tool.poetry.dependencies]
python = "^3.11"
langgraph = ">=0.2"
langchain = ">=0.3"
langchain-openai = ">=0.2"
langchain-community = ">=0.3"
langchain-postgres = ">=0.0.9"
openai = ">=1.0"
fastapi = ">=0.111"
uvicorn = {extras = ["standard"], version = ">=0.29"}
sqlalchemy = {extras = ["asyncio"], version = ">=2.0"}
asyncpg = ">=0.29"
alembic = ">=1.13"
pgvector = ">=0.3"
psycopg = {extras = ["binary"], version = ">=3.1"}
pytesseract = ">=0.3"
opencv-python-headless = ">=4.9"
pillow = ">=10.3"
pydantic-settings = ">=2.2"
python-multipart = ">=0.0.9"
pypdf = ">=4.2"
faiss-cpu = ">=1.8"  # fallback if pgvector isn't available
tenacity = ">=8.3"
structlog = ">=24.1"

[tool.poetry.dev-dependencies]
pytest = ">=8.0"
pytest-asyncio = ">=0.23"
pytest-mock = ">=3.12"
httpx = ">=0.27"
ruff = ">=0.4"
mypy = ">=1.9"
```

#### 1.3 Docker Setup

```yaml
# docker-compose.yml
version: "3.9"
services:
  postgres:
    image: pgvector/pgvector:pg16
    environment:
      POSTGRES_USER: repair_agent
      POSTGRES_PASSWORD: repair_agent
      POSTGRES_DB: repair_agent_db
    ports:
      - "5432:5432"
    volumes:
      - pgdata:/var/lib/postgresql/data

volumes:
  pgdata:
```

#### 1.4 Environment Variables

```bash
# .env.example
OPENAI_API_KEY=sk-...
DATABASE_URL=postgresql+asyncpg://repair_agent:repair_agent@localhost:5432/repair_agent_db
SYNC_DATABASE_URL=postgresql+psycopg://repair_agent:repair_agent@localhost:5432/repair_agent_db
CORPUS_DIR=./docs/corpus
EMBEDDING_MODEL=text-embedding-3-small
LLM_MODEL=gpt-4o
CONFIDENCE_THRESHOLD=0.75      # below this, self-correction loop triggers
MAX_CORRECTION_RETRIES=3
TESSERACT_CMD=/usr/bin/tesseract
LOG_LEVEL=INFO
```

---

### Phase 2 — Agent State & Graph Definition (Day 2)

**Objective:** Define the LangGraph graph with all nodes, edges, and stateful memory.

#### 2.1 Agent State (`src/repair_agent/agent/state.py`)

```python
from typing import TypedDict, Annotated, Optional, List
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Input
    image_bytes: bytes                      # Raw image input
    image_path: Optional[str]               # Optional file path

    # CV outputs
    defect_type: Optional[str]              # e.g., "burn_mark", "crack", "corrosion"
    defect_confidence: Optional[float]      # 0.0 - 1.0
    defect_bbox: Optional[tuple]            # (x, y, w, h) bounding box
    cropped_image_bytes: Optional[bytes]    # Cropped region for OCR

    # OCR outputs
    serial_number: Optional[str]            # Extracted serial number
    ocr_confidence: Optional[float]

    # RAG outputs
    rag_documents: Optional[List[dict]]     # Retrieved document chunks
    rag_query: Optional[str]               # Query used for RAG
    diagnosis: Optional[str]               # Final diagnosis text

    # Control flow
    correction_attempts: int               # Self-correction iteration count
    self_correction_triggered: bool        # Whether correction loop ran

    # Message history (LangGraph native)
    messages: Annotated[List[BaseMessage], add_messages]
```

#### 2.2 Graph Definition (`src/repair_agent/agent/graph.py`)

```python
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from repair_agent.agent.state import AgentState
from repair_agent.agent.nodes.cv_node import cv_node
from repair_agent.agent.nodes.rag_node import rag_node
from repair_agent.agent.nodes.ocr_node import ocr_node
from repair_agent.agent.nodes.diagnosis_node import diagnosis_node
from repair_agent.agent.edges import should_self_correct, correction_complete

def build_graph(checkpointer) -> StateGraph:
    graph = StateGraph(AgentState)

    # Register nodes
    graph.add_node("cv", cv_node)
    graph.add_node("rag_initial", rag_node)
    graph.add_node("ocr", ocr_node)
    graph.add_node("rag_corrected", rag_node)          # Same node, re-queried with serial
    graph.add_node("diagnosis", diagnosis_node)

    # Entry → CV
    graph.set_entry_point("cv")

    # CV → Initial RAG
    graph.add_edge("cv", "rag_initial")

    # Initial RAG → conditional: self-correct or diagnose
    graph.add_conditional_edges(
        "rag_initial",
        should_self_correct,
        {
            "self_correct": "ocr",
            "diagnose": "diagnosis",
        }
    )

    # OCR → Corrected RAG
    graph.add_edge("ocr", "rag_corrected")

    # Corrected RAG → conditional: retry or diagnose
    graph.add_conditional_edges(
        "rag_corrected",
        correction_complete,
        {
            "retry": "ocr",
            "diagnose": "diagnosis",
        }
    )

    # Diagnosis → END
    graph.add_edge("diagnosis", END)

    return graph.compile(checkpointer=checkpointer)


async def get_agent(db_connection_string: str):
    checkpointer = AsyncPostgresSaver.from_conn_string(db_connection_string)
    await checkpointer.setup()
    return build_graph(checkpointer)
```

#### 2.3 Conditional Edges (`src/repair_agent/agent/edges.py`)

```python
from repair_agent.agent.state import AgentState
from repair_agent.config import settings

def should_self_correct(state: AgentState) -> str:
    """Trigger OCR self-correction if RAG confidence is below threshold."""
    confidence = state.get("defect_confidence", 0.0)
    attempts = state.get("correction_attempts", 0)
    if confidence < settings.CONFIDENCE_THRESHOLD and attempts < settings.MAX_CORRECTION_RETRIES:
        return "self_correct"
    return "diagnose"

def correction_complete(state: AgentState) -> str:
    """After OCR + re-query: decide if another retry is needed or proceed to diagnosis."""
    attempts = state.get("correction_attempts", 0)
    serial = state.get("serial_number")
    if serial and attempts < settings.MAX_CORRECTION_RETRIES:
        return "diagnose"
    if attempts >= settings.MAX_CORRECTION_RETRIES:
        return "diagnose"
    return "retry"
```

---

### Phase 3 — Node Implementations (Days 3–5)

#### 3.1 CV Node (`src/repair_agent/agent/nodes/cv_node.py`)

**Responsibilities:**
- Accept raw image bytes
- Preprocess (grayscale, denoise, threshold)
- Detect defect regions using contour detection or pre-trained classifier
- Return defect type, confidence, and bounding box

**Implementation Steps:**
1. Decode image bytes with `cv2.imdecode`
2. Convert to grayscale → Gaussian blur → Otsu thresholding
3. Find contours → filter by area (ignore noise)
4. Classify defect type:
   - **Simple approach:** Rule-based (burn = brown/black regions; crack = elongated contours; corrosion = greenish hue)
   - **Advanced approach:** Load a fine-tuned `torchvision` or ONNX classifier
5. Crop the region of interest (ROI) centered on the largest defect contour
6. Return updated state with `defect_type`, `defect_confidence`, `defect_bbox`, `cropped_image_bytes`

```python
import cv2
import numpy as np
from repair_agent.agent.state import AgentState

DEFECT_CLASSES = ["burn_mark", "crack", "corrosion", "delamination", "normal"]

async def cv_node(state: AgentState) -> dict:
    img_array = np.frombuffer(state["image_bytes"], np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) > 500]

    if not contours:
        return {"defect_type": "normal", "defect_confidence": 0.95}

    largest = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest)
    cropped = img[y:y+h, x:x+w]

    # TODO: Replace with trained classifier for production accuracy
    defect_type = _classify_defect_heuristic(img, largest)
    confidence = _estimate_confidence(contours)

    _, crop_encoded = cv2.imencode(".png", cropped)
    cropped_bytes = crop_encoded.tobytes()

    return {
        "defect_type": defect_type,
        "defect_confidence": confidence,
        "defect_bbox": (x, y, w, h),
        "cropped_image_bytes": cropped_bytes,
    }


def _classify_defect_heuristic(img, contour) -> str:
    # Heuristic: aspect ratio → crack; dark region → burn; etc.
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = w / max(h, 1)
    if aspect_ratio > 4:
        return "crack"
    roi = img[y:y+h, x:x+w]
    mean_val = roi.mean()
    if mean_val < 60:
        return "burn_mark"
    return "corrosion"


def _estimate_confidence(contours) -> float:
    # More contours + larger area = higher confidence
    total_area = sum(cv2.contourArea(c) for c in contours)
    return min(0.5 + total_area / 100000, 0.99)
```

#### 3.2 OCR Node (`src/repair_agent/agent/nodes/ocr_node.py`)

**Responsibilities:**
- Take `cropped_image_bytes` from state
- Preprocess for OCR (scale up, sharpen, binarize)
- Run pytesseract to extract text
- Parse serial number using regex
- Update state and increment `correction_attempts`

```python
import re
import pytesseract
import cv2
import numpy as np
from PIL import Image
import io
from repair_agent.agent.state import AgentState
from repair_agent.config import settings

SERIAL_PATTERN = re.compile(r"[A-Z0-9]{2,4}[-\s]?[A-Z0-9]{4,10}", re.IGNORECASE)

async def ocr_node(state: AgentState) -> dict:
    img_bytes = state.get("cropped_image_bytes") or state["image_bytes"]
    img_array = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    # Preprocessing: upscale + sharpen + binarize
    img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sharpened = cv2.filter2D(gray, -1, np.array([[0,-1,0],[-1,5,-1],[0,-1,0]]))
    _, binary = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    pil_img = Image.fromarray(binary)
    raw_text = pytesseract.image_to_string(
        pil_img,
        config="--psm 6 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-"
    )

    serial_match = SERIAL_PATTERN.search(raw_text)
    serial_number = serial_match.group(0).strip() if serial_match else None

    ocr_data = pytesseract.image_to_data(pil_img, output_type=pytesseract.Output.DICT)
    confidences = [int(c) for c in ocr_data["conf"] if int(c) > 0]
    ocr_conf = sum(confidences) / len(confidences) / 100 if confidences else 0.0

    return {
        "serial_number": serial_number,
        "ocr_confidence": ocr_conf,
        "correction_attempts": state.get("correction_attempts", 0) + 1,
        "self_correction_triggered": True,
    }
```

#### 3.3 RAG Node (`src/repair_agent/agent/nodes/rag_node.py`)

**Responsibilities:**
- Build a query string from current state (defect type + optional serial number)
- Query pgvector retriever for top-k document chunks
- Return retrieved documents and constructed query

```python
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from repair_agent.agent.state import AgentState
from repair_agent.config import settings

async def rag_node(state: AgentState) -> dict:
    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name="hardware_docs",
        connection=settings.SYNC_DATABASE_URL,
    )

    # Build query: use serial number if available, otherwise defect type only
    defect = state.get("defect_type", "unknown defect")
    serial = state.get("serial_number")

    if serial:
        query = f"Hardware diagnostic for serial number {serial} with defect: {defect}"
    else:
        query = f"Hardware diagnostic procedure for {defect}"

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    docs = await retriever.ainvoke(query)

    rag_docs = [
        {"content": doc.page_content, "metadata": doc.metadata, "score": getattr(doc, "score", None)}
        for doc in docs
    ]

    return {
        "rag_query": query,
        "rag_documents": rag_docs,
    }
```

#### 3.4 Diagnosis Node (`src/repair_agent/agent/nodes/diagnosis_node.py`)

**Responsibilities:**
- Synthesize a final diagnosis from CV results + RAG documents
- Use GPT-4o to generate structured diagnosis text
- Attach metadata: confidence, self-correction flag, serial number

```python
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from repair_agent.agent.state import AgentState
from repair_agent.config import settings

DIAGNOSIS_SYSTEM_PROMPT = """You are an expert hardware diagnostic AI. Given defect detection results
and retrieved documentation, produce a structured diagnosis including:
1. Defect classification
2. Root cause analysis
3. Recommended repair actions (step-by-step)
4. Parts or references from documentation
Be concise, factual, and cite the document sources by their metadata filenames."""

async def diagnosis_node(state: AgentState) -> dict:
    llm = ChatOpenAI(model=settings.LLM_MODEL, temperature=0)

    docs_text = "\n\n".join(
        f"[Doc: {d['metadata'].get('source', 'unknown')}]\n{d['content']}"
        for d in (state.get("rag_documents") or [])
    )

    user_message = f"""
Defect Type: {state.get('defect_type', 'unknown')}
Confidence: {state.get('defect_confidence', 0):.2%}
Serial Number: {state.get('serial_number', 'not detected')}
Self-Correction Applied: {state.get('self_correction_triggered', False)}
Correction Attempts: {state.get('correction_attempts', 0)}

Retrieved Documentation:
{docs_text or 'No documentation retrieved.'}

Provide a complete diagnostic report.
"""

    response = await llm.ainvoke([
        SystemMessage(content=DIAGNOSIS_SYSTEM_PROMPT),
        HumanMessage(content=user_message),
    ])

    return {"diagnosis": response.content}
```

---

### Phase 4 — RAG Corpus Ingestion (Day 6)

**Objective:** Load, chunk, embed, and store 500+ hardware diagnostic documents in PostgreSQL + pgvector.

#### 4.1 Ingestor (`src/repair_agent/rag/ingestor.py`)

**Steps:**
1. Walk `CORPUS_DIR` for `.pdf`, `.txt`, `.docx` files
2. Load with `langchain_community.document_loaders` (PyPDFLoader, TextLoader, etc.)
3. Chunk with `RecursiveCharacterTextSplitter` — chunk size 1000, overlap 150
4. Embed with `text-embedding-3-small`
5. Store in pgvector collection `hardware_docs`
6. Track ingested files to avoid re-embedding on re-runs

```python
import asyncio
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_postgres.vectorstores import PGVector
from repair_agent.config import settings

async def ingest_corpus():
    loader = DirectoryLoader(
        settings.CORPUS_DIR,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)
    vectorstore = PGVector.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="hardware_docs",
        connection=settings.SYNC_DATABASE_URL,
        pre_delete_collection=False,  # incremental ingestion
    )

    print(f"Ingested {len(chunks)} chunks from {len(docs)} documents.")

if __name__ == "__main__":
    asyncio.run(ingest_corpus())
```

---

### Phase 5 — FastAPI Layer (Day 7)

**Objective:** Expose the agent via a clean REST API.

#### 5.1 API Schema (`src/repair_agent/api/schemas.py`)

```python
from pydantic import BaseModel
from typing import Optional

class DiagnoseResponse(BaseModel):
    session_id: str
    diagnosis: str
    defect_type: Optional[str]
    defect_confidence: Optional[float]
    serial_number: Optional[str]
    self_correction_triggered: bool
    correction_attempts: int
    rag_documents_used: int
```

#### 5.2 Diagnose Route (`src/repair_agent/api/routes/diagnose.py`)

```python
import uuid
from fastapi import APIRouter, UploadFile, File, Depends
from repair_agent.api.schemas import DiagnoseResponse
from repair_agent.agent.graph import get_agent
from repair_agent.agent.state import AgentState
from repair_agent.config import settings

router = APIRouter()

@router.post("/diagnose", response_model=DiagnoseResponse)
async def diagnose(file: UploadFile = File(...)):
    image_bytes = await file.read()
    session_id = str(uuid.uuid4())

    agent = await get_agent(settings.SYNC_DATABASE_URL)
    config = {"configurable": {"thread_id": session_id}}

    initial_state: AgentState = {
        "image_bytes": image_bytes,
        "correction_attempts": 0,
        "self_correction_triggered": False,
        "messages": [],
    }

    final_state = await agent.ainvoke(initial_state, config=config)

    return DiagnoseResponse(
        session_id=session_id,
        diagnosis=final_state.get("diagnosis", "Unable to diagnose."),
        defect_type=final_state.get("defect_type"),
        defect_confidence=final_state.get("defect_confidence"),
        serial_number=final_state.get("serial_number"),
        self_correction_triggered=final_state.get("self_correction_triggered", False),
        correction_attempts=final_state.get("correction_attempts", 0),
        rag_documents_used=len(final_state.get("rag_documents") or []),
    )
```

#### 5.3 FastAPI App (`src/repair_agent/api/main.py`)

```python
from fastapi import FastAPI
from repair_agent.api.routes.diagnose import router as diagnose_router
from repair_agent.api.routes.health import router as health_router

app = FastAPI(title="Vision Repair Agent", version="1.0.0")
app.include_router(diagnose_router, prefix="/api/v1")
app.include_router(health_router, prefix="/api/v1")
```

**Run:** `uvicorn repair_agent.api.main:app --reload`

---

### Phase 6 — Database Models & Migrations (Day 8)

#### 6.1 ORM Models (`src/repair_agent/db/models.py`)

```python
from sqlalchemy import Column, String, Float, Boolean, Integer, Text, DateTime, func
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase): pass

class DiagnosticSession(Base):
    __tablename__ = "diagnostic_sessions"
    id = Column(String, primary_key=True)
    defect_type = Column(String)
    defect_confidence = Column(Float)
    serial_number = Column(String, nullable=True)
    diagnosis = Column(Text)
    self_correction_triggered = Column(Boolean, default=False)
    correction_attempts = Column(Integer, default=0)
    created_at = Column(DateTime, server_default=func.now())
```

#### 6.2 Alembic Setup

```bash
alembic init src/repair_agent/db/migrations
alembic revision --autogenerate -m "initial_schema"
alembic upgrade head
```

---

### Phase 7 — Testing (Day 9)

#### 7.1 Unit Tests

**`tests/unit/test_cv_node.py`**
- Load a sample PNG with a known defect → assert `defect_type` is returned
- Test with blank image → assert `normal` classification
- Test bounding box coordinates are non-negative

**`tests/unit/test_ocr_node.py`**
- Create synthetic image with text `SN-ABC12345` → assert serial number extracted correctly
- Test with noisy image → assert partial extraction or graceful degradation
- Assert `correction_attempts` increments

**`tests/unit/test_rag_node.py`**
- Mock pgvector retriever → assert query built correctly with/without serial number
- Assert `rag_documents` list has expected length

#### 7.2 Integration Tests

**`tests/integration/test_agent_graph.py`**
- Submit a real hardware defect image → run full graph → assert `diagnosis` is non-empty
- Assert self-correction triggers when confidence < threshold
- Assert `serial_number` populates after OCR node
- Assert stateful memory: re-invoke with same `thread_id` → agent resumes from checkpoint

**`tests/integration/test_api.py`**
- POST `/api/v1/diagnose` with multipart image file → assert 200 + valid `DiagnoseResponse`
- Test invalid file type → assert 422 validation error
- Test large image → assert within response time SLA

---

### Phase 8 — Makefile & Developer Experience (Day 9)

```makefile
.PHONY: up down migrate ingest test run

up:
	docker-compose up -d

down:
	docker-compose down

migrate:
	alembic upgrade head

ingest:
	python scripts/ingest_corpus.py

test:
	pytest tests/ -v --asyncio-mode=auto

run:
	uvicorn repair_agent.api.main:app --reload --port 8000

lint:
	ruff check src/ tests/
	mypy src/
```

---

## Configuration (`src/repair_agent/config.py`)

```python
from pydantic_settings import BaseSettings, SettingsConfigDict

class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: str
    DATABASE_URL: str
    SYNC_DATABASE_URL: str
    CORPUS_DIR: str = "./docs/corpus"
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o"
    CONFIDENCE_THRESHOLD: float = 0.75
    MAX_CORRECTION_RETRIES: int = 3
    LOG_LEVEL: str = "INFO"

settings = Settings()
```

---

## Build Order (Step-by-Step for Claude Code)

Follow this exact sequence to build the project:

1. **Scaffold** — Create directory structure, `pyproject.toml`, `docker-compose.yml`, `.env.example`, `Makefile`
2. **Config** — Implement `config.py` with pydantic-settings
3. **Database** — `db/connection.py`, `db/models.py`, run `alembic init` and `alembic upgrade head`
4. **Agent State** — `agent/state.py` (TypedDict with all fields)
5. **CV Node** — `agent/nodes/cv_node.py` + `tools/cv_tools.py`
6. **OCR Node** — `agent/nodes/ocr_node.py` + `tools/ocr_tools.py`
7. **RAG Node** — `agent/nodes/rag_node.py` + `rag/ingestor.py` + `rag/retriever.py` + `rag/prompts.py`
8. **Diagnosis Node** — `agent/nodes/diagnosis_node.py`
9. **Edges** — `agent/edges.py`
10. **Graph** — `agent/graph.py` (wires all nodes + checkpointer)
11. **API** — `api/schemas.py`, `api/routes/diagnose.py`, `api/routes/health.py`, `api/main.py`
12. **Scripts** — `scripts/ingest_corpus.py`, `scripts/seed_test_docs.py`
13. **Tests** — All unit and integration tests
14. **README** — Complete setup + run instructions

---

## Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| LangGraph `StateGraph` over custom orchestration | Native stateful memory + checkpointing; conditional edges map cleanly to self-correction logic |
| pgvector over FAISS | Persistent, queryable, supports metadata filtering; no in-memory reloading |
| AsyncPostgresSaver for checkpointing | Same DB as pgvector; eliminates need for Redis or separate state store |
| pytesseract with preprocessing | Free, controllable, excellent for structured label text when image is preprocessed |
| GPT-4o for diagnosis | Stronger at structured output synthesis and following multi-doc prompts than GPT-3.5 |
| FastAPI over Flask | Native async, auto OpenAPI docs, Pydantic validation aligns with state schemas |

---

## Self-Correction Loop — Flow Detail

```
Image In
   │
   ▼
[CV Node] ──── defect_type="burn_mark", confidence=0.62
   │
   ▼
[RAG Node] ── query: "hardware diagnostic for burn_mark"
   │           retrieved 5 docs, but low specificity
   │
[Edge: should_self_correct] ── confidence(0.62) < threshold(0.75) → "self_correct"
   │
   ▼
[OCR Node] ── crops defect region → extracts "SN-XR9821A"
   │           correction_attempts = 1
   │
   ▼
[RAG Node] ── query: "Hardware diagnostic for serial number SN-XR9821A with defect: burn_mark"
   │           retrieved 5 highly specific docs ✓
   │
[Edge: correction_complete] ── serial found, attempts < max → "diagnose"
   │
   ▼
[Diagnosis Node] ── GPT-4o synthesizes diagnosis from CV + OCR + RAG
   │
   ▼
Final DiagnoseResponse → API Response
```

---

## Accuracy Strategy (90%+ Target)

- **Embedding model:** `text-embedding-3-small` with cosine similarity; use `text-embedding-3-large` for higher accuracy if budget allows
- **Chunking:** 1000 tokens with 150 overlap ensures context is preserved across chunk boundaries
- **Metadata filtering:** Filter by `defect_type` field in document metadata to narrow retrieval before vector search
- **Re-ranking:** Add a Cohere Rerank step after retrieval to reorder chunks by relevance before feeding to GPT-4o
- **Confidence tuning:** Adjust `CONFIDENCE_THRESHOLD` from 0.75 based on F1 score evaluation on test set
- **OCR accuracy:** The preprocessing pipeline (upscale 2×, sharpen, Otsu binarize) is critical; evaluate on a labeled serial number dataset

---

## README Sections to Include

1. **Prerequisites** (Python 3.11+, Docker, Tesseract, OpenAI API key)
2. **Quick Start** (clone → `.env` → `make up` → `make migrate` → `make ingest` → `make run`)
3. **API Usage** (curl example for `POST /api/v1/diagnose`)
4. **Architecture Diagram**
5. **Adding Documents to Corpus** (drop PDFs into `docs/corpus/` → `make ingest`)
6. **Running Tests** (`make test`)
7. **Configuration Reference** (all `.env` variables explained)

---

*Generated for Mohamed Afsar Harsath Arif — Vision-Driven Self-Correcting Repair Agent project plan.*
