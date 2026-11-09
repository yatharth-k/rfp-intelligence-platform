"""FastAPI server for RFP Intelligence Platform."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ingestion.report import ingest_folder
from search.chunker import chunk_document
from search.indexer import Indexer
from search.keyword_store import KeywordStore, FileRegistry
from search.vector_store import QdrantVectorStore
from core.embeddings import EmbeddingClient
from core.tracing import Tracer
from core.limiter import RateLimiter
from config.settings import get_settings
from schemas import ParsedDocument, Chunk
from agents.qa import (
    classify_question,
    resolve_bids,
    build_final_record,
    answer_question,
    QAAnswer,
    FinalBidRecord,
)
from agents.orchestrator import Orchestrator
from agents.extraction import run_extraction
from agents.state import init_run, BidState
from search.engine import SearchEngine, SearchMode
from core.llm import LLMGateway

# FastAPI app
app = FastAPI(
    title="RFP Intelligence Platform API",
    description="API for ingesting, indexing, and querying RFP documents",
    version="1.0.0",
)

# CORS for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global settings
cfg = get_settings()

# Initialize shared components
storage_dir = Path("storage")
storage_dir.mkdir(exist_ok=True)

keyword_store = KeywordStore(storage_dir / "keyword.sqlite")
file_registry = FileRegistry(storage_dir / "file_registry.sqlite")
vector_store = QdrantVectorStore(path=str(storage_dir / "qdrant"), dim=2048)
embedding_client = EmbeddingClient(
    model=cfg.EMBED_MODEL,
    embed_url=cfg.NVIDIA_BASE_URL,
    api_key=cfg.NVIDIA_EMBED_KEY or cfg.NVIDIA_API_KEY,
    batch_size=cfg.EMBED_BATCH_SIZE,
)
tracer = Tracer(run_id="api_server")
rate_limiter = RateLimiter(max_per_minute=cfg.MAX_RPM)

indexer = Indexer(
    keyword_store=keyword_store,
    vector_store=vector_store,
    file_registry=file_registry,
    embedding_client=embedding_client,
    tracer=tracer,
    rate_limiter=rate_limiter,
)


# ============================================================
# Pydantic models for API requests/responses
# ============================================================

class IngestRequest(BaseModel):
    bid_id: str
    folder_path: str


class IngestResponse(BaseModel):
    bid_id: str
    files_parsed: int
    ok_count: int
    partial_count: int
    failed_count: int
    skipped_count: int
    total_pages: int
    warnings: List[str]


class ChunkInfo(BaseModel):
    chunk_id: str
    file_name: str
    doc_type: str
    addendum_number: Optional[int]
    page_start: int
    page_end: int
    section: Optional[str]
    kind: str
    token_count: int
    text_preview: str


class IndexRequest(BaseModel):
    bid_id: str
    force: bool = False


class IndexResponse(BaseModel):
    bid_id: str
    files_indexed: int
    chunks_added: int
    embedding_calls: int
    cache_hits: int
    duration_seconds: float


class SearchRequest(BaseModel):
    query: str
    bid_ids: List[str]
    top_k: int = 5
    mode: str = "fusion"  # bm25, vector, fusion, rerank


class SearchResult(BaseModel):
    rank: int
    chunk_id: str
    file_name: str
    doc_type: str
    page_start: int
    page_end: int
    section: Optional[str]
    kind: str
    text: str
    score: Optional[float] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total_count: int
    degraded: bool


class AskRequest(BaseModel):
    bid_id: str
    question: str
    mode: str = "fusion"
    top_k: int = 5


class AskResponse(BaseModel):
    bid_id: str
    question: str
    answer: str
    confidence: float
    sources: List[SearchResult]
    question_type: str


class ExtractionResponse(BaseModel):
    bid_id: str
    status: str
    fields: Dict[str, Any]
    validation: Dict[str, int]


# ============================================================
# API Endpoints
# ============================================================

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "RFP Intelligence Platform"}


@app.get("/bids")
async def list_bids():
    """List all indexed bids."""
    bids = indexer.list_bids()
    return {"bids": bids}


@app.post("/ingest", response_model=IngestResponse)
async def ingest_documents(request: IngestRequest):
    """Ingest documents from a folder."""
    folder = Path(request.folder_path)
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Folder not found: {folder}")

    try:
        docs, report = ingest_folder(folder, bid_id=request.bid_id)

        # Attach chunks to documents for indexing
        for d in docs:
            d.chunks = chunk_document(d, cfg)

        total_pages = sum(len(d.pages) for d in docs)
        warnings = []
        if hasattr(report, 'warnings'):
            warnings = report.warnings

        return IngestResponse(
            bid_id=request.bid_id,
            files_parsed=len(docs),
            ok_count=report.ok_count,
            partial_count=report.partial_count,
            failed_count=report.failed_count,
            skipped_count=report.skipped_count,
            total_pages=total_pages,
            warnings=warnings,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {str(e)}")


@app.post("/index", response_model=IndexResponse)
async def index_documents(request: IndexRequest):
    """Index documents for a bid."""
    folder = Path("data") / request.bid_id
    if not folder.exists():
        raise HTTPException(status_code=404, detail=f"Bid folder not found: {folder}")

    try:
        # Re-ingest to get fresh documents with chunks
        docs, _ = ingest_folder(folder, bid_id=request.bid_id)
        for d in docs:
            d.chunks = chunk_document(d, cfg)

        rep = indexer.index_documents(docs, bid_id=request.bid_id, embed_model=cfg.EMBED_MODEL, force=request.force)

        return IndexResponse(
            bid_id=request.bid_id,
            files_indexed=rep.get("files_indexed", 0),
            chunks_added=rep.get("chunks_added", 0),
            embedding_calls=rep.get("embedding_calls", 0),
            cache_hits=rep.get("cache_hits", 0),
            duration_seconds=rep.get("duration_seconds", 0.0),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Indexing failed: {str(e)}")


@app.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest):
    """Search indexed documents."""
    from search.engine import SearchEngine, SearchMode

    try:
        eng = SearchEngine(
            keyword_store=keyword_store,
            vector_store=vector_store,
            embedding_client=embedding_client,
        )

        # Support both design mode names (dense, bm25, hybrid, hybrid_rerank)
        # and legacy mode names (vector, bm25, fusion, rerank)
        mode_map = {
            "dense": SearchMode.VECTOR,
            "bm25": SearchMode.BM25,
            "hybrid": SearchMode.FUSION,
            "hybrid_rerank": SearchMode.RERANK,
            # Legacy aliases
            "vector": SearchMode.VECTOR,
            "fusion": SearchMode.FUSION,
            "rerank": SearchMode.RERANK,
        }
        mode = mode_map.get(request.mode, SearchMode.FUSION)

        r = eng.search(
            request.query,
            filters={"bid_ids": request.bid_ids},
            top_k=request.top_k,
            mode=mode,
        )

        results = []
        for i, x in enumerate(r.get('results', [])):
            results.append(SearchResult(
                rank=i + 1,
                chunk_id=x.get('chunk_id', ''),
                file_name=x.get('file_name', 'unknown'),
                doc_type=x.get('doc_type', ''),
                page_start=x.get('page_start', 0),
                page_end=x.get('page_end', 0),
                section=x.get('section'),
                kind=x.get('kind', ''),
                text=x.get('text', ''),
                score=x.get('score') or x.get('bm25_score') or x.get('rrf_score') or x.get('cross_encoder_score') or x.get('vector_score'),
            ))

        return SearchResponse(
            query=request.query,
            results=results,
            total_count=len(results),
            degraded=r.get('degraded', False),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@app.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest):
    """Ask a question about a bid."""
    try:
        # Initialize search engine and LLM gateway
        search_engine = SearchEngine(
            keyword_store=keyword_store,
            vector_store=vector_store,
            embedding_client=embedding_client,
        )
        llm_gateway = LLMGateway(
            api_key=cfg.NVIDIA_API_KEY,
            base_url=cfg.NVIDIA_BASE_URL,
        )

        # Create initial state for the Q&A run
        state: BidState = init_run(
            bid_id=request.bid_id,
            goal="ask",
            question=request.question,
        )

        # Use the QA system with proper state
        record = build_final_record(state)

        # Classify question
        qtype, resolution = classify_question(
            request.question,
            state
        )

        # Answer question
        result = await answer_question(
            question=request.question,
            state=state,
            search_engine=search_engine,
        )

        sources = []
        for i, src in enumerate(result.citations):
            sources.append(SearchResult(
                rank=i + 1,
                chunk_id=src.chunk_id if hasattr(src, 'chunk_id') else '',
                file_name=src.file_name,
                doc_type=src.doc_type if hasattr(src, 'doc_type') else '',
                page_start=src.page,
                page_end=src.page,
                section=None,
                kind='',
                text=src.text or '',
                score=None,
            ))

        return AskResponse(
            bid_id=request.bid_id,
            question=request.question,
            answer=result.answer,
            confidence=result.confidence,
            sources=sources,
            question_type=qtype.value if hasattr(qtype, 'value') else str(qtype),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Question answering failed: {str(e)}")


@app.post("/extract/{bid_id}", response_model=ExtractionResponse)
async def extract_fields(bid_id: str):
    """Extract fields from a bid."""
    try:
        # Initialize search engine and LLM gateway
        search_engine = SearchEngine(
            keyword_store=keyword_store,
            vector_store=vector_store,
            embedding_client=embedding_client,
        )
        llm_gateway = LLMGateway(
            api_key=cfg.NVIDIA_API_KEY,
            base_url=cfg.NVIDIA_BASE_URL,
        )

        # Use the orchestrator to run extraction
        orchestrator = Orchestrator(search_engine=search_engine)

        # Create initial state for extraction run
        state: BidState = init_run(
            bid_id=bid_id,
            goal="extract",
        )

        # Run the extraction pipeline via orchestrator
        result_state = orchestrator.run_extract(state)

        # Build final record from the result state
        record = build_final_record(result_state)

        fields = {}
        if hasattr(record, 'fields'):
            for name, field in record.fields.items():
                fields[name] = {
                    "value": field.get("value"),
                    "confidence": field.get("confidence", 0.0),
                    "sources": field.get("sources", []),
                    "status": field.get("status", "not_found"),
                    "notes": field.get("notes", ""),
                }

        validation = {
            "passed": record.validation.get("passed", 0) if hasattr(record, 'validation') else 0,
            "failed": record.validation.get("failed", 0) if hasattr(record, 'validation') else 0,
            "not_found": record.validation.get("not_found", 0) if hasattr(record, 'validation') else 0,
        }

        return ExtractionResponse(
            bid_id=bid_id,
            status="completed",
            fields=fields,
            validation=validation,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Extraction failed: {str(e)}")


@app.get("/documents/{bid_id}")
async def list_documents(bid_id: str):
    """List all documents for a bid."""
    try:
        # Search for all chunks from this bid to get document info
        results = keyword_store.search("*", filters={"bid_ids": [bid_id]}, limit=1000)

        # Group by file
        files = {}
        for chunk in results:
            fname = chunk.get('file_name', 'unknown')
            if fname not in files:
                files[fname] = {
                    "file_name": fname,
                    "doc_type": chunk.get('doc_type', ''),
                    "addendum_number": chunk.get('addendum_number'),
                    "chunks": 0,
                    "pages": set(),
                }
            files[fname]["chunks"] += 1
            files[fname]["pages"].add(chunk.get('page_start', 0))

        # Convert to list
        doc_list = []
        for fname, info in files.items():
            doc_list.append({
                "file_name": fname,
                "doc_type": info["doc_type"],
                "addendum_number": info["addendum_number"],
                "chunk_count": info["chunks"],
                "page_count": len([p for p in info["pages"] if p > 0]),
            })

        return {"bid_id": bid_id, "documents": doc_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list documents: {str(e)}")


@app.get("/chunks/{bid_id}/{file_name}")
async def get_document_chunks(bid_id: str, file_name: str):
    """Get all chunks for a specific document."""
    try:
        results = keyword_store.search(
            "*",
            filters={"bid_ids": [bid_id], "file_names": [file_name]},
            limit=1000
        )

        chunks = []
        for chunk in results:
            chunks.append({
                "chunk_id": chunk.get('chunk_id'),
                "text": chunk.get('text', ''),
                "doc_type": chunk.get('doc_type'),
                "addendum_number": chunk.get('addendum_number'),
                "page_start": chunk.get('page_start'),
                "page_end": chunk.get('page_end'),
                "section": chunk.get('section'),
                "kind": chunk.get('kind'),
                "embedded": chunk.get('embedded', False),
            })

        return {"bid_id": bid_id, "file_name": file_name, "chunks": chunks}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get chunks: {str(e)}")


@app.post("/upload")
async def upload_files(bid_id: str = Form(...), files: List[UploadFile] = File(...)):
    """Upload files to a bid folder."""
    try:
        bid_folder = Path("data") / bid_id
        bid_folder.mkdir(parents=True, exist_ok=True)

        saved_files = []
        for file in files:
            file_path = bid_folder / file.filename
            content = await file.read()
            file_path.write_bytes(content)
            saved_files.append(file.filename)

        return {"bid_id": bid_id, "saved_files": saved_files}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)