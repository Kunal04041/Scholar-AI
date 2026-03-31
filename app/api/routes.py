from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse
from app.api.schemas import IngestRequest, IngestResponse, QueryRequest, QueryResponse
from app.agents.planner import run_pipeline
from app.ingestion.loader import load_document
from app.ingestion.chunker import chunk_documents
from app.retrieval.vector_store import store_chunks
import uuid
import json
import logging

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(request: IngestRequest):
    try:
        docs = await load_document(url=request.url, text=request.text, source=request.source_name)
        chunks = chunk_documents(docs)
        stored = await store_chunks(chunks)
        return IngestResponse(status="success", chunks_stored=stored, source=request.source_name)
    except Exception as e:
        logger.error(f"Ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """Ingest an uploaded PDF file."""
    try:
        # Save to temp file because PyPDFLoader (usually) needs a path
        import tempfile
        import os
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(await file.read())
            tmp_path = tmp.name
        
        try:
            docs = await load_document(url=tmp_path, source=file.filename)
            chunks = chunk_documents(docs)
            stored = await store_chunks(chunks)
            return IngestResponse(status="success", chunks_stored=stored, source=file.filename)
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
                
    except Exception as e:
        logger.error(f"File ingest error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Run multi-agent pipeline and return answer with sources."""
    try:
        session_id = request.session_id or str(uuid.uuid4())
        result = await run_pipeline(
            query=request.query,
            session_id=session_id,
            top_k=request.top_k,
            evaluate=request.evaluate,
        )
        return QueryResponse(**result, session_id=session_id)
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/query/stream")
async def query_stream(request: QueryRequest):
    """Streaming endpoint — returns SSE chunks."""
    session_id = request.session_id or str(uuid.uuid4())

    async def event_generator():
        try:
            async for chunk in run_pipeline(
                query=request.query,
                session_id=session_id,
                top_k=request.top_k,
                stream=True,
            ):
                yield f"data: {json.dumps(chunk)}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
