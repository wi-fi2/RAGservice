"""FastAPI server for RAG service with real LLM integration."""

import os
import time
import asyncio
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, HttpUrl
import uvicorn
import logging
import httpx
import re
from dotenv import load_dotenv
import base64
import io
from typing import BinaryIO

from app.crawler import crawl_website
from app.indexer import VectorIndexer, index_crawled_pages
from app.utils import format_sources_for_prompt, MIN_RELEVANCE_SCORE
from app.metrics import MetricsCollector, metrics_handler

# Load environment variables from .env file
load_dotenv()

logger = logging.getLogger(__name__)

# LLM Configuration
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama")  # "ollama" or "huggingface"
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "gemma2:2b")
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
HUGGINGFACE_MODEL = os.getenv("HUGGINGFACE_MODEL", "meta-llama/Meta-Llama-3-8B-Instruct")

# Global state
app = FastAPI(title="RAG Service with LLM", version="1.0.0")

# Add CORS middleware for web UI
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for web UI
if os.path.exists("static"):
    app.mount("/static", StaticFiles(directory="static"), name="static")

indexer = VectorIndexer()
metrics = MetricsCollector()
crawled_pages = {}


# Pydantic models
class CrawlRequest(BaseModel):
    start_url: HttpUrl
    max_pages: int = Field(50, ge=1, le=500)
    max_depth: int = Field(3, ge=1, le=10)
    crawl_delay_ms: int = Field(500, ge=100, le=5000)

class CrawlResponse(BaseModel):
    page_count: int
    skipped_count: int
    urls: List[str]

class IndexRequest(BaseModel):
    chunk_size: int = Field(800, ge=100, le=2000)
    chunk_overlap: int = Field(200, ge=0, le=500)
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

class IndexResponse(BaseModel):
    vector_count: int
    errors: List[str]

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=500)
    top_k: int = Field(5, ge=1, le=20)

class AskResponse(BaseModel):
    answer: str
    sources: List[Dict[str, str]]
    timings: Dict[str, float]


async def call_ollama_llm(prompt: str, model: str = OLLAMA_MODEL) -> str:
    """Call Ollama API for LLM inference."""
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            return result.get("response", "").strip()
    except Exception as e:
        logger.error(f"Ollama API error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


async def call_huggingface_llm(prompt: str, model: str = HUGGINGFACE_MODEL) -> str:
    """Call HuggingFace Inference API for LLM inference."""
    if not HUGGINGFACE_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="HUGGINGFACE_API_KEY not set. Get one at https://huggingface.co/settings/tokens"
        )
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(
                f"https://api-inference.huggingface.co/models/{model}",
                headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                json={
                    "inputs": prompt,
                    "parameters": {
                        "max_new_tokens": 500,
                        "temperature": 0.7,
                        "top_p": 0.9,
                    }
                }
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                return result[0].get("generated_text", "").strip()
            elif isinstance(result, dict):
                return result.get("generated_text", "").strip()
            else:
                return str(result)
                
    except Exception as e:
        logger.error(f"HuggingFace API error: {e}")
        raise HTTPException(status_code=500, detail=f"LLM error: {str(e)}")


async def generate_llm_answer(question: str, chunks: List[Dict[str, Any]]) -> str:
    """Generate answer using real LLM with retrieved context."""
    
    if not chunks:
        return "NOT_ENOUGH_INFORMATION: No relevant content found in the crawled pages to answer your question."

    # Check if we have high-quality matches
    high_quality = [c for c in chunks if c.get('score', 0) > 0.5]

    if not high_quality:
        return f"NOT_ENOUGH_INFORMATION: The available content has low relevance (best score: {chunks[0].get('score', 0):.2f}) to answer: '{question}'. The crawled pages may not contain sufficient information on this topic."

    # Build context from retrieved chunks
    context_parts = []
    for i, chunk in enumerate(high_quality[:5], 1):
        context_parts.append(f"[Source {i} - {chunk['url']}]\n{chunk['text']}\n")

    context = "\n".join(context_parts)
    
    # Create prompt for LLM
    prompt = f"""You are a helpful assistant that answers questions based on provided context. 
Use ONLY the information from the context below to answer the question. 
If the context doesn't contain enough information, say "NOT_ENOUGH_INFORMATION" followed by a brief explanation.

Context:
{context}

Question: {question}

Answer (be concise and factual, cite sources when possible):"""

    # Call LLM based on provider
    if LLM_PROVIDER.lower() == "ollama":
        answer = await call_ollama_llm(prompt)
    elif LLM_PROVIDER.lower() == "huggingface":
        answer = await call_huggingface_llm(prompt)
    else:
        raise HTTPException(
            status_code=500, 
            detail=f"Unknown LLM provider: {LLM_PROVIDER}. Use 'ollama' or 'huggingface'"
        )
    
    # After obtaining `answer` from the LLM, ensure low_relevance_warning is defined
    # and prepend it safely to the returned answer.
    # If `chunks` is available in this scope (passed into the function), use their scores;
    # otherwise fall back to safe defaults.
    scores = []
    try:
        scores = [c.get("score", 0.0) for c in chunks]  # chunks should be the retrieved candidates
    except Exception:
        scores = []

    best_score = max(scores) if scores else 0.0
    low_relevance_warning = ""
    if best_score < MIN_RELEVANCE_SCORE:
        low_relevance_warning = (
            f"LOW_RELEVANCE: The best retrieval score is {best_score:.3f}, "
            "below the relevance threshold. The answer may be unreliable.\n\n"
        )

    final_answer = low_relevance_warning + (answer or "")
    return final_answer


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "healthy", 
        "service": "RAG Service with LLM",
        "llm_provider": LLM_PROVIDER,
        "llm_model": OLLAMA_MODEL if LLM_PROVIDER == "ollama" else HUGGINGFACE_MODEL
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return metrics_handler()


@app.post("/crawl", response_model=CrawlResponse)
async def crawl_endpoint(request: CrawlRequest):
    """Crawl a website and store pages in memory."""
    global crawled_pages
    
    start_time = time.time()
    
    try:
        crawled_pages.clear()
        
        logger.info(f"Starting crawl of {request.start_url}")
        result = await crawl_website(
            start_url=str(request.start_url),
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            crawl_delay_ms=request.crawl_delay_ms
        )
        
        crawled_pages = result.get('pages', {})
        
        elapsed = time.time() - start_time
        metrics.record_operation('crawl', elapsed, success=True)
        return CrawlResponse(
            page_count=result['page_count'],
            skipped_count=result['skipped_count'],
            urls=result['urls']
        )
        
    except Exception as e:
        logger.error(f"Crawl error: {e}")
        metrics.record_operation('crawl', time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/index", response_model=IndexResponse)
async def index_endpoint(request: IndexRequest):
    """Index the crawled pages."""
    global crawled_pages, indexer
    
    start_time = time.time()
    
    if not crawled_pages:
        raise HTTPException(status_code=400, detail="No pages to index. Run /crawl first.")
    
    try:
        # Create a fresh indexer for this indexing operation (uses embedding model)
        indexer = VectorIndexer(embedding_model=request.embedding_model)

        result = await index_crawled_pages(
            pages=crawled_pages,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            embedding_model=request.embedding_model
        )

        # Reload the global in-memory indexer so that subsequent /ask calls
        # use the newly created index immediately instead of stale data.
        try:
            indexer = VectorIndexer(embedding_model=request.embedding_model)
            logger.info("Reloaded in-memory indexer after indexing")
        except Exception as e:
            logger.warning(f"Failed to reload indexer after indexing: {e}")
        
        elapsed = time.time() - start_time
        metrics.record_operation('index', elapsed, success=True)
        
        return IndexResponse(
            vector_count=result['vector_count'],
            errors=result.get('errors', [])
        )
        
    except Exception as e:
        logger.error(f"Index error: {e}")
        metrics.record_operation('index', time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
async def ask_endpoint(request: AskRequest):
    """Answer a question using the indexed content with real LLM."""
    global indexer
    
    start_time = time.time()
    timings = {}
    
    try:
        # Perform retrieval
        retrieval_start = time.time()
        search_results = indexer.search(
            query=request.question,
            top_k=request.top_k,
            score_threshold=MIN_RELEVANCE_SCORE
        )
        timings['retrieval_ms'] = (time.time() - retrieval_start) * 1000
        
        # Generate answer using real LLM
        generation_start = time.time()
        answer = await generate_llm_answer(request.question, search_results)
        timings['generation_ms'] = (time.time() - generation_start) * 1000
        
        # Build sources list
        sources = []
        if search_results and "NOT_ENOUGH_INFORMATION" not in answer:
            for result in search_results[:3]:
                sources.append({
                    "url": result['url'],
                    "snippet": result['text'][:200] + "...",
                    "score": f"{result['score']:.3f}"
                })
        
        timings['total_ms'] = (time.time() - start_time) * 1000
        
        # Record metrics
        metrics.record_operation('ask', time.time() - start_time, success=True)
        metrics.record_latency('retrieval', timings['retrieval_ms'])
        metrics.record_latency('generation', timings['generation_ms'])
        
        return AskResponse(
            answer=answer,
            sources=sources,
            timings=timings
        )
        
    except Exception as e:
        logger.error(f"Ask error: {e}")
        metrics.record_operation('ask', time.time() - start_time, success=False)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    return {
        "crawled_pages": len(crawled_pages),
        "index_stats": indexer.get_stats(),
        "llm_provider": LLM_PROVIDER,
        "llm_model": OLLAMA_MODEL if LLM_PROVIDER == "ollama" else HUGGINGFACE_MODEL,
        "device": "CPU/GPU (via LLM API)"
    }



@app.post("/generate-image")
async def generate_image_endpoint(prompt: dict):
    """Generate an image from a text prompt using an Ollama image model.
    Expected payload: {"prompt": "a sunny beach"}
    Returns base64-encoded PNG image.
    """
    model = os.getenv("IMAGE_MODEL", "sdxl")
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{OLLAMA_BASE_URL}/api/generate",
                json={"model": model, "prompt": prompt.get("prompt", ""), "stream": False}
            )
            resp.raise_for_status()
            data = resp.json()
            # Assume the API returns base64 image in 'image' field
            img_b64 = data.get("image")
            return {"image": img_b64}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Image generation error: {str(e)}")

@app.post("/chat-persona")
async def chat_persona_endpoint(request: dict):
    """Chat with a selected persona. Payload: {"persona": "friendly", "message": "Hello"}
    The persona influences the system prompt sent to the LLM.
    """
    persona = request.get("persona", "default")
    message = request.get("message", "")
    system_prompt = f"You are a {persona} assistant. Respond accordingly."
    prompt = f"{system_prompt}\nUser: {message}\nAssistant:"
    answer = await call_ollama_llm(prompt) if LLM_PROVIDER.lower() == "ollama" else await call_huggingface_llm(prompt)
    return {"answer": answer}

@app.post("/voice-query")
async def voice_query_endpoint(file: dict):
    """Accept base64-encoded audio, transcribe via Whisper (if available), then answer.
    Payload: {"audio": "<base64>"}
    """
    audio_b64 = file.get("audio")
    if not audio_b64:
        raise HTTPException(status_code=400, detail="No audio provided")
    # Decode and send to a transcription service (placeholder)
    # For now, just raise not implemented
    raise HTTPException(status_code=501, detail="Voice transcription not implemented yet")

@app.get("/playground")
async def playground_endpoint():
    """Return a minimal HTML playground for experimenting with prompts.
    In a real app this would serve a static page; here we return a simple string.
    """
    html = """<html><body><h2>Prompt Playground</h2><textarea id='prompt' rows='4' cols='50'></textarea><button onclick='fetchAnswer()'>Ask</button><pre id='answer'></pre><script>async function fetchAnswer(){const p=document.getElementById('prompt').value;const res=await fetch('/ask',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({question:p,top_k:5}});const data=await res.json();document.getElementById('answer').textContent=JSON.stringify(data,null,2);}</script></body></html>"""
    return HTMLResponse(content=html, status_code=200)


class CalcRequest(BaseModel):
    expression: str

@app.post("/calc")
async def calc_endpoint(request: CalcRequest):
    """Evaluate a simple arithmetic expression safely.
    Supports +, -, *, /, parentheses, and decimal numbers.
    """
    expr = request.expression.strip()
    # Allow only numbers, operators, parentheses, spaces, and decimal points
    if not re.fullmatch(r"[0-9\.\+\-\*/\(\) ]+", expr):
        raise HTTPException(status_code=400, detail="Invalid characters in expression.")
    try:
        # Use eval with empty globals for safety
        result = eval(expr, {"__builtins__": {}}, {})
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error evaluating expression: {e}")
    return {"result": result}

    """Run the FastAPI server."""
    logger.info(f"Starting server with {LLM_PROVIDER} LLM provider")
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()

