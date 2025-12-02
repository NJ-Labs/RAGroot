import os
import sys
import json
import hashlib
import time
import argparse
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn

from utils.indexer import VectorIndexer
from utils.retriever import RAGPipeline
from utils.image_gen import ImageGenerator


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
indexer: Optional[VectorIndexer] = None
rag_pipeline: Optional[RAGPipeline] = None
image_generator: Optional[ImageGenerator] = None
current_dataset_hash: Optional[str] = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    # Startup
    logger.info("Starting up application...")
    try:
        initialize_system()
    except Exception as e:
        logger.error(f"Failed to initialize system: {e}")
        raise
    
    yield
    
    # Shutdown (cleanup if needed)
    logger.info("Shutting down application...")

# Initialize FastAPI with lifespan
app = FastAPI(title="GenAI RAG Application", lifespan=lifespan)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for serving generated images
Path("static/generated_images").mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Request/Response Models
class QueryRequest(BaseModel):
    query: str
    generate_image: bool = False
    top_k: int = 5
    force_cpu: bool = False

class Citation(BaseModel):
    doc_id: str
    title: str
    authors: str

class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    retrieved_context: List[str]
    image_url: Optional[str] = None
    performance_metrics: Optional[Dict] = None

def compute_file_hash(filepath: str) -> str:
    """Compute SHA256 hash of file to detect changes."""
    sha256_hash = hashlib.sha256()
    with open(filepath, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def initialize_system():
    """Initialize the system components."""
    global indexer, rag_pipeline, image_generator
    
    # Import config after it's initialized
    from utils.config import config
    
    # Get paths from utils.config
    data_path = config.DATA_PATH
    index_dir = config.INDEX_DIR
    
    logger.info(f"Initializing system with DATA_PATH: {data_path}")
    
    if not os.path.exists(data_path):
        error_msg = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║ ERROR: Dataset not found at {data_path}
║
║ Please ensure:
║ 1. The file exists and the path is correct
║ 2. You have read permissions for the file
║ 3. The file is a valid .jsonl file with required fields (id, title, abstract)
║
║ For Docker deployments, ensure you mount the dataset:
║   docker run -v $(pwd)/your-dataset.jsonl:/data/arxiv_2.9k.jsonl:ro ...
║
║ For local development:
║   Place your dataset in: data/arxiv_2.9k.jsonl
║   Or set DATA_PATH environment variable
╚══════════════════════════════════════════════════════════════════════════════╝
        """
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
    
    # Check if dataset or embedding model has changed
    new_hash = compute_file_hash(data_path)
    hash_file = Path(index_dir) / "dataset_hash.txt"
    model_file = Path(index_dir) / "embedding_model.txt"
    
    needs_reindex = True
    if hash_file.exists() and model_file.exists():
        with open(hash_file, 'r') as f:
            old_hash = f.read().strip()
        with open(model_file, 'r') as f:
            old_model = f.read().strip()
        
        if old_hash == new_hash and old_model == config.EMBEDDING_MODEL:
            needs_reindex = False
            logger.info(f"Dataset and embedding model unchanged, loading existing index")
        elif old_hash != new_hash:
            logger.info(f"Dataset changed, rebuilding index...")
        elif old_model != config.EMBEDDING_MODEL:
            logger.info(f"Embedding model changed from '{old_model}' to '{config.EMBEDDING_MODEL}', rebuilding index...")
    
    # Initialize components
    indexer = VectorIndexer(index_dir=index_dir)
    
    if needs_reindex:
        logger.info("Building new index...")
        indexer.build_index(data_path)
        # Save hash and model name
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        with open(hash_file, 'w') as f:
            f.write(new_hash)
        with open(model_file, 'w') as f:
            f.write(config.EMBEDDING_MODEL)
        logger.info("Index built successfully")
    else:
        logger.info("Loading existing index...")
        indexer.load_index()
        logger.info("Index loaded successfully")
    
    rag_pipeline = RAGPipeline(indexer)
    image_generator = ImageGenerator()
    current_dataset_hash = new_hash
    
    logger.info("System initialized successfully")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main web UI."""
    from fastapi import Response
    with open("static/index.html", "r", encoding="utf-8") as f:
        content = f.read()
    return Response(
        content=content,
        media_type="text/html",
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0"
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "indexed_documents": len(indexer.documents) if indexer else 0,
        "dataset_hash": current_dataset_hash
    }

@app.post("/answer", response_model=QueryResponse)
async def answer_query(request: QueryRequest):
    """Answer a query using RAG."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        start_time = time.time()
        logger.info(f"Processing query: {request.query}")
        
        # Retrieve and generate answer with timing
        retrieval_start = time.time()
        result = rag_pipeline.answer_query(
            query=request.query,
            top_k=request.top_k
        )
        retrieval_time = time.time() - retrieval_start
        
        total_time = time.time() - start_time
        
        # Build performance metrics (without image generation time)
        performance_metrics = {
            "total_time_ms": round(total_time * 1000, 2),
            "retrieval_time_ms": round(retrieval_time * 1000, 2),
            "documents_retrieved": len(result["citations"]),
            "answer_length_words": len(result["answer"].split())
        }
        
        logger.info(f"Query completed in {total_time:.2f}s (retrieval: {retrieval_time:.2f}s)")
        
        # Return answer immediately - image will be generated separately
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            retrieved_context=result["retrieved_context"],
            image_url=None,  # Image URL will be fetched separately
            performance_metrics=performance_metrics
        )
    
    except Exception as e:
        logger.error(f"Error processing query: {e}")
        raise HTTPException(status_code=500, detail=str(e))

class ImageRequest(BaseModel):
    prompt: str
    query: Optional[str] = None
    image_provider: Optional[str] = None
    api_key: Optional[str] = None
    force_cpu: bool = False

@app.post("/generate-image")
async def generate_image_endpoint(request: ImageRequest):
    """Generate an image independently from the query."""
    if not image_generator:
        raise HTTPException(status_code=503, detail="Image generator not initialized")
    
    try:
        import torch
        from utils.config import config
        
        # Use runtime override for image provider if specified
        provider = request.image_provider or config.IMAGE_API_PROVIDER
        api_key = request.api_key or config.IMAGE_API_KEY
        force_cpu = request.force_cpu
        
        # Check if CUDA is available for local generation
        if provider == "local" and (not torch.cuda.is_available() or force_cpu):
            logger.warning("Image generation skipped: CUDA not available or Force CPU is enabled. Image generation requires GPU for local provider.")
            return {"image_url": None, "error": "GPU required for local image generation. Try OpenAI or Pollinations provider instead."}
        
        # Validate API key for OpenAI
        if provider == "openai" and not api_key:
            logger.warning("Image generation skipped: OpenAI API key not provided")
            return {"image_url": None, "error": "OpenAI API key required for OpenAI provider"}
        
        logger.info(f"Generating image with provider: {provider}")
        start_time = time.time()
        
        image_url = await image_generator.generate_image(
            prompt=request.prompt,
            query=request.query,
            provider_override=provider,
            api_key_override=api_key
        )
        
        image_time = time.time() - start_time
        logger.info(f"Image generated in {image_time:.2f}s")
        
        return {
            "image_url": image_url,
            "generation_time_ms": round(image_time * 1000, 2)
        }
    
    except Exception as e:
        logger.error(f"Error generating image: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stream")
async def stream_answer(request: QueryRequest):
    """Stream answer generation (bonus endpoint)."""
    if not rag_pipeline:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    logger.info(f"Starting stream for query: {request.query}")
    
    async def generate():
        try:
            chunk_count = 0
            async for chunk in rag_pipeline.stream_answer(
                query=request.query,
                top_k=request.top_k
            ):
                chunk_count += 1
                data = f"data: {json.dumps(chunk)}\n\n"
                logger.debug(f"Streaming chunk #{chunk_count}: {chunk.get('type', 'unknown')}")
                yield data
            logger.info(f"Stream completed with {chunk_count} chunks")
        except Exception as e:
            logger.error(f"Streaming error: {e}", exc_info=True)
            yield f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
    
    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
            "Transfer-Encoding": "chunked"
        }
    )

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    if not indexer:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "total_documents": len(indexer.documents),
        "index_size": indexer.get_index_size(),
        "embedding_dimension": indexer.embedding_dim
    }

def run_cli():
    """Command-line interface for the GenAI RAG application."""
    parser = argparse.ArgumentParser(
        description="GenAI RAG Application - Academic Research Assistant",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Index a dataset
  python main.py index --data data/arxiv_2.9k.jsonl --output index/
  
  # Query the system
  python main.py query "What are recent advances in transformers?" --top-k 5
  
  # Query with streaming (live output)
  python main.py query "What are recent advances in transformers?" --stream
  
  # Start the web server
  python main.py serve --port 8080 --host 0.0.0.0
  
  # Evaluate the system
  python main.py evaluate --url http://localhost:8080
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Index command
    index_parser = subparsers.add_parser('index', help='Build vector index from dataset')
    index_parser.add_argument('--data', required=True, help='Path to JSONL dataset')
    index_parser.add_argument('--output', default='index/', help='Output directory for index')
    
    # Query command
    query_parser = subparsers.add_parser('query', help='Query the RAG system')
    query_parser.add_argument('query', help='Query text')
    query_parser.add_argument('--top-k', type=int, default=5, help='Number of documents to retrieve')
    query_parser.add_argument('--index-dir', default='index/', help='Index directory')
    query_parser.add_argument('--model', default='models/llama-model.gguf', help='LLM model path')
    query_parser.add_argument('--stream', action='store_true', help='Stream the answer live as it is generated')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start the web server')
    serve_parser.add_argument('--port', type=int, default=8080, help='Port to listen on')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Host to bind to')
    serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Evaluate command
    evaluate_parser = subparsers.add_parser('evaluate', help='Run evaluation tests')
    evaluate_parser.add_argument('--url', default='http://localhost:8080', help='API base URL')
    
    args = parser.parse_args()
    
    if args.command == 'index':
        print("\n" + "="*70)
        print("📦 Building Vector Index")
        print("="*70)
        print(f"Data source:      {args.data}")
        print(f"Output directory: {args.output}")
        print("="*70 + "\n")
        
        indexer = VectorIndexer(index_dir=args.output)
        indexer.build_index(args.data)
        
        print("\n" + "="*70)
        print("✅ Index Built Successfully!")
        print("="*70)
        print(f"Total documents: {len(indexer.documents)}")
        print(f"Index size: {indexer.get_index_size()}")
        print("="*70 + "\n")
    
    elif args.command == 'query':
        print("\n" + "="*70)
        print("🔍 Querying RAG System")
        print("="*70)
        print(f"Query: {args.query}")
        print(f"Top-K: {args.top_k}")
        print(f"Mode: {'Streaming' if args.stream else 'Standard'}")
        print("="*70 + "\n")
        
        # Load indexer
        indexer = VectorIndexer(index_dir=args.index_dir)
        indexer.load_index()
        
        # Initialize pipeline
        os.environ['MODEL_PATH'] = args.model
        rag_pipeline = RAGPipeline(indexer)
        
        if args.stream:
            # Streaming mode
            import asyncio
            
            async def stream_query():
                full_answer = ""
                citations = []
                context = []
                
                print("📡 Streaming response...\n")
                
                async for chunk in rag_pipeline.stream_answer(args.query, top_k=args.top_k):
                    chunk_type = chunk.get('type')
                    content = chunk.get('content')
                    
                    if chunk_type == 'citations':
                        citations = content
                        print("\n📚 Citations received")
                    elif chunk_type == 'context':
                        context = content
                        print("📄 Context retrieved\n")
                        print("📝 Answer:")
                        print("-" * 70)
                    elif chunk_type == 'token':
                        print(content, end='', flush=True)
                        full_answer += content
                    elif chunk_type == 'complete':
                        print("\n" + "-" * 70)
                        print("\n✅ Streaming complete!\n")
                    elif chunk_type == 'error':
                        print(f"\n\n❌ Error: {content}")
                        return
                
                # Display citations
                if citations:
                    print("\n📚 Citations:")
                    for i, citation in enumerate(citations, 1):
                        print(f"  [{i}] {citation['doc_id']}: {citation['title']}")
                
                # Display context preview
                if context:
                    print("\n📄 Retrieved Context (first 200 chars each):")
                    for i, ctx in enumerate(context, 1):
                        print(f"  [{i}] {ctx[:200]}...")
                print()
            
            asyncio.run(stream_query())
        else:
            # Standard mode (non-streaming)
            result = rag_pipeline.answer_query(args.query, top_k=args.top_k)
            
            print("\n📝 Answer:")
            print("-" * 70)
            print(result['answer'])
            print("-" * 70)
            
            print("\n📚 Citations:")
            for i, citation in enumerate(result['citations'], 1):
                print(f"  [{i}] {citation['doc_id']}: {citation['title']}")
            
            print("\n📄 Retrieved Context (first 200 chars each):")
            for i, context in enumerate(result['retrieved_context'], 1):
                print(f"  [{i}] {context[:200]}...")
            print()
    
    elif args.command == 'serve':
        print("\n" + "="*70)
        print("🚀 Starting GenAI RAG Server")
        print("="*70)
        print(f"Host: {args.host}")
        print(f"Port: {args.port}")
        print(f"URL:  http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}")
        print("="*70 + "\n")
        
        uvicorn.run(
            "main:app",
            host=args.host,
            port=args.port,
            log_level="info",
            reload=args.reload
        )
    
    elif args.command == 'evaluate':
        print("\n" + "="*70)
        print("📊 Running Evaluation")
        print("="*70)
        print(f"API URL: {args.url}")
        print("="*70 + "\n")
        
        # Import and run evaluation
        sys.path.insert(0, 'tests')
        from tests.evaluate_rag import evaluate_system
        evaluate_system(args.url)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    # Check if CLI arguments provided
    if len(sys.argv) > 1:
        run_cli()
    else:
        # Default: start web server
        from utils.config import config

        uvicorn.run(
            "main:app",
            host=config.HOST,
            port=config.PORT,
            log_level=config.LOG_LEVEL
        )
