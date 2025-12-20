import os
import sys
import json
import time
import argparse
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
from contextlib import asynccontextmanager
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

from utils.indexer import VectorIndexer
from utils.retriever import RAGPipeline
from utils.agentic_rag import AgenticRAGPipeline, create_agentic_rag_pipeline
from utils.image_gen import ImageGenerator
from utils.document_processor import DoclingProcessor


# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global state
indexer: Optional[VectorIndexer] = None
rag_pipeline: Optional[RAGPipeline] = None
agentic_rag_pipeline: Optional[AgenticRAGPipeline] = None
image_generator: Optional[ImageGenerator] = None
doc_processor: Optional[DoclingProcessor] = None
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
app = FastAPI(
    title="GenAI RAG Application",
    description="Multi-format document RAG with Hebrew/English support powered by Docling",
    version="2.0.0",
    lifespan=lifespan
)

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
    detected_language: Optional[str] = None  # "hebrew" or "english"
    agent_used: Optional[str] = None  # "Hebrew Agent" or "English Agent"


# =============================================================================
# ROOT (KNOWLEDGE BASE) MODELS AND STORAGE
# =============================================================================

class RootCreate(BaseModel):
    """Request model for creating a new Root."""
    name: Optional[str] = None  # If not provided, will be auto-generated
    description: Optional[str] = None

class RootDocument(BaseModel):
    """Document within a Root."""
    document_id: str
    filename: str
    file_type: str
    language: Optional[str] = None
    added_at: str

class Root(BaseModel):
    """A Knowledge Base (Root) containing documents."""
    id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str  # 4-word summary title
    description: str
    icon_type: str = "docs"  # docs, research, ai, code, data
    documents: List[RootDocument] = []
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    is_featured: bool = False

class RootsStorage:
    """Manages persistent storage of Roots."""
    
    def __init__(self, storage_path: str = "index/roots.json"):
        self.storage_path = Path(storage_path)
        self.roots: Dict[str, Root] = {}
        self._load()
    
    def _load(self):
        """Load roots from disk."""
        if self.storage_path.exists():
            try:
                with open(self.storage_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    for root_data in data:
                        root = Root(**root_data)
                        self.roots[root.id] = root
                logger.info(f"Loaded {len(self.roots)} roots from storage")
            except Exception as e:
                logger.error(f"Error loading roots: {e}")
                self.roots = {}
        else:
            self.roots = {}
    
    def _save(self):
        """Save roots to disk."""
        try:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump([root.model_dump() for root in self.roots.values()], f, indent=2)
        except Exception as e:
            logger.error(f"Error saving roots: {e}")
    
    def create(self, root: Root) -> Root:
        """Create a new root."""
        self.roots[root.id] = root
        self._save()
        return root
    
    def get(self, root_id: str) -> Optional[Root]:
        """Get a root by ID."""
        return self.roots.get(root_id)
    
    def get_all(self) -> List[Root]:
        """Get all roots sorted by updated_at descending."""
        return sorted(self.roots.values(), key=lambda r: r.updated_at, reverse=True)
    
    def update(self, root_id: str, **kwargs) -> Optional[Root]:
        """Update a root."""
        if root_id not in self.roots:
            return None
        root = self.roots[root_id]
        for key, value in kwargs.items():
            if hasattr(root, key):
                setattr(root, key, value)
        root.updated_at = datetime.now().isoformat()
        self._save()
        return root
    
    def add_document(self, root_id: str, doc: RootDocument) -> Optional[Root]:
        """Add a document to a root."""
        if root_id not in self.roots:
            return None
        root = self.roots[root_id]
        # Check if document already exists
        if not any(d.document_id == doc.document_id for d in root.documents):
            root.documents.append(doc)
            root.updated_at = datetime.now().isoformat()
            self._save()
        return root
    
    def remove_document(self, root_id: str, document_id: str) -> Optional[Root]:
        """Remove a document from a root."""
        if root_id not in self.roots:
            return None
        root = self.roots[root_id]
        root.documents = [d for d in root.documents if d.document_id != document_id]
        root.updated_at = datetime.now().isoformat()
        self._save()
        return root
    
    def delete(self, root_id: str) -> bool:
        """Delete a root."""
        if root_id in self.roots:
            del self.roots[root_id]
            self._save()
            return True
        return False

# Global roots storage
roots_storage: Optional[RootsStorage] = None


def initialize_system():
    """Initialize the system components."""
    global indexer, rag_pipeline, agentic_rag_pipeline, image_generator, doc_processor
    
    # Import config after it's initialized
    from utils.config import config
    
    # Get paths from utils.config
    index_dir = config.INDEX_DIR
    uploads_dir = config.UPLOADS_DIR
    
    # Create uploads directory
    Path(uploads_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Uploads directory: {uploads_dir}")
    
    # Initialize document processor with smart chunking (Chroma Research 2024)
    # Reference: https://research.trychroma.com/evaluating-chunking
    doc_processor = DoclingProcessor(
        ocr_enabled=config.OCR_ENABLED,
        ocr_languages=config.OCR_LANGUAGES.split(','),
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        use_gpu=not config.FORCE_CPU,
        chunking_strategy=getattr(config, 'CHUNKING_STRATEGY', 'recursive'),
        use_smart_chunking=getattr(config, 'USE_SMART_CHUNKING', True)
    )
    
    # Initialize indexer
    indexer = VectorIndexer(index_dir=index_dir)
    
    # Check for existing index (previously indexed documents)
    index_path = Path(index_dir) / "faiss.index"
    
    if index_path.exists():
        # Load existing index from previously uploaded documents
        logger.info("Loading existing index from previously uploaded documents...")
        try:
            indexer.load_index()
            logger.info(f"Index loaded: {indexer.get_index_size()}")
        except Exception as e:
            logger.error(f"Failed to load index: {e}")
            logger.info("Starting with empty index - upload documents to populate")
    else:
        logger.info("No existing index found - upload documents to populate the index")
    
    # Initialize legacy RAG pipeline (for backward compatibility)
    rag_pipeline = RAGPipeline(indexer)
    
    # Initialize Agentic RAG Pipeline with Hebrew/English agents
    logger.info("Initializing Agentic RAG Pipeline with language-specific agents...")
    agentic_rag_pipeline = create_agentic_rag_pipeline(indexer)
    
    # Test LLM connectivity
    logger.info("Testing LLM endpoint connectivity...")
    try:
        llm_test = agentic_rag_pipeline.llm_backend.test_connection()
        
        if llm_test.get("success"):
            logger.info(f"✓ LLM connectivity test PASSED: {llm_test.get('message')}")
            if llm_test.get("test_response"):
                logger.info(f"  Test response: {llm_test.get('test_response')}")
            if llm_test.get("streaming_status") == "working":
                logger.info("  ✓ Streaming is working")
            elif llm_test.get("streaming_warning"):
                logger.warning(f"  ⚠ {llm_test.get('streaming_warning')}")
            if llm_test.get("available_models"):
                logger.info(f"  Available models: {llm_test.get('available_models')}")
        else:
            logger.error(f"✗ LLM connectivity test FAILED: {llm_test.get('message')}")
            if llm_test.get("error"):
                logger.error(f"  Error details: {llm_test.get('error')}")
            if llm_test.get("health_check"):
                logger.info(f"  Health check: {llm_test.get('health_check')}")
            logger.warning("  The system will start but queries may fail. Check your LLM configuration.")
            
    except Exception as e:
        logger.error(f"✗ LLM connectivity test failed with exception: {e}")
        logger.warning("  The system will start but queries may fail.")
    
    image_generator = ImageGenerator()
    
    # Initialize Roots storage
    global roots_storage
    roots_storage = RootsStorage()
    
    # Create default root from existing documents if no roots exist
    if len(roots_storage.roots) == 0 and indexer and len(indexer.documents) > 0:
        logger.info("Creating default Root from existing indexed documents...")
        default_root = Root(
            id="default",
            name="Initial Document Collection",
            description="Auto-created knowledge base from previously indexed documents",
            icon_type="docs",
            documents=[
                RootDocument(
                    document_id=doc.get('document_id', ''),
                    filename=doc.get('filename', 'Unknown'),
                    file_type=doc.get('file_type', 'unknown'),
                    language=doc.get('language'),
                    added_at=datetime.now().isoformat()
                )
                for doc in indexer.get_document_list()
            ],
            is_featured=True
        )
        roots_storage.create(default_root)
        logger.info(f"Created default Root with {len(default_root.documents)} documents")
    
    logger.info("System initialized successfully with Agentic RAG (Hebrew + English agents)")

@app.get("/", response_class=HTMLResponse)
async def read_dashboard():
    """Serve the knowledge dashboard landing page (main entry point)."""
    from fastapi import Response
    with open("static/dashboard.html", "r", encoding="utf-8") as f:
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

@app.get("/app", response_class=HTMLResponse)
@app.get("/app/{root_id}", response_class=HTMLResponse)
async def read_app(root_id: Optional[str] = None):
    """Serve the RAG chat application UI."""
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
        "indexed_chunks": len(indexer.chunks) if indexer else 0,
        "index_stats": indexer.get_index_size() if indexer else {}
    }


# =============================================================================
# ROOT (KNOWLEDGE BASE) MANAGEMENT ENDPOINTS
# =============================================================================

def generate_root_title(documents: List[dict]) -> str:
    """Generate a 4-word title from document names/content."""
    if not documents:
        return "New Knowledge Base"
    
    # Extract keywords from filenames
    words = []
    for doc in documents[:3]:  # Use first 3 docs
        filename = doc.get('filename', '')
        # Remove extension and split by common separators
        name = Path(filename).stem
        name_words = name.replace('_', ' ').replace('-', ' ').split()
        # Filter out numbers and very short words
        words.extend([w.title() for w in name_words if len(w) > 2 and not w.isdigit()])
    
    # Take unique words, limit to 4
    unique_words = list(dict.fromkeys(words))[:4]
    if len(unique_words) < 2:
        return "Document Collection"
    return ' '.join(unique_words)

def generate_root_description(documents: List[dict]) -> str:
    """Generate a description from documents."""
    if not documents:
        return "A new knowledge base for your documents"
    
    doc_types = set()
    languages = set()
    for doc in documents:
        doc_types.add(doc.get('file_type', 'document'))
        if doc.get('language'):
            languages.add(doc.get('language'))
    
    type_str = ', '.join(sorted(doc_types))
    lang_str = ' and '.join(sorted(languages)) if languages else 'multiple languages'
    
    return f"Collection of {len(documents)} {type_str} document(s) in {lang_str}"

def get_icon_type(documents: List[dict]) -> str:
    """Determine icon type based on document content."""
    if not documents:
        return "docs"
    
    filenames = ' '.join([d.get('filename', '').lower() for d in documents])
    
    if any(kw in filenames for kw in ['research', 'paper', 'arxiv', 'study']):
        return "research"
    elif any(kw in filenames for kw in ['ai', 'ml', 'model', 'neural', 'transformer']):
        return "ai"
    elif any(kw in filenames for kw in ['code', 'python', 'javascript', 'api']):
        return "code"
    elif any(kw in filenames for kw in ['data', 'dataset', 'csv', 'json']):
        return "data"
    return "docs"


@app.get("/api/roots")
async def list_roots():
    """List all knowledge bases (Roots)."""
    if not roots_storage:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    roots = roots_storage.get_all()
    return {
        "roots": [root.model_dump() for root in roots],
        "total": len(roots),
        "featured": [r.model_dump() for r in roots if r.is_featured]
    }


@app.get("/api/roots/{root_id}")
async def get_root(root_id: str):
    """Get a specific Root by ID."""
    if not roots_storage:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    root = roots_storage.get(root_id)
    if not root:
        raise HTTPException(status_code=404, detail=f"Root {root_id} not found")
    
    return root.model_dump()


@app.post("/api/roots")
async def create_root(request: RootCreate):
    """Create a new Root (Knowledge Base)."""
    if not roots_storage:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    root = Root(
        name=request.name or "New Knowledge Base",
        description=request.description or "A new knowledge base for your documents",
        icon_type="docs",
        documents=[],
        is_featured=False
    )
    
    created_root = roots_storage.create(root)
    return {
        "status": "success",
        "message": "Root created successfully",
        "root": created_root.model_dump()
    }


@app.post("/api/roots/{root_id}/documents")
async def add_document_to_root(
    root_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...)
):
    """Upload and add a document to a specific Root."""
    if not roots_storage or not indexer:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    root = roots_storage.get(root_id)
    if not root:
        raise HTTPException(status_code=404, detail=f"Root {root_id} not found")
    
    from utils.config import config
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    supported = config.SUPPORTED_FORMATS.split(',')
    
    if file_ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {supported}"
        )
    
    try:
        # Save uploaded file
        upload_path = Path(config.UPLOADS_DIR) / f"{int(time.time() * 1000)}_{file.filename}"
        with open(upload_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"File saved to {upload_path}")
        
        # Index the document
        processed_doc = indexer.index_document(str(upload_path))
        
        # Add to root
        root_doc = RootDocument(
            document_id=processed_doc.document_id,
            filename=processed_doc.filename,
            file_type=processed_doc.file_type,
            language=processed_doc.language,
            added_at=datetime.now().isoformat()
        )
        roots_storage.add_document(root_id, root_doc)
        
        # Update root metadata if needed
        updated_root = roots_storage.get(root_id)
        if updated_root and len(updated_root.documents) > 0:
            # Auto-update title and description if still default
            if updated_root.name in ["New Knowledge Base", "Untitled"]:
                doc_list = [{"filename": d.filename, "file_type": d.file_type, "language": d.language} 
                           for d in updated_root.documents]
                new_title = generate_root_title(doc_list)
                new_desc = generate_root_description(doc_list)
                new_icon = get_icon_type(doc_list)
                roots_storage.update(root_id, name=new_title, description=new_desc, icon_type=new_icon)
        
        return {
            "status": "success",
            "message": f"Document added to Root '{root.name}'",
            "document_id": processed_doc.document_id,
            "filename": processed_doc.filename,
            "chunks_created": len(processed_doc.chunks),
            "language": processed_doc.language,
            "root": roots_storage.get(root_id).model_dump()
        }
    except Exception as e:
        logger.error(f"Error adding document to root: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/api/roots/{root_id}/documents/{document_id}")
async def remove_document_from_root(root_id: str, document_id: str):
    """Remove a document from a Root (also removes from index)."""
    if not roots_storage or not indexer:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    root = roots_storage.get(root_id)
    if not root:
        raise HTTPException(status_code=404, detail=f"Root {root_id} not found")
    
    # Remove from index
    indexer.remove_document(document_id)
    
    # Remove from root
    roots_storage.remove_document(root_id, document_id)
    
    return {
        "status": "success",
        "message": f"Document {document_id} removed from Root",
        "root": roots_storage.get(root_id).model_dump()
    }


@app.put("/api/roots/{root_id}")
async def update_root(root_id: str, request: RootCreate):
    """Update a Root's name and description."""
    if not roots_storage:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    root = roots_storage.get(root_id)
    if not root:
        raise HTTPException(status_code=404, detail=f"Root {root_id} not found")
    
    updates = {}
    if request.name:
        updates['name'] = request.name
    if request.description:
        updates['description'] = request.description
    
    updated_root = roots_storage.update(root_id, **updates)
    
    return {
        "status": "success",
        "message": "Root updated successfully",
        "root": updated_root.model_dump()
    }


@app.put("/api/roots/{root_id}/featured")
async def toggle_root_featured(root_id: str, featured: bool = True):
    """Toggle featured status of a Root."""
    if not roots_storage:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    root = roots_storage.get(root_id)
    if not root:
        raise HTTPException(status_code=404, detail=f"Root {root_id} not found")
    
    updated_root = roots_storage.update(root_id, is_featured=featured)
    
    return {
        "status": "success",
        "root": updated_root.model_dump()
    }


@app.delete("/api/roots/{root_id}")
async def delete_root(root_id: str, delete_documents: bool = False):
    """Delete a Root. Optionally delete its documents from the index."""
    if not roots_storage:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    root = roots_storage.get(root_id)
    if not root:
        raise HTTPException(status_code=404, detail=f"Root {root_id} not found")
    
    # Optionally delete documents from index
    if delete_documents and indexer:
        for doc in root.documents:
            indexer.remove_document(doc.document_id)
    
    roots_storage.delete(root_id)
    
    return {
        "status": "success",
        "message": f"Root '{root.name}' deleted" + (" with all documents" if delete_documents else "")
    }


# =============================================================================
# DOCUMENT UPLOAD AND MANAGEMENT ENDPOINTS
# =============================================================================

@app.post("/documents/upload")
async def upload_document(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """
    Upload and index a document.
    
    Supports: PDF, DOCX, PPTX, XLSX, HTML, images (PNG, JPEG, TIFF), 
    Markdown, TXT, CSV, JSON, JSONL
    """
    if not indexer:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    from utils.config import config
    
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    supported = config.SUPPORTED_FORMATS.split(',')
    
    if file_ext not in supported:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: {file_ext}. Supported: {supported}"
        )
    
    try:
        # Save uploaded file
        uploads_dir = Path(config.UPLOADS_DIR)
        uploads_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate unique filename
        timestamp = int(time.time() * 1000)
        safe_filename = f"{timestamp}_{file.filename}"
        file_path = uploads_dir / safe_filename
        
        # Write file to disk
        with open(file_path, "wb") as f:
            content = await file.read()
            f.write(content)
        
        logger.info(f"Uploaded file saved: {file_path}")
        
        # Index the document
        processed_doc = indexer.index_document(str(file_path))
        
        return {
            "status": "success",
            "message": f"Document '{file.filename}' indexed successfully",
            "document_id": processed_doc.document_id,
            "filename": processed_doc.filename,
            "chunks_created": len(processed_doc.chunks),
            "language": processed_doc.language,
            "file_type": processed_doc.file_type
        }
        
    except Exception as e:
        logger.error(f"Error uploading document: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/documents/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(...)
):
    """Upload and index multiple documents at once."""
    if not indexer:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    from utils.config import config
    
    results = []
    supported = config.SUPPORTED_FORMATS.split(',')
    uploads_dir = Path(config.UPLOADS_DIR)
    uploads_dir.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    # First, save all files
    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        
        if file_ext not in supported:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": f"Unsupported format: {file_ext}"
            })
            continue
        
        try:
            timestamp = int(time.time() * 1000)
            safe_filename = f"{timestamp}_{file.filename}"
            file_path = uploads_dir / safe_filename
            
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            saved_paths.append((file.filename, str(file_path)))
            
        except Exception as e:
            results.append({
                "filename": file.filename,
                "status": "error",
                "error": str(e)
            })
    
    # Index all saved files
    if saved_paths:
        try:
            paths_only = [p for _, p in saved_paths]
            processed_docs = indexer.index_documents(paths_only)
            
            for (orig_name, _), doc in zip(saved_paths, processed_docs):
                results.append({
                    "filename": orig_name,
                    "status": "success",
                    "document_id": doc.document_id,
                    "chunks_created": len(doc.chunks),
                    "language": doc.language
                })
        except Exception as e:
            for orig_name, _ in saved_paths:
                results.append({
                    "filename": orig_name,
                    "status": "error",
                    "error": str(e)
                })
    
    return {
        "status": "completed",
        "total_files": len(files),
        "successful": sum(1 for r in results if r.get("status") == "success"),
        "failed": sum(1 for r in results if r.get("status") == "error"),
        "results": results
    }


class UrlRequest(BaseModel):
    url: str

@app.post("/documents/index-url")
async def index_url(request: UrlRequest):
    """Index a document from a URL."""
    if not indexer:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    try:
        logger.info(f"Indexing URL: {request.url}")
        processed_doc = indexer.index_document(request.url)
        
        return {
            "status": "success",
            "message": f"URL indexed successfully",
            "document_id": processed_doc.document_id,
            "filename": processed_doc.filename,
            "chunks_created": len(processed_doc.chunks),
            "language": processed_doc.language
        }
    except Exception as e:
        logger.error(f"Error indexing URL: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/documents")
async def list_documents():
    """List all indexed documents."""
    if not indexer:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    return {
        "documents": indexer.get_document_list(),
        "total_documents": len(indexer.documents),
        "total_chunks": len(indexer.chunks)
    }


@app.delete("/documents/{document_id}")
async def delete_document(document_id: str):
    """Remove a document from the index."""
    if not indexer:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    success = indexer.remove_document(document_id)
    
    if success:
        return {
            "status": "success",
            "message": f"Document {document_id} removed from index"
        }
    else:
        raise HTTPException(
            status_code=404,
            detail=f"Document {document_id} not found in index"
        )


@app.delete("/documents")
async def clear_all_documents():
    """Clear all documents from the index."""
    if not indexer:
        raise HTTPException(status_code=503, detail="System not initialized")
    
    indexer.clear_index()
    
    return {
        "status": "success",
        "message": "All documents cleared from index"
    }


@app.get("/documents/supported-formats")
async def get_supported_formats():
    """Get list of supported file formats with chunking info."""
    from utils.config import config
    from utils.document_processor import DoclingProcessor
    
    return {
        "supported_formats": config.SUPPORTED_FORMATS.split(','),
        "format_categories": {
            "documents": [".pdf", ".docx", ".doc"],
            "presentations": [".pptx", ".ppt"],
            "spreadsheets": [".xlsx", ".xls"],
            "text": [".txt", ".md", ".markdown", ".csv", ".json", ".jsonl"],
            "web": [".html", ".htm"],
            "images": [".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"]
        },
        "chunking": {
            "strategy": getattr(config, 'CHUNKING_STRATEGY', 'recursive'),
            "chunk_size": config.CHUNK_SIZE,
            "chunk_overlap": config.CHUNK_OVERLAP,
            "smart_chunking_enabled": getattr(config, 'USE_SMART_CHUNKING', True),
            "description": "Based on Chroma Research 2024 - RecursiveCharacterTextSplitter with 250 tokens/125 overlap = 96%+ recall"
        },
        "ocr_enabled": config.OCR_ENABLED,
        "ocr_languages": config.OCR_LANGUAGES.split(','),
        "hebrew_support": True,
        "english_support": True
    }


# =============================================================================
# RAG QUERY ENDPOINTS
# =============================================================================

@app.post("/answer", response_model=QueryResponse)
async def answer_query(request: QueryRequest):
    """Answer a query using Agentic RAG with language-specific agents."""
    if not agentic_rag_pipeline:
        raise HTTPException(status_code=503, detail="Agentic RAG system not initialized")
    
    try:
        start_time = time.time()
        logger.info(f"Processing query with Agentic RAG: {request.query}")
        
        # Use Agentic RAG Pipeline with automatic language detection and routing
        retrieval_start = time.time()
        result = agentic_rag_pipeline.answer_query(
            query=request.query,
            top_k=request.top_k
        )
        retrieval_time = time.time() - retrieval_start
        
        total_time = time.time() - start_time
        
        # Determine which agent was used
        detected_language = result.get("language", "english")
        agent_used = "Hebrew Agent" if detected_language == "hebrew" else "English Agent"
        
        # Build performance metrics
        performance_metrics = {
            "total_time_ms": round(total_time * 1000, 2),
            "retrieval_time_ms": round(retrieval_time * 1000, 2),
            "documents_retrieved": len(result["citations"]),
            "answer_length_words": len(result["answer"].split()),
            "agent_used": agent_used,
            "detected_language": detected_language
        }
        
        logger.info(f"Query completed in {total_time:.2f}s using {agent_used}")
        
        # Return answer with agent information
        return QueryResponse(
            answer=result["answer"],
            citations=result["citations"],
            retrieved_context=result["retrieved_context"],
            image_url=None,  # Image URL will be fetched separately
            performance_metrics=performance_metrics,
            detected_language=detected_language,
            agent_used=agent_used
        )
    
    except Exception as e:
        logger.error(f"Error processing query with Agentic RAG: {e}")
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
    """Stream answer generation using Agentic RAG with language-specific agents."""
    if not agentic_rag_pipeline:
        raise HTTPException(status_code=503, detail="Agentic RAG system not initialized")
    
    logger.info(f"Starting Agentic RAG stream for query: {request.query}")
    
    async def generate():
        try:
            chunk_count = 0
            async for chunk in agentic_rag_pipeline.stream_answer(
                query=request.query,
                top_k=request.top_k
            ):
                chunk_count += 1
                data = f"data: {json.dumps(chunk)}\n\n"
                logger.debug(f"Streaming chunk #{chunk_count}: {chunk.get('type', 'unknown')}")
                yield data
            logger.info(f"Agentic RAG stream completed with {chunk_count} chunks")
        except Exception as e:
            logger.error(f"Agentic RAG streaming error: {e}", exc_info=True)
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


@app.get("/health/llm")
async def test_llm_health():
    """Test LLM endpoint connectivity and return detailed status."""
    if not agentic_rag_pipeline:
        raise HTTPException(status_code=503, detail="Agentic RAG system not initialized")
    
    try:
        result = agentic_rag_pipeline.llm_backend.test_connection()
        
        # Add timestamp
        result["timestamp"] = time.strftime("%Y-%m-%d %H:%M:%S")
        
        if result.get("success"):
            return JSONResponse(content=result, status_code=200)
        else:
            return JSONResponse(content=result, status_code=503)
            
    except Exception as e:
        logger.error(f"LLM health check failed: {e}", exc_info=True)
        return JSONResponse(
            content={
                "success": False,
                "message": f"Health check failed: {str(e)}",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            status_code=503
        )


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
        print("🔍 Querying Agentic RAG System (Hebrew/English)")
        print("="*70)
        print(f"Query: {args.query}")
        print(f"Top-K: {args.top_k}")
        print(f"Mode: {'Streaming' if args.stream else 'Standard'}")
        print("="*70 + "\n")
        
        # Load indexer
        indexer = VectorIndexer(index_dir=args.index_dir)
        indexer.load_index()
        
        # Initialize Agentic RAG pipeline
        os.environ['MODEL_PATH'] = args.model
        agentic_pipeline = create_agentic_rag_pipeline(indexer)
        
        if args.stream:
            # Streaming mode
            import asyncio
            
            async def stream_query():
                full_answer = ""
                citations = []
                context = []
                
                print("📡 Streaming response with Agentic RAG...\n")
                
                async for chunk in agentic_pipeline.stream_answer(args.query, top_k=args.top_k):
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
            # Standard mode (non-streaming) with Agentic RAG
            result = agentic_pipeline.answer_query(args.query, top_k=args.top_k)
            
            # Show which agent was used
            detected_lang = result.get('language', 'english')
            agent_name = "Hebrew Agent 🇮🇱" if detected_lang == "hebrew" else "English Agent 🇬🇧"
            print(f"🤖 Agent Used: {agent_name}")
            print(f"🌐 Detected Language: {detected_lang.capitalize()}\n")
            
            print("📝 Answer:")
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
