"""
Utils package for the RAG application.

Modules:
- config: Application configuration
- document_processor: Docling-based multi-format document processing
- smart_chunker: Research-backed chunking strategies (Chroma Research 2024)
- document_agents: Format-specific processing agents (Word, PPT, Excel, HTML, MD)
- indexer: Vector indexing with FAISS
- retriever: RAG pipeline with LLM
- agentic_rag: Agentic RAG with Hebrew/English agents (LangGraph + DSPy)
- image_gen: Image generation
- latex_utils: LaTeX processing utilities
- encoders: Custom encoders for embedding models

Smart Chunking Strategies (https://research.trychroma.com/evaluating-chunking):
- RecursiveCharacterTextSplitter: 250 tokens / 125 overlap = 96%+ recall
- ClusterSemanticChunker: Best coherence (200-400 tokens, 0% overlap)
- Structure-aware chunking for formatted documents
- Page-level chunking for PDFs (NVIDIA benchmark winner)
"""

from .config import config
from .document_processor import DoclingProcessor, ProcessedDocument, DocumentChunk
from .indexer import VectorIndexer
from .retriever import RAGPipeline
from .agentic_rag import AgenticRAGPipeline, create_agentic_rag_pipeline, LanguageDetector

# Smart chunking (lazy imports to avoid circular dependencies)
try:
    from .smart_chunker import (
        SmartChunker,
        ChunkingStrategy,
        RecursiveCharacterTextSplitter,
        ClusterSemanticChunker,
        get_optimal_config,
    )
    from .document_agents import (
        DocumentAgentFactory,
        DocumentProcessingOrchestrator,
        WordDocumentAgent,
        PowerPointAgent,
        ExcelAgent,
        HTMLAgent,
        MarkdownAgent,
    )
    SMART_CHUNKING_AVAILABLE = True
except ImportError:
    SMART_CHUNKING_AVAILABLE = False

__all__ = [
    'config',
    'DoclingProcessor',
    'ProcessedDocument', 
    'DocumentChunk',
    'VectorIndexer',
    'RAGPipeline',
    'AgenticRAGPipeline',
    'create_agentic_rag_pipeline',
    'LanguageDetector',
    # Smart Chunking
    'SmartChunker',
    'ChunkingStrategy',
    'RecursiveCharacterTextSplitter',
    'ClusterSemanticChunker',
    'get_optimal_config',
    # Document Agents
    'DocumentAgentFactory',
    'DocumentProcessingOrchestrator',
    'WordDocumentAgent',
    'PowerPointAgent',
    'ExcelAgent',
    'HTMLAgent',
    'MarkdownAgent',
    'SMART_CHUNKING_AVAILABLE',
]