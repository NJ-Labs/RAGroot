"""
Utils package for the RAG application.

Modules:
- config: Application configuration
- document_processor: Docling-based multi-format document processing
- indexer: Vector indexing with FAISS
- retriever: RAG pipeline with LLM
- agentic_rag: Agentic RAG with Hebrew/English agents (LangGraph + DSPy)
- image_gen: Image generation
- latex_utils: LaTeX processing utilities
- encoders: Custom encoders for embedding models
"""

from .config import config
from .document_processor import DoclingProcessor, ProcessedDocument, DocumentChunk
from .indexer import VectorIndexer
from .retriever import RAGPipeline
from .agentic_rag import AgenticRAGPipeline, create_agentic_rag_pipeline, LanguageDetector

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
]