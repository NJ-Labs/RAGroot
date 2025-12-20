"""
Vector indexer for multi-format document processing.
Supports indexing documents processed by Docling with FAISS vector search.
Optimized for Hebrew and English bilingual content.
"""
import json
import pickle
import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Iterator
import logging
import hashlib

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from .document_processor import DoclingProcessor, ProcessedDocument, DocumentChunk

logger = logging.getLogger(__name__)


class VectorIndexer:
    """
    Handles document indexing and vector search for multi-format documents.
    
    Features:
    - Multi-format document processing via Docling
    - Bilingual support (Hebrew/English)
    - FAISS-based vector similarity search
    - Incremental indexing support
    """
    
    def __init__(self, index_dir: str = "index"):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure local-only mode for embedding models
        from .config import config
        
        # Set up cache directory
        cache_dir = Path(config.EMBEDDING_CACHE_DIR)
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure environment for offline operation
        if config.EMBEDDING_LOCAL_ONLY or config.SKIP_CHECK_ST_UPDATES:
            os.environ['HF_HUB_OFFLINE'] = '1'
            os.environ['TRANSFORMERS_OFFLINE'] = '1'
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
            logger.info(f"Local-only mode enabled for embeddings (cache: {cache_dir})")
        else:
            os.environ['SENTENCE_TRANSFORMERS_HOME'] = str(cache_dir)
            logger.info(f"Online mode - will download models if not cached (cache: {cache_dir})")
        
        # Determine device based on configuration
        if config.FORCE_CPU:
            device = 'cpu'
            logger.info("Force CPU mode enabled - using CPU for embeddings")
        else:
            import torch
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Auto-detected device: {device}")
        
        # Initialize embedding model (multilingual for Hebrew/English)
        logger.info(f"Loading embedding model: {config.EMBEDDING_MODEL}")
        
        try:
            self.encoder = SentenceTransformer(
                config.EMBEDDING_MODEL,
                device=device,
                cache_folder=str(cache_dir)
            )
            self.embedding_dim = self.encoder.get_sentence_embedding_dimension()
            logger.info(f"Loaded model from cache: {cache_dir}/{config.EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to load embedding model '{config.EMBEDDING_MODEL}'.")
            logger.error(f"Error: {e}")
            if config.EMBEDDING_LOCAL_ONLY:
                logger.error(f"""\n{'='*70}
❌ EMBEDDING MODEL NOT FOUND IN LOCAL CACHE
{'='*70}
Model: {config.EMBEDDING_MODEL}
Cache directory: {cache_dir}

To fix this:
1. Download the model using the download script:
   python tools/download_models.py

2. Or set EMBEDDING_LOCAL_ONLY=false in .env to allow downloads
{'='*70}""")
            raise
        
        logger.info(f"Embedding model loaded on {device} (dim={self.embedding_dim})")
        
        # Initialize document processor
        self.doc_processor = DoclingProcessor(
            ocr_enabled=config.OCR_ENABLED,
            ocr_languages=config.OCR_LANGUAGES.split(','),
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            use_gpu=not config.FORCE_CPU
        )
        
        self.index = None
        self.documents: List[Dict] = []  # Document metadata
        self.chunks: List[DocumentChunk] = []  # All chunks
        self.embeddings = None
        self.device = device
        
        # Document tracking for incremental updates
        self.document_hashes: Dict[str, str] = {}
    
    def _get_chunk_text_for_embedding(self, chunk: DocumentChunk, is_query: bool = False) -> str:
        """
        Prepare chunk text for embedding with context.
        
        For E5 models (multilingual-e5-*), adds required prefixes:
        - "query: " for search queries
        - "passage: " for document passages
        
        Args:
            chunk: The document chunk
            is_query: Whether this is a query (True) or passage (False)
        """
        from .config import config
        
        parts = []
        
        # Add section title if available
        if chunk.section_title:
            parts.append(f"Section: {chunk.section_title}")
        
        # Add the main content
        parts.append(chunk.content)
        
        text = "\n".join(parts)
        
        # Add E5 prefix if enabled (required for multilingual-e5 models)
        if config.USE_E5_PREFIX:
            prefix = "query: " if is_query else "passage: "
            text = prefix + text
        
        return text
    
    def build_index_from_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        file_filter: Optional[List[str]] = None
    ):
        """
        Build vector index from all documents in a directory.
        
        Args:
            directory_path: Path to directory containing documents
            recursive: Whether to process subdirectories
            file_filter: Optional list of extensions to filter
        """
        logger.info(f"Building index from directory: {directory_path}")
        
        all_chunks = []
        all_documents = []
        
        # Process all documents in directory
        for processed_doc in tqdm(
            self.doc_processor.process_directory(directory_path, recursive, file_filter),
            desc="Processing documents"
        ):
            all_documents.append(processed_doc.to_dict())
            all_chunks.extend(processed_doc.chunks)
            self.document_hashes[processed_doc.document_id] = hashlib.sha256(
                processed_doc.raw_text.encode()
            ).hexdigest()
        
        self._build_index_from_chunks(all_chunks, all_documents)
    
    def build_index_from_jsonl(self, data_path: str):
        """
        Build vector index from JSONL dataset (legacy format support).
        
        Args:
            data_path: Path to JSONL file
        """
        logger.info(f"Building index from JSONL: {data_path}")
        
        all_chunks = []
        all_documents = []
        
        # Count total lines for progress bar
        with open(data_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)
        
        # Process JSONL file
        with tqdm(total=total_lines, desc="Processing JSONL", unit="docs") as pbar:
            for processed_doc in self.doc_processor.process_jsonl(data_path):
                all_documents.append(processed_doc.to_dict())
                all_chunks.extend(processed_doc.chunks)
                pbar.update(1)
        
        self._build_index_from_chunks(all_chunks, all_documents)
    
    def build_index(self, data_path: str):
        """
        Build vector index from data source (auto-detect type).
        
        Args:
            data_path: Path to data source (JSONL file or directory)
        """
        path = Path(data_path)
        
        if path.is_dir():
            self.build_index_from_directory(path)
        elif path.suffix.lower() in ['.jsonl', '.ndjson']:
            self.build_index_from_jsonl(data_path)
        elif path.is_file():
            # Single file
            self.index_document(data_path)
        else:
            raise ValueError(f"Invalid data path: {data_path}")
    
    def _build_index_from_chunks(self, chunks: List[DocumentChunk], documents: List[Dict]):
        """
        Internal method to build FAISS index from chunks.
        
        Args:
            chunks: List of document chunks
            documents: List of document metadata dicts
        """
        if not chunks:
            logger.warning("No chunks to index!")
            return
        
        self.chunks = chunks
        self.documents = documents
        
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        
        from .config import config
        batch_size = config.EMBEDDING_BATCH_SIZE
        
        # Prepare texts for embedding (passages, not queries)
        texts = [self._get_chunk_text_for_embedding(chunk, is_query=False) for chunk in chunks]
        
        # Generate embeddings in batches
        all_embeddings = []
        with tqdm(total=len(texts), desc="Encoding embeddings", unit="chunks") as pbar:
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self.encoder.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True  # Normalize for cosine similarity
                )
                all_embeddings.append(batch_embeddings)
                pbar.update(len(batch))
        
        self.embeddings = np.vstack(all_embeddings).astype('float32')
        logger.info(f"Generated embeddings: {self.embeddings.shape}")
        
        # Build FAISS index with Inner Product (cosine similarity on normalized vectors)
        logger.info("Building FAISS index with cosine similarity (IndexFlatIP)...")
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.index.add(self.embeddings)
        logger.info(f"FAISS index built with {self.index.ntotal} vectors")
        
        # Save index
        self.save_index()
    
    def index_document(
        self,
        file_path: Union[str, Path],
        document_id: Optional[str] = None
    ) -> ProcessedDocument:
        """
        Index a single document (add to existing index).
        
        Args:
            file_path: Path to document
            document_id: Optional custom document ID
            
        Returns:
            ProcessedDocument
        """
        # Process document
        processed_doc = self.doc_processor.process_file(file_path, document_id)
        
        if not processed_doc.chunks:
            logger.warning(f"No chunks extracted from {file_path}")
            return processed_doc
        
        # Generate embeddings for new chunks (passages, not queries)
        texts = [self._get_chunk_text_for_embedding(chunk, is_query=False) for chunk in processed_doc.chunks]
        new_embeddings = self.encoder.encode(
            texts,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Add to existing index or create new one
        if self.index is None:
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])
        
        self.index.add(new_embeddings)
        
        # Store metadata
        self.chunks.extend(processed_doc.chunks)
        self.documents.append(processed_doc.to_dict())
        self.document_hashes[processed_doc.document_id] = hashlib.sha256(
            processed_doc.raw_text.encode()
        ).hexdigest()
        
        logger.info(f"Indexed document: {processed_doc.filename} ({len(processed_doc.chunks)} chunks)")
        
        # Save updated index
        self.save_index()
        
        return processed_doc
    
    def index_documents(
        self,
        file_paths: List[Union[str, Path]]
    ) -> List[ProcessedDocument]:
        """
        Index multiple documents at once.
        
        Args:
            file_paths: List of paths to documents
            
        Returns:
            List of ProcessedDocument
        """
        results = []
        all_chunks = []
        
        for file_path in tqdm(file_paths, desc="Processing documents"):
            try:
                processed_doc = self.doc_processor.process_file(file_path)
                results.append(processed_doc)
                all_chunks.extend(processed_doc.chunks)
                self.documents.append(processed_doc.to_dict())
                self.document_hashes[processed_doc.document_id] = hashlib.sha256(
                    processed_doc.raw_text.encode()
                ).hexdigest()
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
        
        if all_chunks:
            # Generate embeddings for all new chunks (passages, not queries)
            texts = [self._get_chunk_text_for_embedding(chunk, is_query=False) for chunk in all_chunks]
            new_embeddings = self.encoder.encode(
                texts,
                show_progress_bar=True,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).astype('float32')
            
            # Add to index
            if self.index is None:
                self.index = faiss.IndexFlatIP(self.embedding_dim)
                self.embeddings = new_embeddings
            else:
                self.embeddings = np.vstack([self.embeddings, new_embeddings])
            
            self.index.add(new_embeddings)
            self.chunks.extend(all_chunks)
            
            self.save_index()
        
        return results
    
    def remove_document(self, document_id: str) -> bool:
        """
        Remove a document from the index.
        
        Note: This requires rebuilding the index since FAISS doesn't support removal.
        
        Args:
            document_id: ID of document to remove
            
        Returns:
            True if document was found and removed
        """
        # Find chunks belonging to this document
        indices_to_keep = []
        chunks_to_keep = []
        
        for i, chunk in enumerate(self.chunks):
            if chunk.document_id != document_id:
                indices_to_keep.append(i)
                chunks_to_keep.append(chunk)
        
        if len(indices_to_keep) == len(self.chunks):
            logger.warning(f"Document {document_id} not found in index")
            return False
        
        # Remove document metadata
        self.documents = [d for d in self.documents if d.get('document_id') != document_id]
        self.document_hashes.pop(document_id, None)
        
        # Rebuild embeddings and index
        if indices_to_keep:
            self.embeddings = self.embeddings[indices_to_keep]
            self.chunks = chunks_to_keep
            
            self.index = faiss.IndexFlatIP(self.embedding_dim)
            self.index.add(self.embeddings)
        else:
            self.embeddings = None
            self.chunks = []
            self.index = None
        
        self.save_index()
        logger.info(f"Removed document {document_id} from index")
        return True
    
    def save_index(self):
        """Save index and documents to disk."""
        logger.info("Saving index to disk...")
        
        if self.index is not None:
            # Save FAISS index
            faiss.write_index(self.index, str(self.index_dir / "faiss.index"))
        
        # Save chunks
        chunks_data = [c.to_dict() for c in self.chunks]
        with open(self.index_dir / "chunks.json", 'w', encoding='utf-8') as f:
            json.dump(chunks_data, f, ensure_ascii=False, indent=2)
        
        # Save documents metadata
        with open(self.index_dir / "documents.json", 'w', encoding='utf-8') as f:
            json.dump(self.documents, f, ensure_ascii=False, indent=2)
        
        # Save document hashes
        with open(self.index_dir / "document_hashes.json", 'w', encoding='utf-8') as f:
            json.dump(self.document_hashes, f)
        
        # Save embeddings (for backup/debugging)
        if self.embeddings is not None:
            np.save(self.index_dir / "embeddings.npy", self.embeddings)
        
        # Save embedding model name for change detection
        from .config import config
        with open(self.index_dir / "embedding_model.txt", 'w') as f:
            f.write(config.EMBEDDING_MODEL)
        
        logger.info("Index saved successfully")
    
    def load_index(self):
        """Load existing index from disk."""
        logger.info("Loading index from disk...")
        
        # Load FAISS index
        index_path = self.index_dir / "faiss.index"
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found at {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        
        # Load chunks
        chunks_path = self.index_dir / "chunks.json"
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                chunks_data = json.load(f)
            self.chunks = [
                DocumentChunk(
                    chunk_id=c['chunk_id'],
                    document_id=c['document_id'],
                    content=c['content'],
                    metadata=c.get('metadata', {}),
                    page_number=c.get('page_number'),
                    section_title=c.get('section_title'),
                    language=c.get('language')
                )
                for c in chunks_data
            ]
        else:
            # Legacy format: try loading documents.pkl
            pkl_path = self.index_dir / "documents.pkl"
            if pkl_path.exists():
                with open(pkl_path, 'rb') as f:
                    self.documents = pickle.load(f)
                # Convert legacy documents to chunks
                self._convert_legacy_documents()
        
        # Load documents metadata
        docs_path = self.index_dir / "documents.json"
        if docs_path.exists():
            with open(docs_path, 'r', encoding='utf-8') as f:
                self.documents = json.load(f)
        
        # Load document hashes
        hashes_path = self.index_dir / "document_hashes.json"
        if hashes_path.exists():
            with open(hashes_path, 'r', encoding='utf-8') as f:
                self.document_hashes = json.load(f)
        
        # Load embeddings
        embeddings_path = self.index_dir / "embeddings.npy"
        if embeddings_path.exists():
            self.embeddings = np.load(embeddings_path)
        
        logger.info(f"Index loaded: {len(self.chunks)} chunks from {len(self.documents)} documents")
    
    def _convert_legacy_documents(self):
        """Convert legacy document format to new chunk format."""
        self.chunks = []
        for i, doc in enumerate(self.documents):
            chunk = DocumentChunk(
                chunk_id=f"{doc.get('id', i)}_0000",
                document_id=str(doc.get('id', i)),
                content=f"Title: {doc.get('title', '')}\n\nAbstract: {doc.get('abstract', '')}",
                metadata={
                    'title': doc.get('title', ''),
                    'authors': doc.get('authors', ''),
                    'submitter': doc.get('submitter', '')
                }
            )
            self.chunks.append(chunk)
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant document chunks.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with scores
        """
        if self.index is None or len(self.chunks) == 0:
            logger.warning("No documents indexed yet. Please upload documents first.")
            return []
        
        from .config import config
        
        # Retrieve more candidates for reranking
        retrieval_k = min(top_k * config.RERANK_MULTIPLIER, len(self.chunks))
        
        # Prepare query with E5 prefix if enabled
        query_text = query
        if config.USE_E5_PREFIX:
            query_text = "query: " + query
        
        # Encode query
        query_embedding = self.encoder.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).astype('float32')
        
        # Search (returns cosine similarity scores when using IndexFlatIP)
        scores, indices = self.index.search(query_embedding, retrieval_k)
        
        # Get candidates
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                chunk = self.chunks[idx]
                candidates.append({
                    'chunk': chunk,
                    'document': self._get_document_for_chunk(chunk),
                    'score': float(score),
                    'index': int(idx)
                })
        
        # Rerank results
        reranked = self._rerank_results(query, candidates)
        
        return reranked[:top_k]
    
    def _get_document_for_chunk(self, chunk: DocumentChunk) -> Dict:
        """Get document metadata for a chunk."""
        for doc in self.documents:
            if doc.get('document_id') == chunk.document_id:
                return doc
        
        # Return chunk metadata if document not found
        return {
            'document_id': chunk.document_id,
            'title': chunk.section_title or chunk.metadata.get('title', 'Unknown'),
            'filename': chunk.metadata.get('document_filename', 'Unknown')
        }
    
    def _rerank_results(self, query: str, candidates: List[Dict]) -> List[Dict]:
        """Rerank candidates based on semantic similarity."""
        if not candidates:
            return candidates
        
        from .config import config
        
        # Compute similarity scores
        candidate_texts = []
        for c in candidates:
            chunk = c['chunk']
            # Include section context in reranking (with passage prefix)
            text = self._get_chunk_text_for_embedding(chunk, is_query=False)
            candidate_texts.append(text)
        
        # Prepare query with E5 prefix if enabled
        query_text = query
        if config.USE_E5_PREFIX:
            query_text = "query: " + query
        
        query_emb = self.encoder.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        candidate_embs = self.encoder.encode(
            candidate_texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        # Compute cosine similarities
        similarities = []
        for cand_emb in candidate_embs:
            sim = np.dot(query_emb[0], cand_emb)
            similarities.append(float(sim))
        
        # Update scores
        for i, cand in enumerate(candidates):
            cand['rerank_score'] = similarities[i]
            cand['original_score'] = cand['score']
        
        # Sort by rerank score
        reranked = sorted(candidates, key=lambda x: x['rerank_score'], reverse=True)
        
        return reranked
    
    def get_index_size(self) -> dict:
        """Get index size statistics."""
        if self.index is None:
            return {"vectors": 0, "dimension": 0, "documents": 0, "chunks": 0}
        
        return {
            "vectors": self.index.ntotal,
            "dimension": self.embedding_dim,
            "documents": len(self.documents),
            "chunks": len(self.chunks)
        }
    
    def get_document_list(self) -> List[Dict]:
        """Get list of all indexed documents."""
        return [
            {
                "document_id": doc.get("document_id"),
                "filename": doc.get("filename"),
                "file_type": doc.get("file_type"),
                "language": doc.get("language"),
                "chunk_count": len([c for c in self.chunks if c.document_id == doc.get("document_id")])
            }
            for doc in self.documents
        ]
    
    def clear_index(self):
        """Clear the entire index."""
        self.index = None
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.document_hashes = {}
        
        # Remove files
        for file_name in ["faiss.index", "chunks.json", "documents.json", 
                          "document_hashes.json", "embeddings.npy"]:
            file_path = self.index_dir / file_name
            if file_path.exists():
                file_path.unlink()
        
        logger.info("Index cleared")


if __name__ == "__main__":
    """Build index when run as script."""
    from .config import config
    
    # Print configuration
    print("\n" + "="*70)
    print("📦 Building Vector Index")
    print("="*70)
    print(f"Data source:      {config.DATA_PATH}")
    print(f"Output directory: {config.INDEX_DIR}")
    print(f"Embedding model:  {config.EMBEDDING_MODEL}")
    print(f"Batch size:       {config.EMBEDDING_BATCH_SIZE}")
    print("="*70 + "\n")
    
    indexer = VectorIndexer(index_dir=config.INDEX_DIR)
    indexer.build_index(config.DATA_PATH)
    
    print("\n" + "="*70)
    print("✅ Index Built Successfully!")
    print("="*70)
    print(f"Total documents: {len(indexer.documents)}")
    print(f"Total chunks: {len(indexer.chunks)}")
    print(f"Index size: {indexer.get_index_size()}")
    print("="*70 + "\n")
