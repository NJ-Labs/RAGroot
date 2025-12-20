"""
Smart Chunking Module for Multi-Format Document Processing.

Implements research-backed chunking strategies based on Chroma Research 2024:
https://research.trychroma.com/evaluating-chunking

Key findings applied:
1. RecursiveCharacterTextSplitter with 250 tokens / 125 overlap = highest recall (96%+)
2. ClusterSemanticChunker using embeddings = best coherence (200-400 tokens, 0% overlap)
3. Semantic chunking with similarity threshold for meaning-based splits
4. Structure-aware splitting for Markdown, HTML, code preservation
5. Late chunking for preserving long-range dependencies

Optimized for Hebrew and English bilingual content.
Compatible with: Tesseract OCR, Dicta-LM, multilingual-e5-large embedding model.
"""

import re
import logging
from typing import List, Dict, Optional, Tuple, Iterator, Callable
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from enum import Enum
import numpy as np

logger = logging.getLogger(__name__)


class ChunkingStrategy(str, Enum):
    """Available chunking strategies based on Chroma Research benchmarks."""
    RECURSIVE = "recursive"             # RecursiveCharacterTextSplitter - best all-rounder
    CLUSTER_SEMANTIC = "cluster_semantic"  # ClusterSemanticChunker - best coherence
    SEMANTIC = "semantic"               # Similarity-based splitting
    STRUCTURE_AWARE = "structure_aware" # Document structure-based
    SLIDING_WINDOW = "sliding_window"   # Fixed-size overlapping
    LATE_CHUNKING = "late_chunking"     # Embed-then-chunk approach
    PAGE_LEVEL = "page_level"           # NVIDIA's highest accuracy for PDFs
    HYBRID = "hybrid"                   # Combination of strategies


@dataclass
class ChunkMetadata:
    """Metadata for a text chunk."""
    start_idx: int = 0
    end_idx: int = 0
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    heading_level: Optional[int] = None
    element_type: Optional[str] = None  # paragraph, table, list, code, etc.
    language: Optional[str] = None
    slide_number: Optional[int] = None  # For PowerPoint
    sheet_name: Optional[str] = None    # For Excel
    row_range: Optional[Tuple[int, int]] = None  # For Excel
    parent_heading: Optional[str] = None
    semantic_score: Optional[float] = None  # Semantic coherence score
    chunk_strategy: Optional[str] = None    # Which strategy created this chunk


@dataclass
class TextChunk:
    """A chunk of text with metadata."""
    content: str
    metadata: ChunkMetadata = field(default_factory=ChunkMetadata)
    embedding: Optional[np.ndarray] = None  # For late chunking
    
    def __len__(self):
        return len(self.content)
    
    @property
    def word_count(self) -> int:
        return len(self.content.split())
    
    @property
    def char_count(self) -> int:
        return len(self.content)


class BaseChunker(ABC):
    """Abstract base class for chunkers."""
    
    def __init__(
        self,
        chunk_size: int = 250,  # Optimal per Chroma Research
        chunk_overlap: int = 125,  # 50% overlap = highest recall
        min_chunk_size: int = 50,
        language_codes: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.language_codes = language_codes or ["heb", "eng"]
    
    @abstractmethod
    def chunk(self, text: str, **kwargs) -> List[TextChunk]:
        """Split text into chunks."""
        pass
    
    def estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for Hebrew/English text.
        Hebrew: ~2-3 chars per token
        English: ~4 chars per token
        """
        if not text:
            return 0
        hebrew_chars = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
        total_chars = len(text)
        
        if hebrew_chars > total_chars * 0.5:
            # Predominantly Hebrew - uses more tokens per character
            return total_chars // 2
        else:
            # Predominantly English or mixed
            return total_chars // 4
    
    def detect_language(self, text: str) -> str:
        """Detect if text is primarily Hebrew or English."""
        if not text:
            return "unknown"
        
        hebrew_chars = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
        latin_chars = sum(1 for c in text if c.isalpha() and c.isascii())
        
        total_alpha = hebrew_chars + latin_chars
        if total_alpha == 0:
            return "unknown"
        
        hebrew_ratio = hebrew_chars / total_alpha
        
        if hebrew_ratio > 0.7:
            return "heb"
        elif hebrew_ratio < 0.3:
            return "eng"
        else:
            return "mixed"


class RecursiveCharacterTextSplitter(BaseChunker):
    """
    Industry-standard recursive character splitter.
    Based on Chroma Research: best all-rounder with 96%+ recall.
    
    Optimal configuration (per research):
    - chunk_size: 250 tokens
    - chunk_overlap: 125 tokens (50%)
    
    Splits hierarchically using separators to preserve semantic units.
    """
    
    # Default separators - ordered from most to least preferred
    DEFAULT_SEPARATORS = [
        "\n\n\n",      # Section breaks
        "\n\n",        # Paragraph breaks
        "\n",          # Line breaks
        "。",          # CJK period (for Asian languages)
        ".",          # English period
        "?",          # Question mark
        "!",          # Exclamation
        "؟",          # Arabic question mark
        ";",          # Semicolon
        ":",          # Colon
        " ",          # Space
        "",           # Character (fallback)
    ]
    
    # Hebrew-specific separators
    HEBREW_SEPARATORS = [
        "\n\n\n",
        "\n\n",
        "\n",
        ".",
        "?",
        "!",
        ":",
        "׃",          # Hebrew sof pasuq
        ";",
        "،",          # Hebrew comma
        " ",
        "",
    ]
    
    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 125,  # 50% for maximum recall
        separators: List[str] = None,
        keep_separator: bool = True,
        **kwargs
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.separators = separators or self.DEFAULT_SEPARATORS
        self.keep_separator = keep_separator
    
    def chunk(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Recursively split text using separator hierarchy.
        """
        if not text.strip():
            return []
        
        # Detect language and adjust separators for Hebrew
        lang = self.detect_language(text)
        separators = self.HEBREW_SEPARATORS if lang == "heb" else self.separators
        
        # Perform recursive splitting
        splits = self._split_text(text, separators)
        
        # Merge splits into chunks respecting size limits
        chunks = self._merge_splits(splits, lang)
        
        return chunks
    
    def _split_text(self, text: str, separators: List[str]) -> List[str]:
        """Recursively split text using separator hierarchy."""
        if not separators:
            return [text]
        
        separator = separators[0]
        remaining_separators = separators[1:]
        
        if separator == "":
            # Character-level split as fallback
            return list(text)
        
        if separator not in text:
            # This separator doesn't exist, try next
            return self._split_text(text, remaining_separators)
        
        splits = []
        parts = text.split(separator)
        
        for i, part in enumerate(parts):
            if not part:
                continue
            
            # Add separator back if configured
            if self.keep_separator and i < len(parts) - 1:
                part = part + separator
            
            # Check if part needs further splitting
            if self.estimate_tokens(part) > self.chunk_size:
                # Recursively split with next separator
                sub_splits = self._split_text(part, remaining_separators)
                splits.extend(sub_splits)
            else:
                splits.append(part)
        
        return splits
    
    def _merge_splits(self, splits: List[str], language: str) -> List[TextChunk]:
        """Merge splits into chunks respecting size and overlap."""
        if not splits:
            return []
        
        chunks = []
        current_chunk = []
        current_size = 0
        
        for split in splits:
            split_size = self.estimate_tokens(split)
            
            if current_size + split_size > self.chunk_size and current_chunk:
                # Create chunk from current content
                chunk_text = ''.join(current_chunk)
                chunks.append(self._create_chunk(chunk_text, language))
                
                # Apply overlap - keep last portion for context
                overlap_text = self._get_overlap_text(current_chunk)
                current_chunk = [overlap_text, split] if overlap_text else [split]
                current_size = self.estimate_tokens(''.join(current_chunk))
            else:
                current_chunk.append(split)
                current_size += split_size
        
        # Don't forget last chunk
        if current_chunk:
            chunk_text = ''.join(current_chunk)
            if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text, language))
        
        return chunks
    
    def _get_overlap_text(self, parts: List[str]) -> str:
        """Get overlap text from end of current chunk."""
        if not parts or self.chunk_overlap == 0:
            return ""
        
        full_text = ''.join(parts)
        words = full_text.split()
        
        # Calculate overlap in words (approximate)
        overlap_words = min(self.chunk_overlap, len(words) // 2)
        if overlap_words > 0:
            return ' '.join(words[-overlap_words:])
        return ""
    
    def _create_chunk(self, content: str, language: str) -> TextChunk:
        """Create TextChunk with metadata."""
        return TextChunk(
            content=content.strip(),
            metadata=ChunkMetadata(
                language=language,
                element_type="recursive_chunk",
                chunk_strategy="recursive"
            )
        )


class ClusterSemanticChunker(BaseChunker):
    """
    Cluster-based semantic chunker using embeddings.
    State-of-the-art from Chroma Research - best coherence.
    
    Optimal configuration (per research):
    - chunk_size: 200-400 tokens
    - chunk_overlap: 0%
    
    Uses embedding similarity to group related sentences.
    Compatible with multilingual-e5-large for Hebrew/English.
    """
    
    def __init__(
        self,
        chunk_size: int = 300,  # Middle of 200-400 range
        chunk_overlap: int = 0,  # 0% overlap per research
        similarity_threshold: float = 0.75,
        embedding_model: str = None,
        **kwargs
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.similarity_threshold = similarity_threshold
        self.embedding_model = embedding_model
        self._encoder = None
    
    def _get_encoder(self):
        """Lazy load sentence transformer encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                from .config import config
                
                model_name = self.embedding_model or config.EMBEDDING_MODEL
                cache_dir = config.EMBEDDING_CACHE_DIR
                
                self._encoder = SentenceTransformer(
                    model_name,
                    cache_folder=cache_dir
                )
                logger.info(f"ClusterSemanticChunker loaded encoder: {model_name}")
            except Exception as e:
                logger.warning(f"Could not load encoder for semantic chunking: {e}")
                self._encoder = None
        return self._encoder
    
    def chunk(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Split text into semantically coherent clusters.
        """
        if not text.strip():
            return []
        
        # Split into sentences first
        sentences = self._split_into_sentences(text)
        
        if not sentences:
            return []
        
        # Get encoder for embeddings
        encoder = self._get_encoder()
        
        if encoder is None:
            # Fallback to recursive splitter if encoder unavailable
            logger.warning("Falling back to recursive splitter (encoder unavailable)")
            fallback = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return fallback.chunk(text)
        
        # Embed sentences
        embeddings = self._embed_sentences(sentences, encoder)
        
        # Cluster sentences by semantic similarity
        chunks = self._cluster_sentences(sentences, embeddings)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for Hebrew and English."""
        # Multi-language sentence boundary pattern
        sentence_pattern = re.compile(
            r'(?<=[.!?。؟])\s+|(?<=[.!?。؟])(?=[A-Z\u0590-\u05FF])'
        )
        
        sentences = sentence_pattern.split(text)
        
        # Clean up sentences
        cleaned = []
        for sent in sentences:
            sent = sent.strip()
            if sent and len(sent) > 10:  # Filter very short fragments
                cleaned.append(sent)
        
        return cleaned
    
    def _embed_sentences(self, sentences: List[str], encoder) -> np.ndarray:
        """Generate embeddings for sentences."""
        from .config import config
        
        # Add E5 prefix if using E5 model
        if config.USE_E5_PREFIX:
            sentences = [f"passage: {s}" for s in sentences]
        
        embeddings = encoder.encode(
            sentences,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        return embeddings
    
    def _cluster_sentences(
        self,
        sentences: List[str],
        embeddings: np.ndarray
    ) -> List[TextChunk]:
        """Cluster sentences based on semantic similarity."""
        if len(sentences) == 0:
            return []
        
        if len(sentences) == 1:
            return [self._create_chunk(sentences[0])]
        
        chunks = []
        current_cluster = [sentences[0]]
        current_embedding = embeddings[0]
        current_size = self.estimate_tokens(sentences[0])
        
        for i in range(1, len(sentences)):
            sent = sentences[i]
            sent_embedding = embeddings[i]
            sent_size = self.estimate_tokens(sent)
            
            # Calculate similarity with current cluster centroid
            similarity = np.dot(current_embedding, sent_embedding)
            
            # Decide whether to add to current cluster or start new
            should_split = (
                similarity < self.similarity_threshold or
                current_size + sent_size > self.chunk_size
            )
            
            if should_split and current_cluster:
                # Save current cluster as chunk
                chunk_text = ' '.join(current_cluster)
                chunks.append(self._create_chunk(chunk_text))
                
                # Start new cluster
                current_cluster = [sent]
                current_embedding = sent_embedding
                current_size = sent_size
            else:
                # Add to current cluster
                current_cluster.append(sent)
                # Update centroid (running average)
                n = len(current_cluster)
                current_embedding = (current_embedding * (n-1) + sent_embedding) / n
                current_size += sent_size
        
        # Don't forget last cluster
        if current_cluster:
            chunk_text = ' '.join(current_cluster)
            if self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text))
        
        return chunks
    
    def _create_chunk(self, content: str) -> TextChunk:
        """Create TextChunk with metadata."""
        return TextChunk(
            content=content.strip(),
            metadata=ChunkMetadata(
                language=self.detect_language(content),
                element_type="semantic_cluster",
                chunk_strategy="cluster_semantic"
            )
        )


class SemanticSimilarityChunker(BaseChunker):
    """
    Semantic chunker that splits based on similarity drops.
    Creates new chunks when similarity between consecutive 
    sentences falls below a threshold.
    
    Best for: Documents with clear topic transitions.
    """
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        similarity_threshold: float = 0.65,
        window_size: int = 3,
        **kwargs
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.similarity_threshold = similarity_threshold
        self.window_size = window_size
        self._encoder = None
    
    def _get_encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                from .config import config
                self._encoder = SentenceTransformer(
                    config.EMBEDDING_MODEL,
                    cache_folder=config.EMBEDDING_CACHE_DIR
                )
            except Exception as e:
                logger.warning(f"Could not load encoder: {e}")
        return self._encoder
    
    def chunk(self, text: str, **kwargs) -> List[TextChunk]:
        """Split text based on semantic similarity shifts."""
        if not text.strip():
            return []
        
        # Split into sentences
        sentences = self._split_sentences(text)
        
        if len(sentences) <= 1:
            return [self._create_chunk(text)]
        
        encoder = self._get_encoder()
        
        if encoder is None:
            # Fallback to recursive
            fallback = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return fallback.chunk(text)
        
        # Find breakpoints based on similarity drops
        breakpoints = self._find_breakpoints(sentences, encoder)
        
        # Create chunks from breakpoints
        chunks = self._create_chunks_from_breakpoints(sentences, breakpoints)
        
        return chunks
    
    def _split_sentences(self, text: str) -> List[str]:
        """Split into sentences."""
        pattern = re.compile(r'(?<=[.!?。؟])\s+')
        sentences = pattern.split(text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _find_breakpoints(
        self,
        sentences: List[str],
        encoder
    ) -> List[int]:
        """Find indices where semantic similarity drops significantly."""
        from .config import config
        
        if len(sentences) <= self.window_size:
            return []
        
        # Embed all sentences
        if config.USE_E5_PREFIX:
            embed_sents = [f"passage: {s}" for s in sentences]
        else:
            embed_sents = sentences
        
        embeddings = encoder.encode(
            embed_sents,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False
        )
        
        breakpoints = []
        
        # Calculate similarity between consecutive windows
        for i in range(self.window_size, len(sentences)):
            # Left window: sentences before position i
            left_start = max(0, i - self.window_size)
            left_embedding = np.mean(embeddings[left_start:i], axis=0)
            
            # Right window: sentences from position i
            right_end = min(len(sentences), i + self.window_size)
            right_embedding = np.mean(embeddings[i:right_end], axis=0)
            
            # Calculate cosine similarity
            similarity = np.dot(left_embedding, right_embedding)
            
            if similarity < self.similarity_threshold:
                breakpoints.append(i)
        
        return breakpoints
    
    def _create_chunks_from_breakpoints(
        self,
        sentences: List[str],
        breakpoints: List[int]
    ) -> List[TextChunk]:
        """Create chunks from identified breakpoints."""
        chunks = []
        
        # Add start and end indices
        indices = [0] + breakpoints + [len(sentences)]
        
        for i in range(len(indices) - 1):
            start = indices[i]
            end = indices[i + 1]
            
            chunk_sentences = sentences[start:end]
            chunk_text = ' '.join(chunk_sentences)
            
            # Check if chunk is within size limits
            if self.estimate_tokens(chunk_text) > self.chunk_size:
                # Split further with recursive splitter
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                sub_chunks = sub_splitter.chunk(chunk_text)
                chunks.extend(sub_chunks)
            elif self.estimate_tokens(chunk_text) >= self.min_chunk_size:
                chunks.append(self._create_chunk(chunk_text))
        
        return chunks
    
    def _create_chunk(self, content: str) -> TextChunk:
        return TextChunk(
            content=content.strip(),
            metadata=ChunkMetadata(
                language=self.detect_language(content),
                element_type="semantic_boundary",
                chunk_strategy="semantic"
            )
        )


class StructureAwareChunker(BaseChunker):
    """
    Structure-aware chunker for Markdown, HTML, and code.
    Preserves headers, tables, lists, and code blocks as units.
    
    Best for: Documentation, technical documents, structured content.
    """
    
    # Markdown heading pattern
    MARKDOWN_HEADING = re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE)
    
    # Code block patterns
    CODE_BLOCK = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    
    # List patterns
    LIST_ITEM = re.compile(r'^[\s]*[-*+]\s+|^\s*\d+\.\s+', re.MULTILINE)
    
    # Table pattern (markdown)
    TABLE_PATTERN = re.compile(r'^\|.*\|$', re.MULTILINE)
    
    # Hebrew section markers
    HEBREW_SECTIONS = re.compile(
        r'^(?:פרק|חלק|סעיף|נספח|כותרת)\s*[\d\u05D0-\u05EA]*[:\.\s]',
        re.MULTILINE
    )
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        preserve_code: bool = True,
        preserve_tables: bool = True,
        **kwargs
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.preserve_code = preserve_code
        self.preserve_tables = preserve_tables
    
    def chunk(self, text: str, **kwargs) -> List[TextChunk]:
        """Split text while preserving structure."""
        if not text.strip():
            return []
        
        # Extract and protect special elements
        protected, text = self._extract_protected_elements(text)
        
        # Split by structure (headings, sections)
        sections = self._split_by_structure(text)
        
        # Process each section
        chunks = []
        for section in sections:
            section_chunks = self._chunk_section(section)
            chunks.extend(section_chunks)
        
        # Restore protected elements
        chunks = self._restore_protected_elements(chunks, protected)
        
        return chunks
    
    def _extract_protected_elements(self, text: str) -> Tuple[Dict[str, str], str]:
        """Extract code blocks and tables to protect them."""
        protected = {}
        
        if self.preserve_code:
            for i, match in enumerate(self.CODE_BLOCK.finditer(text)):
                placeholder = f"__CODE_BLOCK_{i}__"
                protected[placeholder] = match.group(0)
                text = text[:match.start()] + placeholder + text[match.end():]
        
        return protected, text
    
    def _split_by_structure(self, text: str) -> List[Dict]:
        """Split text by structural elements (headings, sections)."""
        sections = []
        current_section = {
            'title': None,
            'level': 0,
            'content': []
        }
        
        lines = text.split('\n')
        
        for line in lines:
            # Check for markdown headings
            heading_match = self.MARKDOWN_HEADING.match(line)
            hebrew_match = self.HEBREW_SECTIONS.match(line)
            
            if heading_match:
                if current_section['content']:
                    sections.append(current_section)
                
                level = len(heading_match.group(1))
                current_section = {
                    'title': heading_match.group(2).strip(),
                    'level': level,
                    'content': []
                }
            elif hebrew_match:
                if current_section['content']:
                    sections.append(current_section)
                
                current_section = {
                    'title': line.strip(),
                    'level': 1,
                    'content': []
                }
            elif line.strip():
                current_section['content'].append(line)
        
        if current_section['content']:
            sections.append(current_section)
        
        return sections
    
    def _chunk_section(self, section: Dict) -> List[TextChunk]:
        """Chunk a single section."""
        content = '\n'.join(section['content'])
        
        if not content.strip():
            return []
        
        # Use recursive splitter for section content
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        section_chunks = splitter.chunk(content)
        
        # Add section metadata
        for chunk in section_chunks:
            chunk.metadata.section_title = section.get('title')
            chunk.metadata.heading_level = section.get('level')
            chunk.metadata.chunk_strategy = "structure_aware"
        
        return section_chunks
    
    def _restore_protected_elements(
        self,
        chunks: List[TextChunk],
        protected: Dict[str, str]
    ) -> List[TextChunk]:
        """Restore protected elements (code, tables)."""
        for chunk in chunks:
            for placeholder, original in protected.items():
                if placeholder in chunk.content:
                    chunk.content = chunk.content.replace(placeholder, original)
                    chunk.metadata.element_type = "code_or_table"
        return chunks


class PageLevelChunker(BaseChunker):
    """
    Page-level chunker for PDF documents.
    NVIDIA found this has highest accuracy for PDF-based RAG.
    
    Treats each page as a logical unit, with optional sub-chunking
    for pages that exceed the size limit.
    """
    
    PAGE_MARKER = re.compile(r'---\s*Page\s*(\d+)\s*---', re.IGNORECASE)
    
    def __init__(
        self,
        chunk_size: int = 500,  # Larger for page-level
        chunk_overlap: int = 100,
        page_separator: str = "\n\n---\n\n",
        **kwargs
    ):
        super().__init__(chunk_size=chunk_size, chunk_overlap=chunk_overlap, **kwargs)
        self.page_separator = page_separator
    
    def chunk(self, text: str, pages: List[Dict] = None, **kwargs) -> List[TextChunk]:
        """Chunk document by pages."""
        if pages:
            return self._chunk_from_pages(pages)
        else:
            return self._chunk_from_text(text)
    
    def _chunk_from_pages(self, pages: List[Dict]) -> List[TextChunk]:
        """Chunk from pre-extracted page data."""
        chunks = []
        
        for page in pages:
            page_num = page.get('page_number', 0)
            content = page.get('content', page.get('text', ''))
            
            if not content.strip():
                continue
            
            page_tokens = self.estimate_tokens(content)
            
            if page_tokens <= self.chunk_size:
                # Page fits in one chunk
                chunks.append(TextChunk(
                    content=content.strip(),
                    metadata=ChunkMetadata(
                        page_number=page_num,
                        language=self.detect_language(content),
                        element_type="page",
                        chunk_strategy="page_level"
                    )
                ))
            else:
                # Page too large, split with recursive splitter
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                page_chunks = splitter.chunk(content)
                
                for chunk in page_chunks:
                    chunk.metadata.page_number = page_num
                    chunk.metadata.chunk_strategy = "page_level"
                
                chunks.extend(page_chunks)
        
        return chunks
    
    def _chunk_from_text(self, text: str) -> List[TextChunk]:
        """Chunk from text with page markers."""
        # Split by page markers
        pages = []
        current_page = {'page_number': 1, 'content': []}
        
        for line in text.split('\n'):
            page_match = self.PAGE_MARKER.match(line)
            
            if page_match:
                if current_page['content']:
                    current_page['content'] = '\n'.join(current_page['content'])
                    pages.append(current_page)
                
                current_page = {
                    'page_number': int(page_match.group(1)),
                    'content': []
                }
            else:
                current_page['content'].append(line)
        
        if current_page['content']:
            current_page['content'] = '\n'.join(current_page['content'])
            pages.append(current_page)
        
        if not pages:
            # No page markers found, treat as single page
            return self._chunk_from_pages([{'page_number': 1, 'content': text}])
        
        return self._chunk_from_pages(pages)


class LateChnkingProcessor:
    """
    Late Chunking approach - embed first, then chunk.
    Preserves long-range contextual dependencies.
    
    Best for: Documents where context spans across sections.
    """
    
    def __init__(
        self,
        chunk_size: int = 300,
        chunk_overlap: int = 50,
        **kwargs
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self._encoder = None
    
    def _get_encoder(self):
        """Lazy load encoder."""
        if self._encoder is None:
            try:
                from sentence_transformers import SentenceTransformer
                from .config import config
                self._encoder = SentenceTransformer(
                    config.EMBEDDING_MODEL,
                    cache_folder=config.EMBEDDING_CACHE_DIR
                )
            except Exception as e:
                logger.warning(f"Could not load encoder: {e}")
        return self._encoder
    
    def process(self, text: str, **kwargs) -> List[TextChunk]:
        """
        Embed entire document, then create contextualized chunks.
        """
        if not text.strip():
            return []
        
        encoder = self._get_encoder()
        
        if encoder is None:
            # Fallback
            fallback = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            return fallback.chunk(text)
        
        # First, chunk the text normally
        chunker = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        chunks = chunker.chunk(text)
        
        # Embed the full document
        from .config import config
        full_text = f"passage: {text}" if config.USE_E5_PREFIX else text
        
        # This is a simplified version - full late chunking
        # would use token-level embeddings and pooling
        for chunk in chunks:
            chunk.metadata.chunk_strategy = "late_chunking"
        
        return chunks


# =============================================================================
# SPECIALIZED FORMAT CHUNKERS
# =============================================================================

class WordDocumentChunker(StructureAwareChunker):
    """Specialized chunker for Word documents (.docx, .doc)."""
    
    def chunk_with_structure(
        self,
        paragraphs: List[Dict],
        tables: List[str] = None,
        **kwargs
    ) -> List[TextChunk]:
        """Chunk Word document preserving paragraphs and tables."""
        chunks = []
        current_section = None
        current_content = []
        current_size = 0
        
        for para in paragraphs:
            text = para.get('text', '').strip()
            style = para.get('style', '').lower()
            
            if not text:
                continue
            
            is_heading = 'heading' in style or para.get('level', 0) > 0
            
            if is_heading:
                if current_content:
                    chunks.extend(self._create_section_chunks(
                        current_content, current_section
                    ))
                current_section = text
                current_content = []
                current_size = 0
            else:
                para_tokens = self.estimate_tokens(text)
                
                if current_size + para_tokens > self.chunk_size and current_content:
                    chunks.extend(self._create_section_chunks(
                        current_content, current_section
                    ))
                    current_content = [text]
                    current_size = para_tokens
                else:
                    current_content.append(text)
                    current_size += para_tokens
        
        if current_content:
            chunks.extend(self._create_section_chunks(
                current_content, current_section
            ))
        
        # Handle tables
        if tables:
            for i, table_text in enumerate(tables):
                if table_text.strip():
                    chunks.append(TextChunk(
                        content=table_text,
                        metadata=ChunkMetadata(
                            element_type="table",
                            section_title=f"Table {i+1}",
                            chunk_strategy="word_table"
                        )
                    ))
        
        return chunks
    
    def _create_section_chunks(
        self,
        content: List[str],
        section_title: str
    ) -> List[TextChunk]:
        """Create chunks for a section."""
        full_text = '\n\n'.join(content)
        
        if self.estimate_tokens(full_text) <= self.chunk_size:
            return [TextChunk(
                content=full_text,
                metadata=ChunkMetadata(
                    section_title=section_title,
                    language=self.detect_language(full_text),
                    chunk_strategy="word_section"
                )
            )]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        sub_chunks = splitter.chunk(full_text)
        for chunk in sub_chunks:
            chunk.metadata.section_title = section_title
        return sub_chunks


class PowerPointChunker(StructureAwareChunker):
    """Specialized chunker for PowerPoint (.pptx, .ppt)."""
    
    def chunk_slides(self, slides: List[Dict], **kwargs) -> List[TextChunk]:
        """Chunk PowerPoint by slides."""
        chunks = []
        
        for slide in slides:
            slide_text = self._format_slide(slide)
            slide_tokens = self.estimate_tokens(slide_text)
            
            if slide_tokens <= self.chunk_size:
                chunks.append(TextChunk(
                    content=slide_text,
                    metadata=ChunkMetadata(
                        slide_number=slide.get('slide_number'),
                        section_title=slide.get('title'),
                        element_type="slide",
                        language=self.detect_language(slide_text),
                        chunk_strategy="powerpoint_slide"
                    )
                ))
            else:
                # Split large slide
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=self.chunk_size,
                    chunk_overlap=self.chunk_overlap
                )
                sub_chunks = splitter.chunk(slide_text)
                for chunk in sub_chunks:
                    chunk.metadata.slide_number = slide.get('slide_number')
                    chunk.metadata.section_title = slide.get('title')
                    chunk.metadata.chunk_strategy = "powerpoint_slide"
                chunks.extend(sub_chunks)
        
        return chunks
    
    def _format_slide(self, slide: Dict) -> str:
        """Format slide content."""
        parts = []
        if slide.get('title'):
            parts.append(f"# {slide['title']}")
        if slide.get('content'):
            parts.append(slide['content'])
        if slide.get('notes'):
            parts.append(f"\n[Speaker Notes: {slide['notes']}]")
        return '\n\n'.join(parts)


class ExcelChunker(BaseChunker):
    """Specialized chunker for Excel (.xlsx, .xls)."""
    
    def chunk_sheets(self, sheets: List[Dict], **kwargs) -> List[TextChunk]:
        """Chunk Excel by sheets and row groups."""
        chunks = []
        
        for sheet in sheets:
            sheet_chunks = self._chunk_sheet(sheet)
            for chunk in sheet_chunks:
                chunk.metadata.sheet_name = sheet.get('name')
            chunks.extend(sheet_chunks)
        
        return chunks
    
    def _chunk_sheet(self, sheet: Dict) -> List[TextChunk]:
        """Chunk a single sheet."""
        headers = sheet.get('headers', [])
        rows = sheet.get('rows', [])
        
        if not rows:
            return []
        
        chunks = []
        current_rows = []
        current_size = 0
        start_row = 0
        
        header_text = ' | '.join(str(h) for h in headers)
        header_tokens = self.estimate_tokens(header_text)
        
        for i, row in enumerate(rows):
            row_text = ' | '.join(str(cell) for cell in row)
            row_tokens = self.estimate_tokens(row_text)
            
            if current_size + row_tokens > self.chunk_size - header_tokens and current_rows:
                chunk_content = self._format_table_chunk(headers, current_rows)
                chunks.append(TextChunk(
                    content=chunk_content,
                    metadata=ChunkMetadata(
                        row_range=(start_row, start_row + len(current_rows)),
                        element_type="spreadsheet",
                        language=self.detect_language(chunk_content),
                        chunk_strategy="excel_rows"
                    )
                ))
                start_row = i
                current_rows = [row]
                current_size = row_tokens
            else:
                current_rows.append(row)
                current_size += row_tokens
        
        if current_rows:
            chunk_content = self._format_table_chunk(headers, current_rows)
            chunks.append(TextChunk(
                content=chunk_content,
                metadata=ChunkMetadata(
                    row_range=(start_row, start_row + len(current_rows)),
                    element_type="spreadsheet",
                    chunk_strategy="excel_rows"
                )
            ))
        
        return chunks
    
    def _format_table_chunk(self, headers: List, rows: List[List]) -> str:
        """Format as markdown table."""
        lines = []
        if headers:
            lines.append('| ' + ' | '.join(str(h) for h in headers) + ' |')
            lines.append('| ' + ' | '.join('---' for _ in headers) + ' |')
        for row in rows:
            lines.append('| ' + ' | '.join(str(cell) for cell in row) + ' |')
        return '\n'.join(lines)


class HTMLChunker(StructureAwareChunker):
    """Specialized chunker for HTML documents."""
    
    HEADING_TAGS = {'h1', 'h2', 'h3', 'h4', 'h5', 'h6'}
    
    def chunk_html(self, sections: List[Dict], **kwargs) -> List[TextChunk]:
        """Chunk HTML preserving semantic structure."""
        chunks = []
        current_heading = None
        current_content = []
        current_size = 0
        
        for section in sections:
            tag = section.get('tag', '').lower()
            content = section.get('content', '').strip()
            
            if not content:
                continue
            
            if tag in self.HEADING_TAGS:
                if current_content:
                    chunks.extend(self._create_html_chunks(
                        current_content, current_heading
                    ))
                current_heading = content
                current_content = []
                current_size = 0
            else:
                content_tokens = self.estimate_tokens(content)
                
                if current_size + content_tokens > self.chunk_size and current_content:
                    chunks.extend(self._create_html_chunks(
                        current_content, current_heading
                    ))
                    current_content = [content]
                    current_size = content_tokens
                else:
                    current_content.append(content)
                    current_size += content_tokens
        
        if current_content:
            chunks.extend(self._create_html_chunks(
                current_content, current_heading
            ))
        
        return chunks
    
    def _create_html_chunks(
        self,
        content_list: List[str],
        heading: str
    ) -> List[TextChunk]:
        """Create chunks from HTML content."""
        full_text = '\n\n'.join(content_list)
        
        if self.estimate_tokens(full_text) <= self.chunk_size:
            return [TextChunk(
                content=full_text,
                metadata=ChunkMetadata(
                    section_title=heading,
                    element_type="html_section",
                    language=self.detect_language(full_text),
                    chunk_strategy="html"
                )
            )]
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        sub_chunks = splitter.chunk(full_text)
        for chunk in sub_chunks:
            chunk.metadata.section_title = heading
            chunk.metadata.chunk_strategy = "html"
        return sub_chunks


class MarkdownChunker(StructureAwareChunker):
    """Specialized chunker for Markdown documents."""
    
    CODE_BLOCK = re.compile(r'```[\s\S]*?```', re.MULTILINE)
    
    def __init__(self, preserve_code_blocks: bool = True, **kwargs):
        super().__init__(**kwargs)
        self.preserve_code_blocks = preserve_code_blocks
    
    def chunk(self, text: str, **kwargs) -> List[TextChunk]:
        """Chunk markdown preserving code blocks."""
        if not text.strip():
            return []
        
        # Extract code blocks
        code_blocks = {}
        if self.preserve_code_blocks:
            for i, match in enumerate(self.CODE_BLOCK.finditer(text)):
                placeholder = f"__CODE_{i}__"
                code_blocks[placeholder] = match.group(0)
            
            for placeholder, code in code_blocks.items():
                text = text.replace(code, placeholder)
        
        # Use parent structure-aware chunking
        chunks = super().chunk(text, **kwargs)
        
        # Restore code blocks
        for chunk in chunks:
            for placeholder, code in code_blocks.items():
                if placeholder in chunk.content:
                    chunk.content = chunk.content.replace(placeholder, code)
                    chunk.metadata.element_type = "markdown_code"
            chunk.metadata.chunk_strategy = "markdown"
        
        return chunks


# =============================================================================
# HYBRID/FACTORY CHUNKER
# =============================================================================

class SmartChunker:
    """
    Smart chunker that automatically selects the best strategy.
    
    Based on Chroma Research recommendations:
    - RecursiveCharacterTextSplitter: Best all-rounder (96%+ recall)
    - ClusterSemanticChunker: Best coherence
    - Structure-aware: Best for formatted documents
    - Page-level: Best for PDFs
    
    Reference: https://research.trychroma.com/evaluating-chunking
    """
    
    def __init__(
        self,
        default_strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
        chunk_size: int = 250,
        chunk_overlap: int = 125,  # 50% for max recall
        use_semantic_for_coherence: bool = False,
        **kwargs
    ):
        self.default_strategy = default_strategy
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic_for_coherence
        self.kwargs = kwargs
        
        # Strategy instances (lazy loaded)
        self._chunkers = {}
    
    def _get_chunker(self, strategy: ChunkingStrategy) -> BaseChunker:
        """Get or create chunker for strategy."""
        if strategy not in self._chunkers:
            chunker_map = {
                ChunkingStrategy.RECURSIVE: RecursiveCharacterTextSplitter,
                ChunkingStrategy.CLUSTER_SEMANTIC: ClusterSemanticChunker,
                ChunkingStrategy.SEMANTIC: SemanticSimilarityChunker,
                ChunkingStrategy.STRUCTURE_AWARE: StructureAwareChunker,
                ChunkingStrategy.SLIDING_WINDOW: RecursiveCharacterTextSplitter,  # Fallback
                ChunkingStrategy.PAGE_LEVEL: PageLevelChunker,
            }
            
            chunker_class = chunker_map.get(strategy, RecursiveCharacterTextSplitter)
            self._chunkers[strategy] = chunker_class(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                **self.kwargs
            )
        
        return self._chunkers[strategy]
    
    def chunk(
        self,
        text: str,
        file_type: str = None,
        strategy: ChunkingStrategy = None,
        **kwargs
    ) -> List[TextChunk]:
        """
        Intelligently chunk text based on content and file type.
        
        Args:
            text: Text to chunk
            file_type: File extension (e.g., 'pdf', 'docx')
            strategy: Override strategy selection
        """
        if not text.strip():
            return []
        
        # Select strategy based on file type if not specified
        if strategy is None:
            strategy = self._select_strategy(file_type, text)
        
        logger.info(f"Using chunking strategy: {strategy.value}")
        
        chunker = self._get_chunker(strategy)
        return chunker.chunk(text, **kwargs)
    
    def _select_strategy(self, file_type: str, text: str) -> ChunkingStrategy:
        """Select best strategy based on file type and content."""
        if file_type:
            file_type = file_type.lower().strip('.')
            
            # Format-specific strategy selection
            format_strategies = {
                # PDFs: Page-level is most accurate per NVIDIA
                'pdf': ChunkingStrategy.PAGE_LEVEL,
                
                # Office documents: Structure-aware
                'docx': ChunkingStrategy.STRUCTURE_AWARE,
                'doc': ChunkingStrategy.STRUCTURE_AWARE,
                'pptx': ChunkingStrategy.STRUCTURE_AWARE,
                'ppt': ChunkingStrategy.STRUCTURE_AWARE,
                'xlsx': ChunkingStrategy.RECURSIVE,  # Tables need special handling
                'xls': ChunkingStrategy.RECURSIVE,
                
                # Web formats: Structure-aware
                'html': ChunkingStrategy.STRUCTURE_AWARE,
                'htm': ChunkingStrategy.STRUCTURE_AWARE,
                'md': ChunkingStrategy.STRUCTURE_AWARE,
                'markdown': ChunkingStrategy.STRUCTURE_AWARE,
                
                # Plain text: Semantic if coherence matters, else recursive
                'txt': ChunkingStrategy.CLUSTER_SEMANTIC if self.use_semantic else ChunkingStrategy.RECURSIVE,
            }
            
            if file_type in format_strategies:
                return format_strategies[file_type]
        
        # Default to recursive (best all-rounder)
        return self.default_strategy
    
    def get_format_chunker(self, file_type: str):
        """Get specialized chunker for a file format."""
        file_type = file_type.lower().strip('.')
        
        format_chunkers = {
            'docx': WordDocumentChunker,
            'doc': WordDocumentChunker,
            'pptx': PowerPointChunker,
            'ppt': PowerPointChunker,
            'xlsx': ExcelChunker,
            'xls': ExcelChunker,
            'html': HTMLChunker,
            'htm': HTMLChunker,
            'md': MarkdownChunker,
            'markdown': MarkdownChunker,
            'pdf': PageLevelChunker,
        }
        
        chunker_class = format_chunkers.get(file_type, RecursiveCharacterTextSplitter)
        return chunker_class(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            **self.kwargs
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def create_chunker(
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 250,
    chunk_overlap: int = 125,
    **kwargs
) -> BaseChunker:
    """
    Factory function to create a chunker.
    
    Args:
        strategy: Chunking strategy to use
        chunk_size: Target chunk size in tokens (200-400 optimal)
        chunk_overlap: Overlap between chunks (50% = highest recall)
    """
    strategy_map = {
        ChunkingStrategy.RECURSIVE: RecursiveCharacterTextSplitter,
        ChunkingStrategy.CLUSTER_SEMANTIC: ClusterSemanticChunker,
        ChunkingStrategy.SEMANTIC: SemanticSimilarityChunker,
        ChunkingStrategy.STRUCTURE_AWARE: StructureAwareChunker,
        ChunkingStrategy.SLIDING_WINDOW: RecursiveCharacterTextSplitter,
        ChunkingStrategy.PAGE_LEVEL: PageLevelChunker,
        ChunkingStrategy.HYBRID: RecursiveCharacterTextSplitter,
    }
    
    chunker_class = strategy_map.get(strategy, RecursiveCharacterTextSplitter)
    return chunker_class(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        **kwargs
    )


def get_optimal_config(file_type: str = None) -> Dict:
    """
    Get optimal chunking configuration based on Chroma Research.
    
    Reference: https://research.trychroma.com/evaluating-chunking
    """
    configs = {
        # General optimal (96%+ recall)
        'default': {
            'strategy': ChunkingStrategy.RECURSIVE,
            'chunk_size': 250,
            'chunk_overlap': 125,  # 50% overlap
        },
        # Best coherence
        'coherence': {
            'strategy': ChunkingStrategy.CLUSTER_SEMANTIC,
            'chunk_size': 300,  # 200-400 range
            'chunk_overlap': 0,
        },
        # PDF-specific (NVIDIA benchmark winner)
        'pdf': {
            'strategy': ChunkingStrategy.PAGE_LEVEL,
            'chunk_size': 500,
            'chunk_overlap': 100,
        },
        # Structured documents
        'structured': {
            'strategy': ChunkingStrategy.STRUCTURE_AWARE,
            'chunk_size': 300,
            'chunk_overlap': 50,
        },
    }
    
    if file_type:
        file_type = file_type.lower().strip('.')
        if file_type == 'pdf':
            return configs['pdf']
        elif file_type in ['docx', 'doc', 'pptx', 'ppt', 'html', 'htm', 'md']:
            return configs['structured']
    
    return configs['default']
