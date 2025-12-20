"""
Document processing module using Docling for multi-format document parsing.
Supports PDF, DOCX, PPTX, XLSX, HTML, images, and more.
Optimized for Hebrew and English content.

Enhanced URL handling with:
- Robust HTTP fetching with retries and proper headers
- Fallback PDF text extraction using PyMuPDF
- Better support for Hebrew and multilingual documents
"""
import os
import hashlib
import logging
import tempfile
import time
from pathlib import Path
from typing import List, Dict, Optional, Union, Iterator, Tuple
from dataclasses import dataclass, field
from enum import Enum
from urllib.parse import urlparse, unquote
import json

logger = logging.getLogger(__name__)


# =============================================================================
# URL Fetcher with Robust Handling
# =============================================================================

class URLFetcher:
    """
    Robust URL fetcher with retry logic, proper headers, and SSL handling.
    Supports downloading PDFs and web pages for document processing.
    """
    
    # User agents to rotate for better compatibility
    USER_AGENTS = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    ]
    
    def __init__(
        self,
        timeout: int = 60,
        max_retries: int = 3,
        verify_ssl: bool = True
    ):
        self.timeout = timeout
        self.max_retries = max_retries
        self.verify_ssl = verify_ssl
        self._user_agent_idx = 0
    
    def _get_headers(self, url: str) -> Dict[str, str]:
        """Get HTTP headers with rotating user agent."""
        self._user_agent_idx = (self._user_agent_idx + 1) % len(self.USER_AGENTS)
        
        return {
            "User-Agent": self.USER_AGENTS[self._user_agent_idx],
            "Accept": "application/pdf,text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,he;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
    
    def _get_filename_from_url(self, url: str, content_type: str = None) -> str:
        """Extract filename from URL or content headers."""
        parsed = urlparse(url)
        path = unquote(parsed.path)
        filename = path.split('/')[-1] if path else ''
        
        if not filename or '.' not in filename:
            # Try to determine extension from content type
            ext_map = {
                'application/pdf': '.pdf',
                'text/html': '.html',
                'application/msword': '.doc',
                'application/vnd.openxmlformats-officedocument.wordprocessingml.document': '.docx',
            }
            ext = ext_map.get(content_type, '.html')
            filename = f"document{ext}"
        
        return filename
    
    def fetch_to_file(self, url: str, target_dir: Optional[str] = None) -> Tuple[Path, str, Dict]:
        """
        Fetch URL content and save to a temporary file.
        
        Args:
            url: The URL to fetch
            target_dir: Optional directory to save the file (uses tempdir if None)
            
        Returns:
            Tuple of (file_path, filename, metadata)
            
        Raises:
            Exception: If fetching fails after all retries
        """
        import httpx
        
        # Check if HTTP/2 is available
        try:
            import h2
            http2_available = True
        except ImportError:
            http2_available = False
        
        last_error = None
        
        for attempt in range(self.max_retries):
            try:
                logger.info(f"Fetching URL (attempt {attempt + 1}/{self.max_retries}): {url}")
                
                # Use httpx with HTTP/2 only if h2 package is available
                with httpx.Client(
                    timeout=self.timeout,
                    follow_redirects=True,
                    verify=self.verify_ssl,
                    http2=http2_available
                ) as client:
                    response = client.get(url, headers=self._get_headers(url))
                    response.raise_for_status()
                    
                    content_type = response.headers.get('content-type', '').split(';')[0].strip()
                    filename = self._get_filename_from_url(url, content_type)
                    
                    # Determine target directory
                    if target_dir:
                        save_dir = Path(target_dir)
                        save_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        save_dir = Path(tempfile.gettempdir()) / "rag_pdf_downloads"
                        save_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Generate unique filename to avoid conflicts
                    file_path = save_dir / f"{int(time.time() * 1000)}_{filename}"
                    
                    # Write content to file
                    with open(file_path, 'wb') as f:
                        f.write(response.content)
                    
                    metadata = {
                        "url": url,
                        "content_type": content_type,
                        "content_length": len(response.content),
                        "status_code": response.status_code,
                        "encoding": response.encoding,
                    }
                    
                    logger.info(f"Successfully downloaded: {filename} ({len(response.content)} bytes)")
                    return file_path, filename, metadata
                    
            except httpx.HTTPStatusError as e:
                last_error = e
                logger.warning(f"HTTP error {e.response.status_code} for {url}")
                if e.response.status_code in [403, 404, 410]:
                    raise  # Don't retry for these status codes
                    
            except (httpx.ConnectError, httpx.TimeoutException) as e:
                last_error = e
                logger.warning(f"Connection error for {url}: {e}")
                
            except Exception as e:
                last_error = e
                logger.warning(f"Error fetching {url}: {e}")
            
            # Wait before retry with exponential backoff
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
        
        raise Exception(f"Failed to fetch URL after {self.max_retries} attempts: {last_error}")
    
    def fetch_html_content(self, url: str) -> Tuple[str, Dict]:
        """
        Fetch HTML content and extract text with BeautifulSoup.
        Useful for web pages that aren't documents.
        
        Args:
            url: The URL to fetch
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        import httpx
        from bs4 import BeautifulSoup
        
        # Check if HTTP/2 is available
        try:
            import h2
            http2_available = True
        except ImportError:
            http2_available = False
        
        with httpx.Client(
            timeout=self.timeout,
            follow_redirects=True,
            verify=self.verify_ssl,
            http2=http2_available
        ) as client:
            response = client.get(url, headers=self._get_headers(url))
            response.raise_for_status()
            
            # Parse HTML
            soup = BeautifulSoup(response.text, 'lxml')
            
            # Remove script and style elements
            for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside']):
                element.decompose()
            
            # Extract title
            title = soup.title.string if soup.title else ''
            
            # Extract main content (try common content containers)
            main_content = None
            for selector in ['main', 'article', '[role="main"]', '.content', '#content', '.post-content']:
                main_content = soup.select_one(selector)
                if main_content:
                    break
            
            if main_content:
                text = main_content.get_text(separator='\n', strip=True)
            else:
                text = soup.get_text(separator='\n', strip=True)
            
            # Clean up text
            lines = [line.strip() for line in text.split('\n') if line.strip()]
            text = '\n'.join(lines)
            
            metadata = {
                "url": url,
                "title": title,
                "content_length": len(text),
            }
            
            return text, metadata


# =============================================================================
# PyMuPDF Fallback Extractor
# =============================================================================

class PyMuPDFExtractor:
    """
    Fallback PDF text extractor using PyMuPDF (fitz).
    Used when Docling OCR fails or produces poor results.
    Better support for embedded fonts and Hebrew text.
    """
    
    @staticmethod
    def is_available() -> bool:
        """Check if PyMuPDF is available."""
        try:
            import fitz
            return True
        except ImportError:
            return False
    
    @staticmethod
    def extract_text(file_path: Union[str, Path]) -> Tuple[str, Dict]:
        """
        Extract text from PDF using PyMuPDF.
        
        Args:
            file_path: Path to the PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        import fitz  # PyMuPDF
        
        file_path = Path(file_path)
        doc = fitz.open(file_path)
        
        text_parts = []
        page_texts = []
        
        for page_num, page in enumerate(doc, 1):
            # Extract text with layout preservation
            page_text = page.get_text("text", sort=True)
            
            if page_text.strip():
                text_parts.append(f"--- Page {page_num} ---\n{page_text}")
                page_texts.append({
                    "page": page_num,
                    "text": page_text,
                    "char_count": len(page_text)
                })
        
        full_text = '\n\n'.join(text_parts)
        
        metadata = {
            "page_count": len(doc),
            "title": doc.metadata.get("title", ""),
            "author": doc.metadata.get("author", ""),
            "subject": doc.metadata.get("subject", ""),
            "keywords": doc.metadata.get("keywords", ""),
            "pages": page_texts,
            "extraction_method": "pymupdf"
        }
        
        doc.close()
        
        return full_text, metadata
    
    @staticmethod
    def has_extractable_text(file_path: Union[str, Path], min_chars: int = 100) -> bool:
        """
        Check if PDF has extractable text (not just scanned images).
        
        Args:
            file_path: Path to the PDF file
            min_chars: Minimum characters to consider as having text
            
        Returns:
            True if PDF has extractable text
        """
        try:
            import fitz
            doc = fitz.open(file_path)
            
            total_chars = 0
            for page in doc:
                text = page.get_text("text")
                total_chars += len(text.strip())
                if total_chars >= min_chars:
                    doc.close()
                    return True
            
            doc.close()
            return total_chars >= min_chars
            
        except Exception as e:
            logger.warning(f"Error checking PDF text: {e}")
            return False


class DocumentLanguage(str, Enum):
    """Supported document languages."""
    HEBREW = "heb"
    ENGLISH = "eng"
    AUTO = "auto"  # Auto-detect


@dataclass
class DocumentChunk:
    """Represents a chunk of a processed document."""
    chunk_id: str
    document_id: str
    content: str
    metadata: Dict = field(default_factory=dict)
    page_number: Optional[int] = None
    section_title: Optional[str] = None
    language: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            "chunk_id": self.chunk_id,
            "document_id": self.document_id,
            "content": self.content,
            "metadata": self.metadata,
            "page_number": self.page_number,
            "section_title": self.section_title,
            "language": self.language
        }


@dataclass
class ProcessedDocument:
    """Represents a fully processed document."""
    document_id: str
    filename: str
    file_type: str
    chunks: List[DocumentChunk] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    raw_text: str = ""
    language: Optional[str] = None
    page_count: int = 0
    
    def to_dict(self) -> Dict:
        return {
            "document_id": self.document_id,
            "filename": self.filename,
            "file_type": self.file_type,
            "chunks": [c.to_dict() for c in self.chunks],
            "metadata": self.metadata,
            "language": self.language,
            "page_count": self.page_count
        }


class DoclingProcessor:
    """
    Document processor using IBM's Docling library.
    
    Supports:
    - PDF (with OCR for scanned documents)
    - DOCX, XLSX, PPTX (Office formats)
    - HTML, XHTML
    - Images (PNG, JPEG, TIFF, BMP, WEBP)
    - Markdown, AsciiDoc
    - CSV
    - Audio files (WAV, MP3) with ASR
    """
    
    # Supported file extensions
    SUPPORTED_EXTENSIONS = {
        # Documents
        ".pdf": "pdf",
        ".docx": "docx",
        ".doc": "doc",
        ".pptx": "pptx",
        ".ppt": "ppt",
        ".xlsx": "xlsx",
        ".xls": "xls",
        # Web formats
        ".html": "html",
        ".htm": "html",
        ".xhtml": "xhtml",
        # Text formats
        ".md": "markdown",
        ".markdown": "markdown",
        ".adoc": "asciidoc",
        ".txt": "text",
        ".csv": "csv",
        ".json": "json",
        ".jsonl": "jsonl",
        # Images
        ".png": "image",
        ".jpg": "image",
        ".jpeg": "image",
        ".tiff": "image",
        ".tif": "image",
        ".bmp": "image",
        ".webp": "image",
        # Audio (requires ASR)
        ".wav": "audio",
        ".mp3": "audio",
        # Video text tracks
        ".vtt": "webvtt",
    }
    
    def __init__(
        self,
        ocr_enabled: bool = True,
        ocr_languages: List[str] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        use_gpu: bool = False
    ):
        """
        Initialize the Docling document processor.
        
        Args:
            ocr_enabled: Enable OCR for scanned documents and images
            ocr_languages: List of OCR language codes (e.g., ['heb', 'eng'])
            chunk_size: Target chunk size in tokens
            chunk_overlap: Overlap between chunks in tokens
            use_gpu: Use GPU acceleration for OCR and models
        """
        from .config import config
        
        self.ocr_enabled = ocr_enabled
        self.ocr_languages = ocr_languages or ["heb", "eng"]
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_gpu = use_gpu and not config.FORCE_CPU
        
        self._converter = None
        self._chunker = None
        self._initialized = False
        
    def _initialize(self):
        """Lazy initialization of Docling components."""
        if self._initialized:
            return
            
        try:
            from docling.document_converter import DocumentConverter, PdfFormatOption
            from docling.datamodel.pipeline_options import (
                PdfPipelineOptions,
                TableFormerMode,
                AcceleratorDevice,
                AcceleratorOptions
            )
            from docling.datamodel.base_models import InputFormat
            from docling.chunking import HybridChunker
            
            logger.info("Initializing Docling document processor...")
            
            # Configure PDF pipeline with OCR
            pdf_pipeline_options = PdfPipelineOptions()
            pdf_pipeline_options.do_table_structure = True
            pdf_pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            
            # Configure OCR if enabled - try Tesseract first, fall back to RapidOCR
            if self.ocr_enabled:
                pdf_pipeline_options.do_ocr = True
                
                # Try Tesseract first (best Hebrew support)
                tesseract_available = False
                try:
                    import shutil
                    import tesserocr  # This is required for Docling's TesseractOcrOptions
                    
                    if shutil.which('tesseract'):
                        from docling.datamodel.pipeline_options import TesseractOcrOptions
                        # Map language codes to Tesseract format
                        tesseract_lang_map = {
                            'he': 'heb', 'en': 'eng', 'ar': 'ara',
                            'heb': 'heb', 'eng': 'eng', 'ara': 'ara',
                        }
                        tesseract_langs = [tesseract_lang_map.get(lang, lang) for lang in self.ocr_languages]
                        pdf_pipeline_options.ocr_options = TesseractOcrOptions(
                            lang=tesseract_langs,
                        )
                        tesseract_available = True
                        logger.info(f"OCR enabled with Tesseract for languages: {tesseract_langs}")
                    else:
                        logger.warning("Tesseract binary not found in PATH")
                except ImportError as e:
                    logger.warning(f"tesserocr not available: {e}")
                except Exception as e:
                    logger.warning(f"Tesseract initialization error: {e}")
                
                if not tesseract_available:
                    # Fall back to RapidOCR (no system installation needed)
                    try:
                        from docling.datamodel.pipeline_options import RapidOcrOptions
                        pdf_pipeline_options.ocr_options = RapidOcrOptions()
                        logger.info("OCR enabled with RapidOCR (Tesseract not available)")
                    except ImportError:
                        logger.warning("RapidOCR not available either, OCR may not work")
            
            # Configure GPU acceleration
            if self.use_gpu:
                pdf_pipeline_options.accelerator_options = AcceleratorOptions(
                    device=AcceleratorDevice.CUDA
                )
                logger.info("GPU acceleration enabled for Docling")
            
            # Create converter with format-specific options
            self._converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: PdfFormatOption(
                        pipeline_options=pdf_pipeline_options
                    )
                }
            )
            
            # Initialize chunker with tokenizer alignment
            # Using the same model as embeddings for consistency
            from .config import config
            self._chunker = HybridChunker(
                tokenizer=config.EMBEDDING_MODEL,
                max_tokens=self.chunk_size,
                merge_peers=True
            )
            
            self._initialized = True
            logger.info("Docling processor initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import Docling: {e}")
            logger.error("Please install docling: pip install docling")
            raise ImportError(
                "Docling is required for document processing. "
                "Install with: pip install docling"
            ) from e
        except Exception as e:
            logger.error(f"Failed to initialize Docling: {e}")
            raise
    
    @staticmethod
    def generate_document_id(file_path: Union[str, Path], content_hash: bool = True) -> str:
        """Generate a unique document ID based on file path and optionally content."""
        file_path = Path(file_path)
        
        if content_hash and file_path.exists():
            # Include content hash for deduplication
            hasher = hashlib.sha256()
            hasher.update(str(file_path.name).encode())
            with open(file_path, 'rb') as f:
                for chunk in iter(lambda: f.read(8192), b''):
                    hasher.update(chunk)
            return hasher.hexdigest()[:16]
        else:
            # Just use filename hash
            return hashlib.sha256(str(file_path.name).encode()).hexdigest()[:16]
    
    @staticmethod
    def is_supported(file_path: Union[str, Path]) -> bool:
        """Check if a file format is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in DoclingProcessor.SUPPORTED_EXTENSIONS
    
    @staticmethod
    def get_file_type(file_path: Union[str, Path]) -> Optional[str]:
        """Get the file type category."""
        ext = Path(file_path).suffix.lower()
        return DoclingProcessor.SUPPORTED_EXTENSIONS.get(ext)
    
    def detect_language(self, text: str) -> str:
        """
        Simple language detection for Hebrew vs English.
        
        Returns 'heb' for Hebrew, 'eng' for English, or 'mixed' for both.
        """
        if not text:
            return "unknown"
        
        # Count Hebrew characters (Unicode range for Hebrew)
        hebrew_chars = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
        # Count Latin characters
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
    
    def process_file(
        self,
        file_path: Union[str, Path],
        document_id: Optional[str] = None,
        timeout: int = 120,
        use_fallback: bool = True
    ) -> ProcessedDocument:
        """
        Process a single document file or URL.
        
        Args:
            file_path: Path to the document file or URL
            document_id: Optional custom document ID
            timeout: Timeout for URL fetching in seconds
            use_fallback: Use PyMuPDF fallback if Docling fails
            
        Returns:
            ProcessedDocument with chunks and metadata
        """
        self._initialize()
        
        is_url = str(file_path).startswith(('http://', 'https://'))
        temp_file = None
        original_url = None
        
        try:
            if is_url:
                # Handle URL with robust fetching
                original_url = str(file_path)
                logger.info(f"Fetching document from URL: {original_url}")
                
                fetcher = URLFetcher(timeout=timeout, max_retries=3)
                
                try:
                    temp_file, filename, url_metadata = fetcher.fetch_to_file(original_url)
                    file_path = temp_file
                    doc_id = document_id or hashlib.sha256(original_url.encode()).hexdigest()[:16]
                    file_type = self.get_file_type(filename) or "web"
                    file_size = url_metadata.get("content_length", 0)
                    source = original_url
                    
                    logger.info(f"Downloaded: {filename} ({file_size} bytes), type: {file_type}")
                    
                except Exception as fetch_error:
                    logger.warning(f"File download failed, trying HTML extraction: {fetch_error}")
                    
                    # Try HTML extraction for web pages
                    try:
                        text, html_metadata = fetcher.fetch_html_content(original_url)
                        
                        if text.strip():
                            # Process as raw text
                            return self.process_text(
                                text,
                                source_name=html_metadata.get("title", original_url)
                            )
                        else:
                            raise Exception("No content extracted from URL")
                            
                    except Exception as html_error:
                        raise Exception(f"Failed to fetch URL: {fetch_error}; HTML fallback also failed: {html_error}")
            else:
                file_path = Path(file_path)
                if not file_path.exists():
                    raise FileNotFoundError(f"File not found: {file_path}")
                
                if not self.is_supported(file_path):
                    raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
                # Generate document ID if not provided
                doc_id = document_id or self.generate_document_id(file_path)
                file_type = self.get_file_type(file_path)
                filename = file_path.name
                file_size = file_path.stat().st_size
                source = str(file_path)
            
            logger.info(f"Processing document: {filename} (type: {file_type})")
            
            raw_text = ""
            chunks = []
            page_count = 0
            extraction_method = None
            extraction_success = False
            
            # For PDFs, try PyMuPDF FIRST (faster and better for Hebrew embedded fonts)
            if file_type == "pdf" and use_fallback and PyMuPDFExtractor.is_available():
                logger.info("Trying PyMuPDF first for PDF text extraction (best for Hebrew)")
                
                try:
                    # Check if PDF has extractable text
                    if PyMuPDFExtractor.has_extractable_text(file_path, min_chars=200):
                        raw_text, pdf_metadata = PyMuPDFExtractor.extract_text(file_path)
                        extraction_method = "pymupdf"
                        page_count = pdf_metadata.get("page_count", 0)
                        
                        if len(raw_text.strip()) > 100:
                            # Chunk the extracted text
                            chunks = self._chunk_raw_text(raw_text, doc_id, filename, file_type)
                            extraction_success = True
                            logger.info(f"PyMuPDF extracted {len(raw_text)} chars, {len(chunks)} chunks (Hebrew-optimized)")
                        else:
                            logger.info("PyMuPDF extracted minimal text, will try Docling OCR")
                    else:
                        logger.info("PDF appears to be scanned/image-based, will try Docling OCR")
                        
                except Exception as pymupdf_error:
                    logger.warning(f"PyMuPDF extraction failed: {pymupdf_error}")
            
            # If PyMuPDF didn't work, try Docling (with timeout for OCR)
            if not extraction_success:
                import concurrent.futures
                
                logger.info("Trying Docling for document processing...")
                
                def docling_convert():
                    result = self._converter.convert(str(file_path))
                    return result.document
                
                try:
                    # Use ThreadPoolExecutor for timeout on Docling processing
                    docling_timeout = 180  # 3 minutes max for OCR
                    
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(docling_convert)
                        
                        try:
                            docling_doc = future.result(timeout=docling_timeout)
                            
                            # Extract full text
                            raw_text = docling_doc.export_to_markdown()
                            
                            # Check if we got meaningful content
                            if len(raw_text.strip()) >= 50:
                                extraction_success = True
                                extraction_method = "docling"
                                page_count = len(docling_doc.pages) if hasattr(docling_doc, 'pages') and docling_doc.pages else 0
                                
                                # Chunk the document using Docling's chunker
                                chunk_idx = 0
                                for chunk in self._chunker.chunk(docling_doc):
                                    chunk_text = self._chunker.contextualize(chunk)
                                    
                                    if not chunk_text.strip():
                                        continue
                                    
                                    chunk_language = self.detect_language(chunk_text)
                                    
                                    # Build chunk metadata
                                    chunk_metadata = {
                                        "document_filename": filename,
                                        "file_type": file_type,
                                    }
                                    
                                    # Extract section headers if available
                                    section_title = None
                                    if hasattr(chunk, 'meta') and chunk.meta:
                                        if hasattr(chunk.meta, 'headings') and chunk.meta.headings:
                                            section_title = " > ".join(chunk.meta.headings)
                                            chunk_metadata["headings"] = chunk.meta.headings
                                        if hasattr(chunk.meta, 'captions') and chunk.meta.captions:
                                            chunk_metadata["captions"] = chunk.meta.captions
                                    
                                    # Get page number if available
                                    page_num = None
                                    if hasattr(chunk, 'meta') and chunk.meta and hasattr(chunk.meta, 'page_no'):
                                        page_num = chunk.meta.page_no
                                    
                                    doc_chunk = DocumentChunk(
                                        chunk_id=f"{doc_id}_{chunk_idx:04d}",
                                        document_id=doc_id,
                                        content=chunk_text,
                                        metadata=chunk_metadata,
                                        page_number=page_num,
                                        section_title=section_title,
                                        language=chunk_language
                                    )
                                    chunks.append(doc_chunk)
                                    chunk_idx += 1
                                
                                logger.info(f"Docling extracted {len(raw_text)} chars, {len(chunks)} chunks")
                            else:
                                logger.warning(f"Docling extracted very little text ({len(raw_text)} chars)")
                                
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"Docling processing timed out after {docling_timeout}s")
                            future.cancel()
                            
                except Exception as docling_error:
                    logger.warning(f"Docling processing failed: {docling_error}")
            
            # Final fallback: try PyMuPDF if we haven't already and Docling failed
            if not extraction_success and file_type == "pdf" and extraction_method != "pymupdf":
                if PyMuPDFExtractor.is_available():
                    logger.info("Final attempt: PyMuPDF fallback for PDF text extraction")
                    
                    try:
                        raw_text, pdf_metadata = PyMuPDFExtractor.extract_text(file_path)
                        extraction_method = "pymupdf_fallback"
                        page_count = pdf_metadata.get("page_count", 0)
                        
                        if len(raw_text.strip()) > 50:
                            chunks = self._chunk_raw_text(raw_text, doc_id, filename, file_type)
                            extraction_success = True
                            logger.info(f"PyMuPDF fallback extracted {len(raw_text)} chars, {len(chunks)} chunks")
                        else:
                            logger.warning("PyMuPDF also extracted minimal text - document may be image-only")
                            
                    except Exception as pymupdf_error:
                        logger.error(f"PyMuPDF fallback also failed: {pymupdf_error}")
                        
            if not extraction_success:
                logger.warning(f"Could not extract meaningful text from {filename}")
            
            # Detect document language
            language = self.detect_language(raw_text)
            logger.info(f"Detected language: {language}, extraction method: {extraction_method}")
            
            # Extract metadata
            metadata = {
                "source": source,
                "filename": filename,
                "file_size": file_size,
                "file_type": file_type,
                "language": language,
                "extraction_method": extraction_method,
            }
            
            if original_url:
                metadata["original_url"] = original_url
            
            logger.info(f"Extracted {len(chunks)} chunks from {filename}")
            
            return ProcessedDocument(
                document_id=doc_id,
                filename=filename,
                file_type=file_type,
                chunks=chunks,
                metadata=metadata,
                raw_text=raw_text,
                language=language,
                page_count=page_count
            )
            
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            raise
            
        finally:
            # Clean up temporary file if created
            if temp_file and Path(temp_file).exists():
                try:
                    Path(temp_file).unlink()
                    logger.debug(f"Cleaned up temp file: {temp_file}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to clean up temp file: {cleanup_error}")
    
    def _chunk_raw_text(
        self,
        text: str,
        doc_id: str,
        filename: str,
        file_type: str
    ) -> List[DocumentChunk]:
        """
        Chunk raw text when Docling chunker is not available.
        Uses paragraph-based chunking with size limits.
        """
        chunks = []
        
        # Split by paragraphs (double newline) or pages (--- Page X ---)
        parts = []
        current_page = None
        
        for line in text.split('\n'):
            if line.startswith('--- Page ') and line.endswith(' ---'):
                try:
                    current_page = int(line.replace('--- Page ', '').replace(' ---', ''))
                except:
                    pass
                continue
            parts.append((line, current_page))
        
        # Reconstruct paragraphs with page info
        paragraphs = []
        current_para = []
        current_para_page = None
        
        for line, page in parts:
            if not line.strip():
                if current_para:
                    paragraphs.append(('\n'.join(current_para), current_para_page))
                    current_para = []
                    current_para_page = None
            else:
                current_para.append(line)
                if page is not None:
                    current_para_page = page
        
        if current_para:
            paragraphs.append(('\n'.join(current_para), current_para_page))
        
        # Chunk paragraphs with size limits
        current_chunk = []
        current_chunk_size = 0
        current_chunk_page = None
        chunk_idx = 0
        max_chars = self.chunk_size * 4  # Rough estimate: 4 chars per token
        
        for para, page in paragraphs:
            para_size = len(para)
            
            if current_chunk_size + para_size > max_chars and current_chunk:
                # Save current chunk
                chunk_text = '\n\n'.join(current_chunk)
                chunk_language = self.detect_language(chunk_text)
                
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_{chunk_idx:04d}",
                    document_id=doc_id,
                    content=chunk_text,
                    metadata={
                        "document_filename": filename,
                        "file_type": file_type,
                    },
                    page_number=current_chunk_page,
                    language=chunk_language
                ))
                chunk_idx += 1
                current_chunk = []
                current_chunk_size = 0
            
            current_chunk.append(para)
            current_chunk_size += para_size
            if page is not None:
                current_chunk_page = page
        
        # Don't forget the last chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_language = self.detect_language(chunk_text)
            
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_{chunk_idx:04d}",
                document_id=doc_id,
                content=chunk_text,
                metadata={
                    "document_filename": filename,
                    "file_type": file_type,
                },
                page_number=current_chunk_page,
                language=chunk_language
            ))
        
        return chunks
    
    def process_directory(
        self,
        directory_path: Union[str, Path],
        recursive: bool = True,
        file_filter: Optional[List[str]] = None
    ) -> Iterator[ProcessedDocument]:
        """
        Process all supported documents in a directory.
        
        Args:
            directory_path: Path to the directory
            recursive: Whether to process subdirectories
            file_filter: Optional list of extensions to filter (e.g., ['.pdf', '.docx'])
            
        Yields:
            ProcessedDocument for each successfully processed file
        """
        directory_path = Path(directory_path)
        if not directory_path.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory_path}")
        
        # Get all files
        if recursive:
            files = list(directory_path.rglob("*"))
        else:
            files = list(directory_path.glob("*"))
        
        # Filter to supported files
        supported_files = []
        for f in files:
            if not f.is_file():
                continue
            if not self.is_supported(f):
                continue
            if file_filter and f.suffix.lower() not in file_filter:
                continue
            supported_files.append(f)
        
        logger.info(f"Found {len(supported_files)} supported files in {directory_path}")
        
        for file_path in supported_files:
            try:
                yield self.process_file(file_path)
            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")
                continue
    
    def process_text(self, text: str, source_name: str = "text_input") -> ProcessedDocument:
        """
        Process raw text content directly.
        
        Args:
            text: The text content to process
            source_name: A name for the text source
            
        Returns:
            ProcessedDocument with chunks
        """
        doc_id = hashlib.sha256(text.encode()).hexdigest()[:16]
        language = self.detect_language(text)
        
        # Simple chunking for raw text (by paragraphs with size limits)
        paragraphs = text.split('\n\n')
        chunks = []
        current_chunk = ""
        chunk_idx = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            
            # Check if adding this paragraph would exceed limit
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para
            
            # Rough estimate: 4 chars per token
            if len(test_chunk) / 4 > self.chunk_size and current_chunk:
                # Save current chunk
                chunks.append(DocumentChunk(
                    chunk_id=f"{doc_id}_{chunk_idx:04d}",
                    document_id=doc_id,
                    content=current_chunk,
                    metadata={"source": source_name},
                    language=self.detect_language(current_chunk)
                ))
                chunk_idx += 1
                current_chunk = para
            else:
                current_chunk = test_chunk
        
        # Don't forget the last chunk
        if current_chunk:
            chunks.append(DocumentChunk(
                chunk_id=f"{doc_id}_{chunk_idx:04d}",
                document_id=doc_id,
                content=current_chunk,
                metadata={"source": source_name},
                language=self.detect_language(current_chunk)
            ))
        
        return ProcessedDocument(
            document_id=doc_id,
            filename=source_name,
            file_type="text",
            chunks=chunks,
            metadata={"source": source_name},
            raw_text=text,
            language=language
        )
    
    def process_jsonl(self, file_path: Union[str, Path]) -> Iterator[ProcessedDocument]:
        """
        Process a JSONL file where each line is a document.
        
        Expected format (flexible):
        - Can have 'id', 'title', 'abstract', 'content', 'text' fields
        - Will extract whatever text fields are available
        
        Args:
            file_path: Path to the JSONL file
            
        Yields:
            ProcessedDocument for each line
        """
        file_path = Path(file_path)
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    
                    # Extract document ID
                    doc_id = data.get('id', data.get('doc_id', f"line_{line_num}"))
                    
                    # Extract title
                    title = data.get('title', data.get('name', ''))
                    
                    # Extract main content (try various field names)
                    content_fields = ['abstract', 'content', 'text', 'body', 'description']
                    content = ""
                    for field in content_fields:
                        if field in data and data[field]:
                            content = data[field]
                            break
                    
                    # Combine title and content
                    full_text = f"{title}\n\n{content}" if title else content
                    
                    if not full_text.strip():
                        logger.warning(f"Line {line_num}: No content found, skipping")
                        continue
                    
                    # Create document
                    language = self.detect_language(full_text)
                    
                    # Create a single chunk for JSONL entries (they're usually pre-chunked)
                    chunk = DocumentChunk(
                        chunk_id=f"{doc_id}_0000",
                        document_id=str(doc_id),
                        content=full_text,
                        metadata={
                            "source": str(file_path),
                            "line_number": line_num,
                            "title": title,
                            **{k: v for k, v in data.items() if k not in ['id', 'title', 'abstract', 'content', 'text']}
                        },
                        language=language
                    )
                    
                    yield ProcessedDocument(
                        document_id=str(doc_id),
                        filename=f"{file_path.name}:line_{line_num}",
                        file_type="jsonl",
                        chunks=[chunk],
                        metadata={
                            "source": str(file_path),
                            "line_number": line_num,
                            "title": title,
                            "authors": data.get('authors', ''),
                        },
                        raw_text=full_text,
                        language=language
                    )
                    
                except json.JSONDecodeError as e:
                    logger.warning(f"Line {line_num}: JSON decode error - {e}")
                except Exception as e:
                    logger.warning(f"Line {line_num}: Error processing - {e}")


# Convenience function for quick processing
def process_document(
    source: Union[str, Path],
    ocr_enabled: bool = True,
    ocr_languages: List[str] = None,
    chunk_size: int = 512,
    timeout: int = 120,
    use_fallback: bool = True
) -> ProcessedDocument:
    """
    Convenience function to process a single document.
    
    Args:
        source: Path to document or URL
        ocr_enabled: Enable OCR for scanned documents
        ocr_languages: OCR language codes
        chunk_size: Target chunk size in tokens
        timeout: Timeout for URL fetching in seconds
        use_fallback: Use PyMuPDF fallback if Docling fails
        
    Returns:
        ProcessedDocument
    """
    processor = DoclingProcessor(
        ocr_enabled=ocr_enabled,
        ocr_languages=ocr_languages or ["heb", "eng"],
        chunk_size=chunk_size
    )
    return processor.process_file(source, timeout=timeout, use_fallback=use_fallback)


# URL-specific processing function
def process_url(
    url: str,
    timeout: int = 120,
    use_fallback: bool = True
) -> ProcessedDocument:
    """
    Convenience function specifically for processing URLs.
    Uses robust fetching with retries and fallback extraction.
    
    Args:
        url: The URL to process
        timeout: Timeout for fetching in seconds
        use_fallback: Use PyMuPDF fallback if Docling fails
        
    Returns:
        ProcessedDocument
    """
    processor = DoclingProcessor(
        ocr_enabled=True,
        ocr_languages=["heb", "eng"],
        chunk_size=512
    )
    return processor.process_file(url, timeout=timeout, use_fallback=use_fallback)
