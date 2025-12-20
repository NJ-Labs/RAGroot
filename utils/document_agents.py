"""
Document Processing Agents for Multi-Format Support.

Implements specialized agents for processing different document formats:
- Word (.docx, .doc)
- PowerPoint (.pptx, .ppt)
- Excel (.xlsx, .xls)
- HTML (.html, .htm)
- Markdown (.md)

Each agent extracts content while preserving structure and supports
Hebrew and English bilingual content.

Compatible with:
- Tesseract OCR for Hebrew text recognition
- Dicta-LM for Hebrew language processing
- multilingual-e5-large embedding model
"""

import os
import re
import logging
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from .smart_chunker import (
    SmartChunker,
    TextChunk,
    ChunkMetadata,
    ChunkingStrategy,
    WordDocumentChunker,
    PowerPointChunker,
    ExcelChunker,
    HTMLChunker,
    MarkdownChunker,
    RecursiveCharacterTextSplitter,
    get_optimal_config,
)

logger = logging.getLogger(__name__)


# =============================================================================
# BASE DOCUMENT AGENT
# =============================================================================

@dataclass
class ExtractedContent:
    """Container for extracted document content."""
    text: str
    paragraphs: List[Dict] = field(default_factory=list)
    tables: List[str] = field(default_factory=list)
    slides: List[Dict] = field(default_factory=list)
    sheets: List[Dict] = field(default_factory=list)
    sections: List[Dict] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)
    language: str = "unknown"
    page_count: int = 0


class BaseDocumentAgent(ABC):
    """
    Base class for document processing agents.
    
    Each agent specializes in extracting content from specific
    document formats while preserving structure for smart chunking.
    """
    
    SUPPORTED_EXTENSIONS: List[str] = []
    
    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 125,
        language_codes: List[str] = None
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.language_codes = language_codes or ["heb", "eng"]
        
        # Get optimal config for this agent's file types
        if self.SUPPORTED_EXTENSIONS:
            config = get_optimal_config(self.SUPPORTED_EXTENSIONS[0])
            self.chunk_size = config.get('chunk_size', chunk_size)
            self.chunk_overlap = config.get('chunk_overlap', chunk_overlap)
    
    @abstractmethod
    def can_process(self, file_path: Union[str, Path]) -> bool:
        """Check if this agent can process the given file."""
        pass
    
    @abstractmethod
    def extract_content(self, file_path: Union[str, Path]) -> ExtractedContent:
        """Extract structured content from the document."""
        pass
    
    @abstractmethod
    def chunk_content(self, content: ExtractedContent) -> List[TextChunk]:
        """Chunk the extracted content using appropriate strategy."""
        pass
    
    def process(self, file_path: Union[str, Path]) -> List[TextChunk]:
        """
        Full processing pipeline: extract and chunk.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of TextChunks with metadata
        """
        file_path = Path(file_path)
        
        if not self.can_process(file_path):
            raise ValueError(f"Cannot process file: {file_path}")
        
        logger.info(f"Processing {file_path.name} with {self.__class__.__name__}")
        
        # Extract structured content
        content = self.extract_content(file_path)
        
        # Chunk the content
        chunks = self.chunk_content(content)
        
        # Add source metadata to all chunks
        for chunk in chunks:
            chunk.metadata.element_type = chunk.metadata.element_type or "text"
        
        logger.info(f"Extracted {len(chunks)} chunks from {file_path.name}")
        
        return chunks
    
    def detect_language(self, text: str) -> str:
        """Detect if text is Hebrew or English."""
        if not text:
            return "unknown"
        
        hebrew_chars = sum(1 for c in text if '\u0590' <= c <= '\u05FF')
        latin_chars = sum(1 for c in text if c.isalpha() and c.isascii())
        
        total = hebrew_chars + latin_chars
        if total == 0:
            return "unknown"
        
        if hebrew_chars / total > 0.7:
            return "heb"
        elif hebrew_chars / total < 0.3:
            return "eng"
        return "mixed"


# =============================================================================
# WORD DOCUMENT AGENT
# =============================================================================

class WordDocumentAgent(BaseDocumentAgent):
    """
    Agent for processing Word documents (.docx, .doc).
    
    Extracts:
    - Paragraphs with style information (headings, body text)
    - Tables
    - Document metadata
    
    Uses structure-aware chunking to preserve heading hierarchy.
    """
    
    SUPPORTED_EXTENSIONS = ['.docx', '.doc']
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def extract_content(self, file_path: Union[str, Path]) -> ExtractedContent:
        """Extract content from Word document."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext == '.docx':
            return self._extract_docx(file_path)
        elif ext == '.doc':
            return self._extract_doc(file_path)
        
        raise ValueError(f"Unsupported Word format: {ext}")
    
    def _extract_docx(self, file_path: Path) -> ExtractedContent:
        """Extract from .docx using python-docx."""
        try:
            from docx import Document
            from docx.enum.text import WD_PARAGRAPH_ALIGNMENT
        except ImportError:
            logger.warning("python-docx not installed, using fallback extraction")
            return self._fallback_extract(file_path)
        
        doc = Document(file_path)
        
        paragraphs = []
        tables = []
        all_text = []
        
        # Extract paragraphs with styles
        for para in doc.paragraphs:
            text = para.text.strip()
            if not text:
                continue
            
            style_name = para.style.name.lower() if para.style else ""
            
            # Determine heading level
            level = 0
            if 'heading' in style_name:
                try:
                    level = int(re.search(r'\d+', style_name).group())
                except:
                    level = 1
            
            paragraphs.append({
                'text': text,
                'style': style_name,
                'level': level
            })
            all_text.append(text)
        
        # Extract tables
        for table in doc.tables:
            table_rows = []
            for row in table.rows:
                row_cells = [cell.text.strip() for cell in row.cells]
                table_rows.append(' | '.join(row_cells))
            
            if table_rows:
                table_text = '\n'.join(table_rows)
                tables.append(table_text)
                all_text.append(table_text)
        
        full_text = '\n\n'.join(all_text)
        
        # Extract metadata
        metadata = {}
        try:
            core_props = doc.core_properties
            metadata = {
                'title': core_props.title or '',
                'author': core_props.author or '',
                'subject': core_props.subject or '',
                'created': str(core_props.created) if core_props.created else '',
            }
        except Exception as e:
            logger.debug(f"Could not extract docx metadata: {e}")
        
        return ExtractedContent(
            text=full_text,
            paragraphs=paragraphs,
            tables=tables,
            metadata=metadata,
            language=self.detect_language(full_text)
        )
    
    def _extract_doc(self, file_path: Path) -> ExtractedContent:
        """Extract from legacy .doc format."""
        # Try using antiword or textract
        try:
            import textract
            text = textract.process(str(file_path)).decode('utf-8')
            
            return ExtractedContent(
                text=text,
                paragraphs=[{'text': p, 'style': '', 'level': 0} 
                           for p in text.split('\n\n') if p.strip()],
                language=self.detect_language(text)
            )
        except ImportError:
            pass
        
        # Try subprocess with antiword
        try:
            import subprocess
            result = subprocess.run(
                ['antiword', str(file_path)],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                text = result.stdout
                return ExtractedContent(
                    text=text,
                    paragraphs=[{'text': p, 'style': '', 'level': 0}
                               for p in text.split('\n\n') if p.strip()],
                    language=self.detect_language(text)
                )
        except Exception as e:
            logger.debug(f"antiword extraction failed: {e}")
        
        return self._fallback_extract(file_path)
    
    def _fallback_extract(self, file_path: Path) -> ExtractedContent:
        """Fallback extraction using Docling."""
        logger.info("Using Docling for Word extraction")
        
        try:
            from docling.document_converter import DocumentConverter
            
            converter = DocumentConverter()
            result = converter.convert(str(file_path))
            
            text = result.document.export_to_markdown()
            
            return ExtractedContent(
                text=text,
                paragraphs=[{'text': p, 'style': '', 'level': 0}
                           for p in text.split('\n\n') if p.strip()],
                language=self.detect_language(text)
            )
        except Exception as e:
            logger.error(f"Docling extraction failed: {e}")
            raise
    
    def chunk_content(self, content: ExtractedContent) -> List[TextChunk]:
        """Chunk Word document content."""
        chunker = WordDocumentChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        if content.paragraphs:
            chunks = chunker.chunk_with_structure(
                paragraphs=content.paragraphs,
                tables=content.tables
            )
        else:
            # Fall back to text chunking
            chunks = chunker.chunk(content.text)
        
        # Add language to all chunks
        for chunk in chunks:
            chunk.metadata.language = chunk.metadata.language or content.language
        
        return chunks


# =============================================================================
# POWERPOINT AGENT
# =============================================================================

class PowerPointAgent(BaseDocumentAgent):
    """
    Agent for processing PowerPoint presentations (.pptx, .ppt).
    
    Extracts:
    - Slide titles and content
    - Speaker notes
    - Tables and charts (as text)
    
    Chunks by slide for optimal retrieval.
    """
    
    SUPPORTED_EXTENSIONS = ['.pptx', '.ppt']
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def extract_content(self, file_path: Union[str, Path]) -> ExtractedContent:
        """Extract content from PowerPoint."""
        file_path = Path(file_path)
        ext = file_path.suffix.lower()
        
        if ext == '.pptx':
            return self._extract_pptx(file_path)
        elif ext == '.ppt':
            return self._extract_ppt(file_path)
        
        raise ValueError(f"Unsupported PowerPoint format: {ext}")
    
    def _extract_pptx(self, file_path: Path) -> ExtractedContent:
        """Extract from .pptx using python-pptx."""
        try:
            from pptx import Presentation
            from pptx.util import Inches
        except ImportError:
            logger.warning("python-pptx not installed, using fallback")
            return self._fallback_extract(file_path)
        
        prs = Presentation(file_path)
        
        slides = []
        all_text = []
        
        for slide_num, slide in enumerate(prs.slides, 1):
            slide_content = {
                'slide_number': slide_num,
                'title': '',
                'content': [],
                'notes': ''
            }
            
            # Extract title
            if slide.shapes.title:
                slide_content['title'] = slide.shapes.title.text.strip()
            
            # Extract text from all shapes
            for shape in slide.shapes:
                if hasattr(shape, 'text') and shape.text.strip():
                    if shape != slide.shapes.title:
                        slide_content['content'].append(shape.text.strip())
                
                # Extract table content
                if shape.has_table:
                    table_text = self._extract_table(shape.table)
                    slide_content['content'].append(table_text)
            
            # Extract speaker notes
            if slide.has_notes_slide:
                notes_frame = slide.notes_slide.notes_text_frame
                if notes_frame:
                    slide_content['notes'] = notes_frame.text.strip()
            
            # Combine content
            slide_content['content'] = '\n'.join(slide_content['content'])
            slides.append(slide_content)
            
            # Build full text
            slide_text = f"Slide {slide_num}: {slide_content['title']}\n{slide_content['content']}"
            if slide_content['notes']:
                slide_text += f"\nNotes: {slide_content['notes']}"
            all_text.append(slide_text)
        
        full_text = '\n\n'.join(all_text)
        
        return ExtractedContent(
            text=full_text,
            slides=slides,
            metadata={'slide_count': len(slides)},
            language=self.detect_language(full_text),
            page_count=len(slides)
        )
    
    def _extract_table(self, table) -> str:
        """Extract text from PowerPoint table."""
        rows = []
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]
            rows.append(' | '.join(cells))
        return '\n'.join(rows)
    
    def _extract_ppt(self, file_path: Path) -> ExtractedContent:
        """Extract from legacy .ppt format."""
        return self._fallback_extract(file_path)
    
    def _fallback_extract(self, file_path: Path) -> ExtractedContent:
        """Fallback extraction using Docling."""
        try:
            from docling.document_converter import DocumentConverter
            
            converter = DocumentConverter()
            result = converter.convert(str(file_path))
            
            text = result.document.export_to_markdown()
            
            # Parse into pseudo-slides based on headings
            slides = []
            current_slide = {'slide_number': 1, 'title': '', 'content': '', 'notes': ''}
            
            for line in text.split('\n'):
                if line.startswith('# '):
                    if current_slide['title'] or current_slide['content']:
                        slides.append(current_slide)
                    current_slide = {
                        'slide_number': len(slides) + 1,
                        'title': line[2:].strip(),
                        'content': '',
                        'notes': ''
                    }
                else:
                    current_slide['content'] += line + '\n'
            
            if current_slide['title'] or current_slide['content']:
                slides.append(current_slide)
            
            return ExtractedContent(
                text=text,
                slides=slides,
                language=self.detect_language(text),
                page_count=len(slides)
            )
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            raise
    
    def chunk_content(self, content: ExtractedContent) -> List[TextChunk]:
        """Chunk PowerPoint by slides."""
        chunker = PowerPointChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        if content.slides:
            chunks = chunker.chunk_slides(content.slides)
        else:
            chunks = chunker.chunk(content.text)
        
        for chunk in chunks:
            chunk.metadata.language = chunk.metadata.language or content.language
        
        return chunks


# =============================================================================
# EXCEL AGENT
# =============================================================================

class ExcelAgent(BaseDocumentAgent):
    """
    Agent for processing Excel spreadsheets (.xlsx, .xls).
    
    Extracts:
    - Sheet names and data
    - Headers and row data
    - Formulas as computed values
    
    Chunks by row groups to maintain tabular context.
    """
    
    SUPPORTED_EXTENSIONS = ['.xlsx', '.xls']
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def extract_content(self, file_path: Union[str, Path]) -> ExtractedContent:
        """Extract content from Excel."""
        file_path = Path(file_path)
        
        try:
            import openpyxl
            return self._extract_xlsx(file_path)
        except ImportError:
            logger.warning("openpyxl not installed, trying pandas")
        
        try:
            import pandas as pd
            return self._extract_with_pandas(file_path)
        except ImportError:
            logger.warning("pandas not installed, using fallback")
        
        return self._fallback_extract(file_path)
    
    def _extract_xlsx(self, file_path: Path) -> ExtractedContent:
        """Extract using openpyxl."""
        import openpyxl
        
        wb = openpyxl.load_workbook(file_path, data_only=True)
        
        sheets = []
        all_text = []
        
        for sheet_name in wb.sheetnames:
            ws = wb[sheet_name]
            
            rows = []
            headers = []
            
            for row_idx, row in enumerate(ws.iter_rows(values_only=True)):
                # Filter out completely empty rows
                if not any(cell is not None for cell in row):
                    continue
                
                row_data = [str(cell) if cell is not None else '' for cell in row]
                
                if row_idx == 0:
                    headers = row_data
                else:
                    rows.append(row_data)
            
            if headers or rows:
                sheet_data = {
                    'name': sheet_name,
                    'headers': headers,
                    'rows': rows
                }
                sheets.append(sheet_data)
                
                # Build text representation
                text_lines = [f"Sheet: {sheet_name}"]
                if headers:
                    text_lines.append(' | '.join(headers))
                for row in rows:
                    text_lines.append(' | '.join(row))
                all_text.append('\n'.join(text_lines))
        
        full_text = '\n\n'.join(all_text)
        
        return ExtractedContent(
            text=full_text,
            sheets=sheets,
            metadata={'sheet_count': len(sheets)},
            language=self.detect_language(full_text)
        )
    
    def _extract_with_pandas(self, file_path: Path) -> ExtractedContent:
        """Extract using pandas."""
        import pandas as pd
        
        # Read all sheets
        excel_file = pd.ExcelFile(file_path)
        
        sheets = []
        all_text = []
        
        for sheet_name in excel_file.sheet_names:
            df = pd.read_excel(excel_file, sheet_name=sheet_name)
            
            headers = df.columns.tolist()
            rows = df.values.tolist()
            
            sheet_data = {
                'name': sheet_name,
                'headers': [str(h) for h in headers],
                'rows': [[str(cell) if pd.notna(cell) else '' for cell in row] for row in rows]
            }
            sheets.append(sheet_data)
            
            # Build text
            text_lines = [f"Sheet: {sheet_name}"]
            text_lines.append(' | '.join(str(h) for h in headers))
            for row in rows:
                text_lines.append(' | '.join(str(c) if pd.notna(c) else '' for c in row))
            all_text.append('\n'.join(text_lines))
        
        full_text = '\n\n'.join(all_text)
        
        return ExtractedContent(
            text=full_text,
            sheets=sheets,
            language=self.detect_language(full_text)
        )
    
    def _fallback_extract(self, file_path: Path) -> ExtractedContent:
        """Fallback extraction."""
        try:
            from docling.document_converter import DocumentConverter
            
            converter = DocumentConverter()
            result = converter.convert(str(file_path))
            
            text = result.document.export_to_markdown()
            
            return ExtractedContent(
                text=text,
                language=self.detect_language(text)
            )
        except Exception as e:
            logger.error(f"Fallback extraction failed: {e}")
            raise
    
    def chunk_content(self, content: ExtractedContent) -> List[TextChunk]:
        """Chunk Excel by sheets and row groups."""
        chunker = ExcelChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        if content.sheets:
            chunks = chunker.chunk_sheets(content.sheets)
        else:
            # Fall back to text chunking
            fallback = RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap
            )
            chunks = fallback.chunk(content.text)
        
        for chunk in chunks:
            chunk.metadata.language = chunk.metadata.language or content.language
        
        return chunks


# =============================================================================
# HTML AGENT
# =============================================================================

class HTMLAgent(BaseDocumentAgent):
    """
    Agent for processing HTML documents (.html, .htm).
    
    Extracts:
    - Text content from semantic elements
    - Heading hierarchy
    - Lists and tables
    
    Preserves DOM structure for smart chunking.
    """
    
    SUPPORTED_EXTENSIONS = ['.html', '.htm']
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def extract_content(self, file_path: Union[str, Path]) -> ExtractedContent:
        """Extract content from HTML."""
        file_path = Path(file_path)
        
        try:
            from bs4 import BeautifulSoup
            return self._extract_with_beautifulsoup(file_path)
        except ImportError:
            logger.warning("BeautifulSoup not installed, using fallback")
            return self._fallback_extract(file_path)
    
    def _extract_with_beautifulsoup(self, file_path: Path) -> ExtractedContent:
        """Extract using BeautifulSoup."""
        from bs4 import BeautifulSoup
        
        # Read file with encoding detection
        content = self._read_file_with_encoding(file_path)
        
        soup = BeautifulSoup(content, 'lxml')
        
        # Remove script, style, and other non-content elements
        for element in soup(['script', 'style', 'nav', 'footer', 'header', 'aside', 'noscript']):
            element.decompose()
        
        sections = []
        all_text = []
        
        # Extract title
        title = soup.title.string if soup.title else ''
        metadata = {'title': title}
        
        # Find main content area
        main = soup.find('main') or soup.find('article') or soup.find('body')
        
        if main:
            # Extract structured content
            for element in main.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'div', 'section', 'article', 'ul', 'ol', 'table']):
                tag = element.name
                
                # Get text content
                if tag == 'table':
                    text = self._extract_html_table(element)
                elif tag in ['ul', 'ol']:
                    items = [li.get_text(strip=True) for li in element.find_all('li')]
                    text = '\n'.join(f"• {item}" for item in items if item)
                else:
                    text = element.get_text(strip=True)
                
                if text:
                    sections.append({
                        'tag': tag,
                        'content': text
                    })
                    all_text.append(text)
        
        full_text = '\n\n'.join(all_text)
        
        return ExtractedContent(
            text=full_text,
            sections=sections,
            metadata=metadata,
            language=self.detect_language(full_text)
        )
    
    def _read_file_with_encoding(self, file_path: Path) -> str:
        """Read file with automatic encoding detection."""
        encodings = ['utf-8', 'utf-8-sig', 'windows-1255', 'iso-8859-8', 'windows-1252', 'iso-8859-1']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue
        
        # Last resort: read as binary and decode with errors ignored
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    
    def _extract_html_table(self, table) -> str:
        """Extract text from HTML table."""
        rows = []
        for tr in table.find_all('tr'):
            cells = []
            for td in tr.find_all(['td', 'th']):
                cells.append(td.get_text(strip=True))
            if cells:
                rows.append(' | '.join(cells))
        return '\n'.join(rows)
    
    def _fallback_extract(self, file_path: Path) -> ExtractedContent:
        """Fallback extraction."""
        # Simple regex-based extraction
        content = self._read_file_with_encoding(file_path)
        
        # Remove tags
        text = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        return ExtractedContent(
            text=text,
            language=self.detect_language(text)
        )
    
    def chunk_content(self, content: ExtractedContent) -> List[TextChunk]:
        """Chunk HTML preserving structure."""
        chunker = HTMLChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        if content.sections:
            chunks = chunker.chunk_html(content.sections)
        else:
            chunks = chunker.chunk(content.text)
        
        for chunk in chunks:
            chunk.metadata.language = chunk.metadata.language or content.language
        
        return chunks


# =============================================================================
# MARKDOWN AGENT
# =============================================================================

class MarkdownAgent(BaseDocumentAgent):
    """
    Agent for processing Markdown documents (.md).
    
    Extracts:
    - Heading hierarchy
    - Code blocks
    - Lists and tables
    
    Preserves markdown structure for smart chunking.
    """
    
    SUPPORTED_EXTENSIONS = ['.md', '.markdown']
    
    def can_process(self, file_path: Union[str, Path]) -> bool:
        ext = Path(file_path).suffix.lower()
        return ext in self.SUPPORTED_EXTENSIONS
    
    def extract_content(self, file_path: Union[str, Path]) -> ExtractedContent:
        """Extract content from Markdown."""
        file_path = Path(file_path)
        
        # Read file
        text = self._read_file_with_encoding(file_path)
        
        # Parse structure
        sections = self._parse_markdown(text)
        
        # Extract metadata from frontmatter if present
        metadata, text_without_frontmatter = self._extract_frontmatter(text)
        
        return ExtractedContent(
            text=text_without_frontmatter,
            sections=sections,
            metadata=metadata,
            language=self.detect_language(text)
        )
    
    def _read_file_with_encoding(self, file_path: Path) -> str:
        """Read file with encoding detection."""
        encodings = ['utf-8', 'utf-8-sig', 'windows-1255', 'iso-8859-8']
        
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    return f.read()
            except (UnicodeDecodeError, LookupError):
                continue
        
        with open(file_path, 'rb') as f:
            return f.read().decode('utf-8', errors='ignore')
    
    def _parse_markdown(self, text: str) -> List[Dict]:
        """Parse markdown into sections."""
        sections = []
        current_section = {'title': None, 'level': 0, 'content': []}
        
        lines = text.split('\n')
        i = 0
        
        while i < len(lines):
            line = lines[i]
            
            # Check for ATX headings (# Heading)
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                # Save current section
                if current_section['content']:
                    sections.append(current_section)
                
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                current_section = {
                    'title': title,
                    'level': level,
                    'content': []
                }
            else:
                # Add content
                if line.strip():
                    current_section['content'].append(line)
            
            i += 1
        
        # Don't forget last section
        if current_section['content'] or current_section['title']:
            sections.append(current_section)
        
        return sections
    
    def _extract_frontmatter(self, text: str) -> Tuple[Dict, str]:
        """Extract YAML frontmatter if present."""
        metadata = {}
        
        if text.startswith('---'):
            parts = text.split('---', 2)
            if len(parts) >= 3:
                frontmatter = parts[1].strip()
                text = parts[2].strip()
                
                # Simple YAML parsing
                for line in frontmatter.split('\n'):
                    if ':' in line:
                        key, value = line.split(':', 1)
                        metadata[key.strip()] = value.strip()
        
        return metadata, text
    
    def chunk_content(self, content: ExtractedContent) -> List[TextChunk]:
        """Chunk Markdown preserving structure."""
        chunker = MarkdownChunker(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            preserve_code_blocks=True
        )
        
        chunks = chunker.chunk(content.text)
        
        for chunk in chunks:
            chunk.metadata.language = chunk.metadata.language or content.language
        
        return chunks


# =============================================================================
# AGENT FACTORY
# =============================================================================

class DocumentAgentFactory:
    """
    Factory for creating appropriate document agents.
    
    Automatically selects the right agent based on file type.
    """
    
    # Map extensions to agents
    AGENT_MAP = {
        '.docx': WordDocumentAgent,
        '.doc': WordDocumentAgent,
        '.pptx': PowerPointAgent,
        '.ppt': PowerPointAgent,
        '.xlsx': ExcelAgent,
        '.xls': ExcelAgent,
        '.html': HTMLAgent,
        '.htm': HTMLAgent,
        '.md': MarkdownAgent,
        '.markdown': MarkdownAgent,
    }
    
    @classmethod
    def get_agent(
        cls,
        file_path: Union[str, Path],
        chunk_size: int = 250,
        chunk_overlap: int = 125
    ) -> Optional[BaseDocumentAgent]:
        """
        Get the appropriate agent for a file.
        
        Args:
            file_path: Path to the document
            chunk_size: Target chunk size
            chunk_overlap: Chunk overlap
            
        Returns:
            Appropriate document agent or None if unsupported
        """
        ext = Path(file_path).suffix.lower()
        
        agent_class = cls.AGENT_MAP.get(ext)
        
        if agent_class:
            return agent_class(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        
        return None
    
    @classmethod
    def is_supported(cls, file_path: Union[str, Path]) -> bool:
        """Check if file type is supported."""
        ext = Path(file_path).suffix.lower()
        return ext in cls.AGENT_MAP
    
    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported extensions."""
        return list(cls.AGENT_MAP.keys())


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class DocumentProcessingOrchestrator:
    """
    Orchestrator for multi-format document processing.
    
    Coordinates agents to process documents of any supported format.
    """
    
    def __init__(
        self,
        chunk_size: int = 250,
        chunk_overlap: int = 125,
        use_semantic_chunking: bool = False
    ):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.use_semantic = use_semantic_chunking
        
        # Smart chunker for fallback
        self.smart_chunker = SmartChunker(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            use_semantic_for_coherence=use_semantic_chunking
        )
    
    def process(self, file_path: Union[str, Path]) -> List[TextChunk]:
        """
        Process any supported document.
        
        Args:
            file_path: Path to the document
            
        Returns:
            List of TextChunks
        """
        file_path = Path(file_path)
        
        # Try to get specialized agent
        agent = DocumentAgentFactory.get_agent(
            file_path,
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )
        
        if agent:
            try:
                return agent.process(file_path)
            except Exception as e:
                logger.warning(f"Agent processing failed, using fallback: {e}")
        
        # Fallback: read as text and use smart chunker
        return self._fallback_process(file_path)
    
    def _fallback_process(self, file_path: Path) -> List[TextChunk]:
        """Fallback processing using smart chunker."""
        # Try to read file as text
        encodings = ['utf-8', 'utf-8-sig', 'windows-1255', 'iso-8859-8', 'windows-1252']
        
        text = None
        for encoding in encodings:
            try:
                with open(file_path, 'r', encoding=encoding) as f:
                    text = f.read()
                break
            except (UnicodeDecodeError, LookupError):
                continue
        
        if text is None:
            with open(file_path, 'rb') as f:
                text = f.read().decode('utf-8', errors='ignore')
        
        return self.smart_chunker.chunk(
            text,
            file_type=file_path.suffix.lstrip('.')
        )
    
    def get_supported_formats(self) -> Dict[str, List[str]]:
        """Get supported formats by category."""
        return {
            'documents': ['.docx', '.doc'],
            'presentations': ['.pptx', '.ppt'],
            'spreadsheets': ['.xlsx', '.xls'],
            'web': ['.html', '.htm'],
            'markdown': ['.md', '.markdown'],
        }


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def process_document(
    file_path: Union[str, Path],
    chunk_size: int = 250,
    chunk_overlap: int = 125
) -> List[TextChunk]:
    """
    Convenience function to process any supported document.
    
    Args:
        file_path: Path to document
        chunk_size: Target chunk size (250 optimal per Chroma Research)
        chunk_overlap: Overlap (125 = 50% for max recall)
    
    Returns:
        List of TextChunks
    """
    orchestrator = DocumentProcessingOrchestrator(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return orchestrator.process(file_path)


def get_agent_for_file(file_path: Union[str, Path]) -> Optional[BaseDocumentAgent]:
    """Get the appropriate agent for a file."""
    return DocumentAgentFactory.get_agent(file_path)


def is_format_supported(file_path: Union[str, Path]) -> bool:
    """Check if a file format is supported."""
    return DocumentAgentFactory.is_supported(file_path)

