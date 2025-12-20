# Multi-Format RAG System with Docling

This document describes the updated RAG (Retrieval-Augmented Generation) system that uses **Docling** for document processing, enabling support for a wide variety of file formats with Hebrew and English language support.

## Overview

The new RAG flow uses IBM's Docling library to process documents in multiple formats, extract text with OCR support, and create semantically meaningful chunks for vector search.

## Supported File Formats

### Documents
- **PDF** - With OCR support for scanned documents
- **DOCX/DOC** - Microsoft Word documents
- **PPTX/PPT** - PowerPoint presentations
- **XLSX/XLS** - Excel spreadsheets

### Text Formats
- **Markdown** (.md)
- **Plain Text** (.txt)
- **CSV** - Comma-separated values
- **JSON/JSONL** - JSON documents and newline-delimited JSON

### Web Formats
- **HTML/XHTML** - Web pages

### Images (with OCR)
- **PNG, JPEG, TIFF, BMP, WebP** - Image formats with text extraction via OCR

## Language Support

The system is optimized for **Hebrew** and **English** content:

- **OCR Languages**: Tesseract OCR with Hebrew (`heb`) and English (`eng`) support
- **Embedding Model**: Multilingual model (`paraphrase-multilingual-mpnet-base-v2`) for cross-language semantic search
- **Language Detection**: Automatic detection of Hebrew vs English content

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      File Upload / Input                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DoclingProcessor                              │
│  ┌───────────────┐  ┌───────────────┐  ┌───────────────┐       │
│  │   PDF Parser  │  │  DOCX Parser  │  │  Image + OCR  │  ...  │
│  └───────────────┘  └───────────────┘  └───────────────┘       │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │  DoclingDocument  │                        │
│                    └─────────┬─────────┘                        │
│                              │                                   │
│                    ┌─────────▼─────────┐                        │
│                    │   HybridChunker   │                        │
│                    └───────────────────┘                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      VectorIndexer                               │
│  ┌───────────────────────┐  ┌───────────────────────┐          │
│  │ Multilingual Embedder │  │     FAISS Index       │          │
│  └───────────────────────┘  └───────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       RAGPipeline                                │
│  ┌───────────────────────┐  ┌───────────────────────┐          │
│  │   Query Processing    │  │    LLM Generation     │          │
│  └───────────────────────┘  └───────────────────────┘          │
└─────────────────────────────────────────────────────────────────┘
```

## API Endpoints

### Document Management

#### Upload a Document
```http
POST /documents/upload
Content-Type: multipart/form-data

file: <binary file data>
```

**Response:**
```json
{
  "status": "success",
  "document_id": "abc123...",
  "filename": "document.pdf",
  "chunks_created": 15,
  "language": "heb",
  "file_type": "pdf"
}
```

#### Upload Multiple Documents
```http
POST /documents/upload-multiple
Content-Type: multipart/form-data

files: [<file1>, <file2>, ...]
```

#### List Documents
```http
GET /documents
```

**Response:**
```json
{
  "documents": [
    {
      "document_id": "abc123",
      "filename": "document.pdf",
      "file_type": "pdf",
      "language": "heb",
      "chunk_count": 15
    }
  ],
  "total_documents": 1,
  "total_chunks": 15
}
```

#### Delete a Document
```http
DELETE /documents/{document_id}
```

#### Clear All Documents
```http
DELETE /documents
```

#### Get Supported Formats
```http
GET /documents/supported-formats
```

### Query Endpoints

#### Answer a Query
```http
POST /answer
Content-Type: application/json

{
  "query": "מה זה בינה מלאכותית?",
  "top_k": 5
}
```

#### Stream Answer
```http
POST /stream
Content-Type: application/json

{
  "query": "What is machine learning?",
  "top_k": 5
}
```

## Configuration

### Environment Variables

```env
# Document Processing
OCR_ENABLED=true
OCR_LANGUAGES=heb,eng
CHUNK_SIZE=512
CHUNK_OVERLAP=50

# Language Settings
PRIMARY_LANGUAGE=hebrew
BILINGUAL_RESPONSES=false

# Embedding Model (multilingual recommended)
EMBEDDING_MODEL=sentence-transformers/paraphrase-multilingual-mpnet-base-v2

# Directories
UPLOADS_DIR=uploads
INDEX_DIR=index
```

## Installation

1. Install system dependencies for OCR:
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr tesseract-ocr-heb tesseract-ocr-eng

# Windows - Download installer from https://github.com/UB-Mannheim/tesseract/wiki
```

2. Install Python dependencies:
```bash
pip install -r requirements.txt
```

3. Download the multilingual embedding model:
```bash
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/paraphrase-multilingual-mpnet-base-v2')"
```

4. Start the server:
```bash
python main.py
```

## Usage Examples

### Python Client

```python
import requests

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8080/documents/upload",
        files={"file": f}
    )
print(response.json())

# Query in Hebrew
response = requests.post(
    "http://localhost:8080/answer",
    json={"query": "מה הנושא המרכזי של המסמך?", "top_k": 5}
)
print(response.json()["answer"])

# Query in English
response = requests.post(
    "http://localhost:8080/answer",
    json={"query": "What is the main topic?", "top_k": 5}
)
print(response.json()["answer"])
```

### cURL Examples

```bash
# Upload a PDF
curl -X POST "http://localhost:8080/documents/upload" \
  -F "file=@document.pdf"

# Query
curl -X POST "http://localhost:8080/answer" \
  -H "Content-Type: application/json" \
  -d '{"query": "מה הנושא המרכזי?", "top_k": 5}'

# List documents
curl "http://localhost:8080/documents"
```

## Performance Considerations

1. **OCR Processing**: OCR can be slow for large documents. Consider:
   - Disabling OCR for text-based PDFs
   - Using GPU acceleration when available

2. **Chunking**: The `CHUNK_SIZE` affects retrieval quality:
   - Larger chunks = more context but less precision
   - Smaller chunks = more precision but may lose context

3. **Embedding Model**: The multilingual model is larger than English-only models:
   - `paraphrase-multilingual-mpnet-base-v2`: ~280MB, 768 dimensions
   - Supports 50+ languages including Hebrew

## Troubleshooting

### OCR Not Working
- Ensure Tesseract is installed and in PATH
- Check that language packs are installed (`tesseract --list-langs`)

### Hebrew Text Garbled
- Ensure UTF-8 encoding throughout
- Check that the source document has proper Hebrew fonts

### Slow Processing
- Enable GPU acceleration (`FORCE_CPU=false`)
- Reduce `CHUNK_SIZE` for faster embedding
- Consider disabling OCR for text-based documents
