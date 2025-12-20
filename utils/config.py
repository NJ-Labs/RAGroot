"""
Configuration management for the RAG application.
Loads settings from .env file with sensible defaults.
"""
import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file from the project root
env_path = Path(__file__).parent.parent / ".env"
if env_path.exists():
    load_dotenv(env_path)
    print(f"✅ Loaded configuration from {env_path}")
else:
    print(f"⚠️  No .env file found at {env_path}, using defaults")
    print(f"   Create one by copying .env.example to .env")


class Config:
    """Application configuration with environment variable support."""
    
    # =============================================================================
    # DATA CONFIGURATION
    # =============================================================================
    
    # DATA_PATH is empty by default - the app only indexes user-uploaded documents
    DATA_PATH: str = os.getenv("DATA_PATH", "")
    INDEX_DIR: str = os.getenv("INDEX_DIR", "index")
    UPLOADS_DIR: str = os.getenv("UPLOADS_DIR", "uploads")
    
    # =============================================================================
    # DOCUMENT PROCESSING (DOCLING)
    # =============================================================================
    
    # OCR Configuration
    OCR_ENABLED: bool = os.getenv("OCR_ENABLED", "true").lower() == "true"
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "heb,eng")  # Comma-separated
    
    # Chunking Configuration
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "512"))  # Target chunk size in tokens
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))  # Overlap between chunks
    
    # Supported input types
    SUPPORTED_FORMATS: str = os.getenv(
        "SUPPORTED_FORMATS",
        ".pdf,.docx,.doc,.pptx,.xlsx,.html,.htm,.md,.txt,.csv,.png,.jpg,.jpeg,.tiff,.bmp,.webp,.json,.jsonl"
    )
    
    # =============================================================================
    # LLM MODEL CONFIGURATION
    # =============================================================================
    
    # LLM Backend: "llama_cpp" for local GGUF models, "vllm" for vLLM API
    LLM_BACKEND: str = os.getenv("LLM_BACKEND", "llama_cpp")
    
    # For llama.cpp backend (local GGUF models)
    MODEL_PATH: str = os.getenv("MODEL_PATH", "models/llama-model.gguf")
    N_THREADS: int = int(os.getenv("N_THREADS", "4"))
    N_CTX: int = int(os.getenv("N_CTX", "4096"))
    N_GPU_LAYERS: int = int(os.getenv("N_GPU_LAYERS", "0"))
    
    # For vLLM backend (OpenAI-compatible API)
    # VLLM_API_URL: The URL of the vLLM server (external microservice)
    VLLM_API_URL: str = os.getenv("VLLM_API_URL", "http://localhost:8000/v1")
    # VLLM_API_TOKEN: API token for vLLM authentication (empty for local/no auth)
    VLLM_API_TOKEN: str = os.getenv("VLLM_API_TOKEN", "")
    
    # Legacy aliases for backward compatibility
    LLM_API_BASE: str = os.getenv("LLM_API_BASE", os.getenv("VLLM_API_URL", "http://localhost:8000/v1"))
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL_NAME", "dicta-il/DictaLM-3.0-24B-Thinking-W4A16")
    
    # =============================================================================
    # EMBEDDING MODEL CONFIGURATION
    # =============================================================================
    
    # multilingual-e5-large: Best multilingual model with excellent Hebrew support
    # Requires "query: " and "passage: " prefixes for optimal performance
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-large")
    EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "16"))  # Smaller batch for larger model
    
    # Enable query/passage prefix for E5 models (improves retrieval quality)
    USE_E5_PREFIX: bool = os.getenv("USE_E5_PREFIX", "true").lower() == "true"
    
    # Local model cache directory (for offline operation)
    EMBEDDING_CACHE_DIR: str = os.getenv("EMBEDDING_CACHE_DIR", "models/embeddings")
    
    # Force local-only mode (no internet downloads)
    EMBEDDING_LOCAL_ONLY: bool = os.getenv("EMBEDDING_LOCAL_ONLY", "true").lower() == "true"
    
    # =============================================================================
    # IMAGE GENERATION (BONUS FEATURE)
    # =============================================================================
    
    IMAGE_API_PROVIDER: str = os.getenv("IMAGE_API_PROVIDER", "local")
    IMAGE_API_KEY: Optional[str] = os.getenv("IMAGE_API_KEY")
    IMAGE_MODEL_PATH: str = os.getenv("IMAGE_MODEL_PATH", "models/sdxl-turbo")
    IMAGE_MODEL_NAME: str = os.getenv("IMAGE_MODEL_NAME", "stabilityai/sdxl-turbo")
    IMAGE_INFERENCE_STEPS: int = int(os.getenv("IMAGE_INFERENCE_STEPS", "1"))
    IMAGE_GUIDANCE_SCALE: float = float(os.getenv("IMAGE_GUIDANCE_SCALE", "0.0"))
    # =============================================================================
    # RETRIEVAL CONFIGURATION
    # =============================================================================
    
    DEFAULT_TOP_K: int = int(os.getenv("DEFAULT_TOP_K", "5"))
    MAX_TOP_K: int = int(os.getenv("MAX_TOP_K", "20"))
    RERANK_MULTIPLIER: int = int(os.getenv("RERANK_MULTIPLIER", "3"))
    
    # =============================================================================
    # LLM GENERATION CONFIGURATION
    # =============================================================================
    
    MAX_TOKENS: int = int(os.getenv("MAX_TOKENS", "600"))
    TEMPERATURE: float = float(os.getenv("TEMPERATURE", "0.3"))
    TOP_P: float = float(os.getenv("TOP_P", "0.85"))
    REPEAT_PENALTY: float = float(os.getenv("REPEAT_PENALTY", "1.1"))
    
    # =============================================================================
    # SERVER CONFIGURATION
    # =============================================================================
    
    HOST: str = os.getenv("HOST", "0.0.0.0")
    PORT: int = int(os.getenv("PORT", "8080"))  # Changed to 8080 per assignment requirements
    LOG_LEVEL: str = os.getenv("LOG_LEVEL", "info").lower()
    ENABLE_CORS: bool = os.getenv("ENABLE_CORS", "true").lower() == "true"
    
    # =============================================================================
    # PERFORMANCE TUNING
    # =============================================================================
    
    FORCE_REBUILD: bool = os.getenv("FORCE_REBUILD", "false").lower() == "true"
    ENABLE_STREAMING: bool = os.getenv("ENABLE_STREAMING", "true").lower() == "true"
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", "120"))
    CACHE_EMBEDDINGS: bool = os.getenv("CACHE_EMBEDDINGS", "true").lower() == "true"
    
    # =============================================================================
    # LATEX PROCESSING
    # =============================================================================
    
    LATEX_PROCESSING_ENABLED: bool = os.getenv("LATEX_PROCESSING_ENABLED", "true").lower() == "true"
    
    # =============================================================================
    # LANGUAGE CONFIGURATION
    # =============================================================================
    
    # Primary language for the RAG system (affects prompts and processing)
    PRIMARY_LANGUAGE: str = os.getenv("PRIMARY_LANGUAGE", "hebrew")  # hebrew, english, or auto
    
    # Enable bilingual responses (answers in both Hebrew and English)
    BILINGUAL_RESPONSES: bool = os.getenv("BILINGUAL_RESPONSES", "false").lower() == "true"
    
    # =============================================================================
    # MODEL LOADING OPTIMIZATION
    # =============================================================================
    
    # Skip checking for sentence-transformers model updates on Hugging Face Hub
    # Set to True to skip version checks and speed up startup
    SKIP_CHECK_ST_UPDATES: bool = os.getenv("SKIP_CHECK_ST_UPDATES", "true").lower() == "true"
    
    # Force CPU usage for all models (embedding and LLM)
    # Set to True to disable GPU/CUDA even if available
    FORCE_CPU: bool = os.getenv("FORCE_CPU", "false").lower() == "true"
    
    @classmethod
    def validate(cls) -> bool:
        """Validate critical configuration values."""
        errors = []
        
        # DATA_PATH is optional - app works with user uploads only
        if cls.DATA_PATH and not Path(cls.DATA_PATH).exists():
            print(f"⚠️  DATA_PATH specified but not found: {cls.DATA_PATH}")
        
        # Check model path exists (only for llama_cpp backend)
        if cls.LLM_BACKEND == "llama_cpp" and not Path(cls.MODEL_PATH).exists():
            errors.append(f"MODEL_PATH not found: {cls.MODEL_PATH}")
        
        # Validate LLM backend
        valid_backends = ["llama_cpp", "vllm"]
        if cls.LLM_BACKEND not in valid_backends:
            errors.append(f"LLM_BACKEND must be one of {valid_backends}, got {cls.LLM_BACKEND}")
        
        # Validate numeric ranges
        if cls.N_THREADS < 1:
            errors.append(f"N_THREADS must be >= 1, got {cls.N_THREADS}")
        
        if cls.N_CTX < 512:
            errors.append(f"N_CTX must be >= 512, got {cls.N_CTX}")
        
        if not 0 <= cls.TEMPERATURE <= 2:
            errors.append(f"TEMPERATURE must be between 0 and 2, got {cls.TEMPERATURE}")
        
        if not 0 <= cls.TOP_P <= 1:
            errors.append(f"TOP_P must be between 0 and 1, got {cls.TOP_P}")
        
        if cls.DEFAULT_TOP_K < 1 or cls.DEFAULT_TOP_K > cls.MAX_TOP_K:
            errors.append(f"DEFAULT_TOP_K must be between 1 and {cls.MAX_TOP_K}, got {cls.DEFAULT_TOP_K}")
        
        # Validate image provider
        valid_providers = ["pollinations", "openai", "local"]
        if cls.IMAGE_API_PROVIDER not in valid_providers:
            errors.append(f"IMAGE_API_PROVIDER must be one of {valid_providers}, got {cls.IMAGE_API_PROVIDER}")
        
        if errors:
            print("\n❌ Configuration validation errors:")
            for error in errors:
                print(f"   - {error}")
            return False
        
        return True
    
    @classmethod
    def print_config(cls) -> None:
        """Print current configuration (masking sensitive values)."""
        print("\n" + "="*70)
        print("📋 Current Configuration")
        print("="*70)
        
        print("\n🗂️  Data & Paths:")
        print(f"   DATA_PATH:           {cls.DATA_PATH}")
        print(f"   INDEX_DIR:           {cls.INDEX_DIR}")
        print(f"   UPLOADS_DIR:         {cls.UPLOADS_DIR}")
        print(f"   MODEL_PATH:          {cls.MODEL_PATH}")
        
        print("\n📄 Document Processing (Docling):")
        print(f"   OCR_ENABLED:         {cls.OCR_ENABLED}")
        print(f"   OCR_LANGUAGES:       {cls.OCR_LANGUAGES}")
        print(f"   CHUNK_SIZE:          {cls.CHUNK_SIZE}")
        print(f"   CHUNK_OVERLAP:       {cls.CHUNK_OVERLAP}")
        print(f"   SUPPORTED_FORMATS:   {cls.SUPPORTED_FORMATS[:50]}...")
        
        print("\n🌍 Language Settings:")
        print(f"   PRIMARY_LANGUAGE:    {cls.PRIMARY_LANGUAGE}")
        print(f"   BILINGUAL_RESPONSES: {cls.BILINGUAL_RESPONSES}")
        
        print("\n🤖 LLM Model Settings:")
        print(f"   LLM_BACKEND:         {cls.LLM_BACKEND}")
        if cls.LLM_BACKEND == "llama_cpp":
            print(f"   MODEL_PATH:          {cls.MODEL_PATH}")
            print(f"   N_THREADS:           {cls.N_THREADS}")
            print(f"   N_CTX:               {cls.N_CTX}")
            print(f"   N_GPU_LAYERS:        {cls.N_GPU_LAYERS}")
        else:
            print(f"   VLLM_API_URL:        {cls.VLLM_API_URL}")
            print(f"   VLLM_API_TOKEN:      {'***' + cls.VLLM_API_TOKEN[-4:] if cls.VLLM_API_TOKEN else 'Not set (local mode)'}")
            print(f"   LLM_MODEL_NAME:      {cls.LLM_MODEL_NAME}")
        
        print("\n🔤 Embedding Settings:")
        print(f"   EMBEDDING_MODEL:     {cls.EMBEDDING_MODEL}")
        print(f"   EMBEDDING_CACHE_DIR: {cls.EMBEDDING_CACHE_DIR}")
        print(f"   EMBEDDING_LOCAL_ONLY: {cls.EMBEDDING_LOCAL_ONLY}")
        print(f"   USE_E5_PREFIX:       {cls.USE_E5_PREFIX}")
        
        print("\n🔍 Retrieval:")
        print(f"   DEFAULT_TOP_K:       {cls.DEFAULT_TOP_K}")
        print(f"   MAX_TOP_K:           {cls.MAX_TOP_K}")
        print(f"   RERANK_MULTIPLIER:   {cls.RERANK_MULTIPLIER}")
        
        print("\n💬 Generation:")
        print(f"   MAX_TOKENS:          {cls.MAX_TOKENS}")
        print(f"   TEMPERATURE:         {cls.TEMPERATURE}")
        print(f"   TOP_P:               {cls.TOP_P}")
        print(f"   REPEAT_PENALTY:      {cls.REPEAT_PENALTY}")
        
        print("\n🌐 Server:")
        print(f"   HOST:                {cls.HOST}")
        print(f"   PORT:                {cls.PORT}")
        print(f"   LOG_LEVEL:           {cls.LOG_LEVEL}")
        print(f"   ENABLE_CORS:         {cls.ENABLE_CORS}")
        
        print("\n🎨 Image Generation:")
        print(f"   PROVIDER:            {cls.IMAGE_API_PROVIDER}")
        print(f"   API_KEY:             {'***' + cls.IMAGE_API_KEY[-4:] if cls.IMAGE_API_KEY else 'Not set'}")
        print(f"   IMAGE_MODEL_PATH:    {cls.IMAGE_MODEL_PATH}")
        print(f"   IMAGE_MODEL_NAME:    {cls.IMAGE_MODEL_NAME}")
        
        print("\n⚙️  Performance:")
        print(f"   EMBEDDING_BATCH:     {cls.EMBEDDING_BATCH_SIZE}")
        print(f"   FORCE_REBUILD:       {cls.FORCE_REBUILD}")
        print(f"   ENABLE_STREAMING:    {cls.ENABLE_STREAMING}")
        print(f"   REQUEST_TIMEOUT:     {cls.REQUEST_TIMEOUT}s")
        print(f"   CACHE_EMBEDDINGS:    {cls.CACHE_EMBEDDINGS}")
        
        print("\n📐 LaTeX:")
        print(f"   PROCESSING_ENABLED:  {cls.LATEX_PROCESSING_ENABLED}")
        
        print("\n⚡ Model Loading:")
        print(f"   SKIP_CHECK_ST_UPDATES: {cls.SKIP_CHECK_ST_UPDATES}")
        print(f"   FORCE_CPU:           {cls.FORCE_CPU}")
        
        print("="*70 + "\n")


# Create global config instance
config = Config()

# Validate on import
if __name__ != "__main__":
    # Only validate when imported, not when running as script
    if not config.validate():
        print("\n⚠️  Warning: Configuration has validation errors!")
        print("   The application may not work correctly.")
        print("   Please check your .env file or environment variables.\n")


if __name__ == "__main__":
    """Run configuration validation and display when executed directly."""
    print("RAG Application - Configuration Checker")
    print("="*70)
    
    config.print_config()
    
    print("\n🔍 Validating configuration...")
    if config.validate():
        print("✅ Configuration is valid!\n")
    else:
        print("\n❌ Configuration has errors. Please fix them before running the application.\n")
        exit(1)
