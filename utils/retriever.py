import logging
from typing import List, Dict, AsyncGenerator, Optional
import os

from .latex_utils import LaTeXMathHandler
from .document_processor import DocumentChunk

logger = logging.getLogger(__name__)


class LLMBackend:
    """Abstract LLM backend interface."""
    
    def generate(self, prompt: str, **kwargs) -> str:
        raise NotImplementedError
    
    def stream(self, prompt: str, **kwargs):
        raise NotImplementedError


class LlamaCppBackend(LLMBackend):
    """Backend using llama.cpp for local GGUF models."""
    
    def __init__(self, config):
        from llama_cpp import Llama
        from pathlib import Path
        
        model_path = Path(config.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {config.MODEL_PATH}\n"
                f"Please ensure the model file exists."
            )
        
        logger.info(f"Loading LLM from {config.MODEL_PATH}")
        logger.info(f"Model size: {model_path.stat().st_size / (1024**3):.2f} GB")
        
        n_gpu_layers = 0 if config.FORCE_CPU else config.N_GPU_LAYERS
        if n_gpu_layers > 0:
            logger.info(f"GPU acceleration enabled with {n_gpu_layers} layers")
        else:
            logger.info("Running on CPU")
        
        self.llm = Llama(
            model_path=str(model_path),
            n_ctx=config.N_CTX,
            n_threads=config.N_THREADS,
            n_gpu_layers=n_gpu_layers,
            verbose=False,
            n_batch=512,
        )
        self.config = config
        logger.info("LLM loaded successfully (llama.cpp backend)")
    
    def test_connection(self) -> dict:
        """
        Test the llama.cpp model is working.
        
        Returns:
            dict with 'success', 'message', and optionally 'test_response'
        """
        result = {
            "success": False,
            "message": "",
            "backend": "llama.cpp",
            "model_path": str(self.config.MODEL_PATH)
        }
        
        try:
            # Test a simple completion
            test_response = self.llm(
                "Say 'OK' if you can hear me.",
                max_tokens=10,
                temperature=0.1,
                echo=False
            )
            
            if test_response['choices'] and test_response['choices'][0]['text']:
                result["test_response"] = test_response['choices'][0]['text'].strip()[:50]
                result["success"] = True
                result["message"] = "llama.cpp model is loaded and responding"
            else:
                result["message"] = "Model responded but returned empty output"
                result["success"] = True
                
        except Exception as e:
            result["message"] = f"Model test failed: {e}"
            result["error"] = str(e)
        
        return result
    
    def generate(self, prompt: str, **kwargs) -> str:
        response = self.llm(
            prompt,
            max_tokens=kwargs.get('max_tokens', self.config.MAX_TOKENS),
            temperature=kwargs.get('temperature', self.config.TEMPERATURE),
            top_p=kwargs.get('top_p', self.config.TOP_P),
            repeat_penalty=kwargs.get('repeat_penalty', self.config.REPEAT_PENALTY),
            stop=kwargs.get('stop', []),
            echo=False
        )
        return response['choices'][0]['text'].strip()
    
    def stream(self, prompt: str, **kwargs):
        for chunk in self.llm(
            prompt,
            max_tokens=kwargs.get('max_tokens', self.config.MAX_TOKENS),
            temperature=kwargs.get('temperature', self.config.TEMPERATURE),
            top_p=kwargs.get('top_p', self.config.TOP_P),
            repeat_penalty=kwargs.get('repeat_penalty', self.config.REPEAT_PENALTY),
            stop=kwargs.get('stop', []),
            stream=True
        ):
            yield chunk['choices'][0]['text']


class VLLMBackend(LLMBackend):
    """Backend using vLLM OpenAI-compatible API (for DictaLM and other models).
    
    Connects to an external vLLM microservice. Configure via environment variables:
    - VLLM_API_URL: The URL of the vLLM server (default: http://localhost:8000/v1)
    - VLLM_API_TOKEN: API token for authentication (default: empty for local/no auth)
    - LLM_MODEL_NAME: The model name to use for inference
    """
    
    def __init__(self, config):
        from openai import OpenAI
        
        # Use VLLM_API_URL with fallback to legacy LLM_API_BASE
        self.api_base = getattr(config, 'VLLM_API_URL', None) or \
                        getattr(config, 'LLM_API_BASE', None) or \
                        os.getenv('VLLM_API_URL', os.getenv('LLM_API_BASE', 'http://localhost:8000/v1'))
        
        # Get API token (empty string means no auth required - typical for local vLLM)
        self.api_token = getattr(config, 'VLLM_API_TOKEN', None) or \
                         os.getenv('VLLM_API_TOKEN', '')
        
        self.model_name = getattr(config, 'LLM_MODEL_NAME', os.getenv('LLM_MODEL_NAME', 'dicta-il/DictaLM-3.0-24B-Thinking-W4A16'))
        
        # Use token if provided, otherwise use "EMPTY" for local vLLM without auth
        api_key = self.api_token if self.api_token else "EMPTY"
        
        self.client = OpenAI(
            api_key=api_key,
            base_url=self.api_base,
        )
        self.config = config
        
        auth_status = "with token" if self.api_token else "no auth"
        logger.info(f"vLLM backend initialized: {self.api_base}, model: {self.model_name} ({auth_status})")
    
    def test_connection(self) -> dict:
        """
        Test connectivity to the vLLM endpoint.
        
        Returns:
            dict with 'success', 'message', and optionally 'models' and 'test_response'
        """
        import httpx
        
        result = {
            "success": False,
            "message": "",
            "endpoint": self.api_base,
            "model": self.model_name
        }
        
        try:
            # Step 1: Check if the endpoint is reachable
            base_url = self.api_base.rstrip('/v1').rstrip('/')
            health_url = f"{base_url}/health"
            
            try:
                with httpx.Client(timeout=10.0) as client:
                    health_response = client.get(health_url)
                    if health_response.status_code == 200:
                        result["health_check"] = "passed"
                    else:
                        result["health_check"] = f"returned {health_response.status_code}"
            except Exception as health_err:
                result["health_check"] = f"not available ({type(health_err).__name__})"
            
            # Step 2: Check available models
            try:
                models = self.client.models.list()
                available_models = [m.id for m in models.data] if models.data else []
                result["available_models"] = available_models
                
                if self.model_name not in available_models and available_models:
                    result["model_warning"] = f"Configured model '{self.model_name}' not in available models: {available_models}"
            except Exception as model_err:
                result["models_error"] = str(model_err)
            
            # Step 3: Test a simple chat completion (required for instruct models)
            try:
                test_response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": "Say 'OK' if you can hear me."}
                    ],
                    max_tokens=10,
                    temperature=0.1,
                )
                
                if test_response.choices and test_response.choices[0].message.content:
                    result["test_response"] = test_response.choices[0].message.content.strip()[:50]
                    result["success"] = True
                    result["message"] = "LLM endpoint is connected and responding"
                else:
                    result["message"] = "LLM responded but returned empty output"
                    result["success"] = True  # Still consider it a success if we got a response
                    
            except Exception as completion_err:
                result["completion_error"] = str(completion_err)
                result["message"] = f"LLM chat completion test failed: {completion_err}"
            
            # Step 4: Test streaming capability with chat completions
            if result["success"]:
                try:
                    stream = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": "Count: 1, 2, 3"}
                        ],
                        max_tokens=10,
                        temperature=0.1,
                        stream=True,
                    )
                    stream_tokens = 0
                    for chunk in stream:
                        if chunk.choices:
                            delta = chunk.choices[0].delta
                            if delta and delta.content:
                                stream_tokens += 1
                    
                    result["streaming_tokens"] = stream_tokens
                    if stream_tokens == 0:
                        result["streaming_warning"] = "Streaming returned 0 tokens - may need to use non-streaming mode"
                    else:
                        result["streaming_status"] = "working"
                        
                except Exception as stream_err:
                    result["streaming_error"] = str(stream_err)
                    result["streaming_warning"] = "Streaming not available, will use fallback"
                    
        except Exception as e:
            result["message"] = f"Connection failed: {e}"
            result["error"] = str(e)
        
        return result
    
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate response using chat completions API (required for instruct models)."""
        stop_sequences = kwargs.get('stop', [])
        
        try:
            # Use chat completions endpoint - required for instruct models like DictaLM
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', self.config.MAX_TOKENS),
                temperature=kwargs.get('temperature', self.config.TEMPERATURE),
                top_p=kwargs.get('top_p', self.config.TOP_P),
                stop=stop_sequences if stop_sequences else None,
            )
            
            if response.choices and response.choices[0].message.content:
                result = response.choices[0].message.content.strip()
                logger.debug(f"vLLM generate returned {len(result)} chars")
                return result
            else:
                finish_reason = response.choices[0].finish_reason if response.choices else 'no choices'
                logger.warning(f"vLLM generate returned empty response. Finish reason: {finish_reason}")
                return ""
                
        except Exception as e:
            logger.error(f"vLLM generate error: {e}", exc_info=True)
            raise
    
    def stream(self, prompt: str, **kwargs):
        """Stream response using chat completions API (required for instruct models)."""
        stop_sequences = kwargs.get('stop', [])
        
        try:
            # Use chat completions endpoint - required for instruct models like DictaLM
            stream = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=kwargs.get('max_tokens', self.config.MAX_TOKENS),
                temperature=kwargs.get('temperature', self.config.TEMPERATURE),
                top_p=kwargs.get('top_p', self.config.TOP_P),
                stop=stop_sequences if stop_sequences else None,
                stream=True,
            )
            
            token_count = 0
            for chunk in stream:
                if chunk.choices:
                    delta = chunk.choices[0].delta
                    if delta and delta.content:
                        token_count += 1
                        yield delta.content
            
            if token_count == 0:
                logger.warning("vLLM streaming returned 0 tokens")
                
        except Exception as e:
            logger.error(f"vLLM streaming error: {e}", exc_info=True)
            raise


class RAGPipeline:
    """Retrieval-Augmented Generation pipeline with Hebrew/English support."""
    
    def __init__(self, indexer):
        from .config import config
        from pathlib import Path
        
        self.indexer = indexer
        self.latex_handler = LaTeXMathHandler(preserve_structure=True)
        self.primary_language = config.PRIMARY_LANGUAGE
        self.bilingual = config.BILINGUAL_RESPONSES
        self.config = config
        
        # Determine which backend to use
        llm_backend = getattr(config, 'LLM_BACKEND', os.getenv('LLM_BACKEND', 'llama_cpp'))
        
        if llm_backend == 'vllm':
            logger.info("Using vLLM backend (OpenAI-compatible API)")
            self.llm_backend = VLLMBackend(config)
        else:
            logger.info("Using llama.cpp backend (local GGUF model)")
            self.llm_backend = LlamaCppBackend(config)
    
    def _detect_language(self, text: str) -> str:
        """Detect if text is primarily Hebrew or English based on character analysis."""
        # Hebrew Unicode range: \u0590-\u05FF (Hebrew block)
        hebrew_chars = sum(1 for char in text if '\u0590' <= char <= '\u05FF')
        latin_chars = sum(1 for char in text if ('a' <= char.lower() <= 'z'))
        
        # If more Hebrew characters than Latin, consider it Hebrew
        if hebrew_chars > latin_chars:
            return "hebrew"
        return "english"
    
    def _build_prompt(self, query: str, context_results: List[Dict]) -> str:
        """Build prompt for LLM with retrieved context from document chunks."""
        
        # Process query to expand LaTeX for better understanding
        processed_query = self.latex_handler.process_text(query)
        
        # Detect query language
        query_language = self._detect_language(query)
        
        # Build context from retrieved chunks with better formatting
        context_parts = []
        for i, result in enumerate(context_results, 1):
            chunk = result.get('chunk')
            doc = result.get('document', {})
            
            if chunk:
                # New chunk-based format
                content = chunk.content if isinstance(chunk, DocumentChunk) else chunk.get('content', '')
                section = chunk.section_title if isinstance(chunk, DocumentChunk) else chunk.get('section_title', '')
                chunk_metadata = chunk.metadata if isinstance(chunk, DocumentChunk) else chunk.get('metadata', {})
                
                processed_content = self.latex_handler.process_text(content)
                
                # Build context entry
                entry_parts = [f"[{i}]"]
                
                # Add document info
                if doc.get('filename'):
                    entry_parts.append(f"Source: {doc.get('filename')}")
                if doc.get('document_id'):
                    entry_parts.append(f"ID: {doc.get('document_id')}")
                if section:
                    entry_parts.append(f"Section: {section}")
                if chunk_metadata.get('title'):
                    entry_parts.append(f"Title: {chunk_metadata.get('title')}")
                if chunk_metadata.get('authors'):
                    entry_parts.append(f"Authors: {chunk_metadata.get('authors')}")
                
                entry_parts.append(f"Content: {processed_content}")
                
                context_parts.append("\n".join(entry_parts))
            else:
                # Legacy document format fallback
                doc = result.get('document', result)
                processed_abstract = self.latex_handler.process_text(doc.get('abstract', ''))
                processed_title = self.latex_handler.process_text(doc.get('title', ''))
                authors = doc.get('authors', 'Unknown')
                
                context_parts.append(
                    f"[{i}] ID: {doc.get('id', 'Unknown')}\n"
                    f"Title: {processed_title}\n"
                    f"Authors: {authors}\n"
                    f"Abstract: {processed_abstract}\n"
                )
        
        context = "\n\n".join(context_parts)
        
        # Build prompt based on detected language
        if query_language == "hebrew":
            prompt = self._build_hebrew_prompt(processed_query, context)
        else:
            prompt = self._build_english_prompt(processed_query, context)
        
        return prompt
    
    def _build_hebrew_prompt(self, processed_query: str, context: str) -> str:
        """Build prompt in Hebrew for Hebrew queries."""
        return f"""אתה עוזר מחקר מומחה. המשימה שלך היא לספק תשובה מפורטת וטכנית תוך שימוש אך ורק במידע מהמסמכים למטה.

דרישות קריטיות:
1. **היה ספציפי ביותר**: ציין שיטות מדויקות, פרטים ותרומות טכניות
2. **כלול פרטים מפתח**: כל העובדות, הנתונים והפרטים הרלוונטיים מהמסמכים
3. **צטט במדויק**: התייחס למסמכים ספציפיים באמצעות [1], [2] וכו' עבור כל טענה עובדתית
4. **פורמט מובנה**: ארגן את התשובה שלך בצורה ברורה עם נקודות עיקריות ופרטים תומכים
5. **ללא הכללות**: הימנע מאמירות מעורפלות - היה ספציפי וקונקרטי
6. **ציטוטים מפורשים**: כל טענה עובדתית חייבת לכלול ציטוט [N]
7. **כיסוי מלא**: אם מספר מסמכים רלוונטיים, הסבר כיצד הם קשורים
8. **אל תוסיף סעיף מקורות**: לעולם אל תכלול סעיף "מקורות:" או "References:" בסוף

הנחיות שפה (קריטי):
- עליך לענות בעברית בלבד
- כל התשובה חייבת להיות בעברית
- שמור על טקסט רלוונטי מהמסמכים, אך ההסבר שלך חייב להיות בעברית

---מסמכים שאוחזרו---
{context}

---שאלה---
{processed_query}

---תשובה---
בהתבסס על המסמכים שסופקו:

חשוב: התשובה שלך צריכה לכלול רק את ההסבר וציטוטים משובצים [N]. אל תכלול סעיף "מקורות:" בסוף."""

    def _build_english_prompt(self, processed_query: str, context: str) -> str:
        """Build prompt in English for English queries."""
        return f"""You are an expert research assistant. Your task is to provide a DETAILED, TECHNICAL answer using ONLY the information from the documents below.

CRITICAL REQUIREMENTS:
1. **Be EXTREMELY SPECIFIC**: Mention exact methods, details, and technical contributions
2. **Include KEY DETAILS**: All relevant facts, data, and specifics from the documents
3. **CITE PRECISELY**: Reference specific documents using [1], [2], etc. for EVERY factual claim
4. **STRUCTURED FORMAT**: Organize your answer clearly with main points and supporting details
5. **NO GENERALIZATIONS**: Avoid vague statements - be specific and concrete
6. **EXPLICIT CITATIONS**: Every factual claim must have a citation [N]
7. **COMPLETE COVERAGE**: If multiple documents are relevant, explain how they relate
8. **DO NOT ADD REFERENCES SECTION**: NEVER include a "References:" section at the end

LANGUAGE INSTRUCTIONS (CRITICAL):
- You MUST respond entirely in English
- Your entire response must be in English
- Preserve any relevant text from documents accurately, but your explanation must be in English

---RETRIEVED DOCUMENTS---
{context}

---QUESTION---
{processed_query}

---ANSWER---
Based on the provided documents:

IMPORTANT: Your answer should contain ONLY the explanation and inline citations [N]. DO NOT include a "References:" section at the end."""
    
    def answer_query(self, query: str, top_k: int = 5) -> Dict:
        """Answer query using RAG with chunk-based retrieval."""
        logger.info(f"Answering query: {query}")
        
        # Retrieve relevant document chunks
        results = self.indexer.search(query, top_k=top_k)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "citations": [],
                "retrieved_context": []
            }
        
        # Build prompt with search results (includes chunks and documents)
        prompt = self._build_prompt(query, results)
        
        # Generate answer using the LLM backend
        logger.info("Generating answer with LLM...")
        stop_sequences = ["Citations", "---CITATIONS---", "---QUESTION---", "---ANSWER---", 
                         "User Question:", "References:", "\nReferences:", "\n\nReferences:"]
        
        answer = self.llm_backend.generate(
            prompt,
            max_tokens=self.config.MAX_TOKENS,
            temperature=self.config.TEMPERATURE,
            top_p=self.config.TOP_P,
            repeat_penalty=self.config.REPEAT_PENALTY,
            stop=stop_sequences,
        )
        
        # Extract citations from results
        citations = []
        seen_docs = set()
        for result in results:
            chunk = result.get('chunk')
            doc = result.get('document', {})
            
            # Get document ID
            if chunk:
                doc_id = chunk.document_id if isinstance(chunk, DocumentChunk) else chunk.get('document_id', '')
                chunk_meta = chunk.metadata if isinstance(chunk, DocumentChunk) else chunk.get('metadata', {})
            else:
                doc_id = doc.get('document_id', doc.get('id', ''))
                chunk_meta = {}
            
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            
            # Build citation
            title = (
                doc.get('filename') or 
                chunk_meta.get('title') or 
                doc.get('title', 'Unknown')
            )
            authors = chunk_meta.get('authors', doc.get('authors', 'Unknown'))
            
            citations.append({
                "doc_id": doc_id,
                "title": title,
                "authors": authors
            })
        
        # Extract retrieved context from chunks
        retrieved_context = []
        for result in results:
            chunk = result.get('chunk')
            if chunk:
                content = chunk.content if isinstance(chunk, DocumentChunk) else chunk.get('content', '')
                retrieved_context.append(content)
            else:
                doc = result.get('document', {})
                retrieved_context.append(doc.get('abstract', doc.get('content', '')))

        logger.info("Answer generated successfully")
        
        return {
            "answer": answer,
            "citations": citations,
            "retrieved_context": retrieved_context
        }

    async def stream_answer(self, query: str, top_k: int = 5) -> AsyncGenerator[Dict, None]:
        """Stream answer generation with chunk-based retrieval."""
        import asyncio
        
        # Retrieve document chunks
        results = self.indexer.search(query, top_k=top_k)
        
        if not results:
            yield {
                "type": "token",
                "content": "I couldn't find any relevant documents to answer your question."
            }
            yield {
                "type": "complete",
                "content": "I couldn't find any relevant documents to answer your question."
            }
            return
        
        # Build citations from results
        citations = []
        seen_docs = set()
        for result in results:
            chunk = result.get('chunk')
            doc = result.get('document', {})
            
            if chunk:
                doc_id = chunk.document_id if isinstance(chunk, DocumentChunk) else chunk.get('document_id', '')
                chunk_meta = chunk.metadata if isinstance(chunk, DocumentChunk) else chunk.get('metadata', {})
            else:
                doc_id = doc.get('document_id', doc.get('id', ''))
                chunk_meta = {}
            
            if doc_id in seen_docs:
                continue
            seen_docs.add(doc_id)
            
            title = doc.get('filename') or chunk_meta.get('title') or doc.get('title', 'Unknown')
            authors = chunk_meta.get('authors', doc.get('authors', 'Unknown'))
            
            citations.append({
                "doc_id": doc_id,
                "title": title,
                "authors": authors
            })
        
        # Send citations first
        yield {
            "type": "citations",
            "content": citations
        }
        
        # Send retrieved context
        retrieved_context = []
        for result in results:
            chunk = result.get('chunk')
            if chunk:
                content = chunk.content if isinstance(chunk, DocumentChunk) else chunk.get('content', '')
                retrieved_context.append(content)
            else:
                doc = result.get('document', {})
                retrieved_context.append(doc.get('abstract', doc.get('content', '')))
        
        yield {
            "type": "context",
            "content": retrieved_context
        }
        
        # Build prompt
        prompt = self._build_prompt(query, results)
        
        # Stream answer using the LLM backend
        full_answer = ""
        stop_sequences = ["Citations", "---CITATIONS---", "---QUESTION---", "---ANSWER---", 
                         "User Question:", "References:", "\nReferences:", "\n\nReferences:"]
        
        token_count = 0
        for token in self.llm_backend.stream(
            prompt,
            max_tokens=self.config.MAX_TOKENS,
            temperature=self.config.TEMPERATURE,
            top_p=self.config.TOP_P,
            repeat_penalty=self.config.REPEAT_PENALTY,
            stop=stop_sequences,
        ):
            full_answer += token
            yield {
                "type": "token",
                "content": token
            }
            
            # Yield control to event loop periodically
            token_count += 1
            if token_count % 5 == 0:
                await asyncio.sleep(0)
        
        # Send completion signal
        yield {
            "type": "complete",
            "content": full_answer.strip()
        }
