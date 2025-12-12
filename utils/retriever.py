import logging
from typing import List, Dict, AsyncGenerator
import os

from llama_cpp import Llama
from .latex_utils import LaTeXMathHandler

logger = logging.getLogger(__name__)

class RAGPipeline:
    """Retrieval-Augmented Generation pipeline."""
    
    def __init__(self, indexer):
        from .config import config
        from pathlib import Path
        
        self.indexer = indexer
        self.latex_handler = LaTeXMathHandler(preserve_structure=True)
        
        # Validate model file exists
        model_path = Path(config.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {config.MODEL_PATH}\n"
                f"Please ensure the model file exists and the path is correct in your .env file."
            )
        
        # Initialize local LLM
        logger.info(f"Loading LLM from {config.MODEL_PATH}")
        logger.info(f"Model size: {model_path.stat().st_size / (1024**3):.2f} GB")
        
        # Determine GPU layers based on FORCE_CPU setting
        if config.FORCE_CPU:
            n_gpu_layers = 0
            logger.info("Force CPU mode enabled - LLM will run on CPU only")
        else:
            n_gpu_layers = config.N_GPU_LAYERS
            if n_gpu_layers > 0:
                logger.info(f"GPU acceleration enabled with {n_gpu_layers} layers")
            else:
                logger.info("Running on CPU (N_GPU_LAYERS=0)")
        
        try:
            # Use llama.cpp for efficient inference (supports LLaMA, Qwen, Phi, etc.)
            self.llm = Llama(
                model_path=str(model_path),
                n_ctx=config.N_CTX,
                n_threads=config.N_THREADS,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
                n_batch=512,  # Batch size for prompt processing
            )
            
            # Verify model loaded correctly
            try:
                vocab_size = self.llm.n_vocab()
                logger.info(f"LLM loaded successfully (vocab size: {vocab_size})")
            except:
                logger.info("LLM loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to load LLM: {e}")
            raise
    
    def _build_prompt(self, query: str, context_docs: List[Dict]) -> str:
        """Build prompt for LLM with retrieved context."""
        
        # Process query to expand LaTeX for better understanding
        processed_query = self.latex_handler.process_text(query)
        
        # Build context from retrieved documents with better formatting
        context_parts = []
        for i, doc in enumerate(context_docs, 1):
            # Process LaTeX in title and abstract for better LLM understanding
            processed_abstract = self.latex_handler.process_text(doc['abstract'])
            processed_title = self.latex_handler.process_text(doc['title'])
            authors = doc.get('authors', 'Unknown')
            
            context_parts.append(
                f"[{i}] ID: {doc['id']}\n"
                f"Title: {processed_title}\n"
                f"Authors: {authors}\n"
                f"Abstract: {processed_abstract}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Build improved prompt with EXPLICIT instructions for specificity
        prompt = f"""You are an expert research assistant specializing in academic papers. Your task is to provide a DETAILED, TECHNICAL answer using ONLY the information from the documents below.

CRITICAL REQUIREMENTS:
1. **Be EXTREMELY SPECIFIC**: Mention exact methods, algorithms, frameworks, architectures, and technical contributions
2. **Include KEY DETAILS**: Datasets used, metrics reported, mathematical formulations, experimental setups
3. **CITE PRECISELY**: Reference specific papers using [1], [2], etc. for EVERY factual claim you make
4. **STRUCTURED FORMAT**: Use this structure
   - Main Topic/Question: [One sentence summary of the answer]
   - Specific Approaches: [Detailed methodology from each relevant paper]
   - Key Technical Details: [Algorithms, models, techniques with specifics]
   - Results/Contributions: [Quantitative results, innovations, impact]
5. **NO GENERALIZATIONS**: Avoid vague statements like "various approaches" or "recent methods"
6. **EXPLICIT CITATIONS**: Every factual claim must have a citation [N]
7. **COMPLETE COVERAGE**: If multiple papers are relevant, explain how they differ or relate
8. **DO NOT ADD REFERENCES SECTION**: NEVER include a "References:" section at the end - citations are handled separately

If the query asks about:
- A specific paper: Describe its EXACT methodology, contributions, and results in detail
- A technical method: Explain HOW it works step-by-step with technical details
- A broad topic: Identify 3-5 specific approaches and explain each in detail
- A comparison: Explicitly compare and contrast the approaches with specifics
- A publication date: the date is stored in the "id" field as YYMM.string use only the first 4 digit for the date(e.g. "id": "2509.21240v1" -> 2509 -> Sep 2025)

---RETRIEVED DOCUMENTS---
{context}

---QUESTION---
{processed_query}

---DETAILED TECHNICAL ANSWER---
Based on the provided research papers:

**Main Topic:**

IMPORTANT: Your answer should contain ONLY the explanation and inline citations [N]. DO NOT include a "References:" section, bibliography, or list of references at the end. The references will be displayed separately in the Citations card."""
        
        return prompt
    
    def answer_query(self, query: str, top_k: int = 5) -> Dict:
        """Answer query using RAG."""
        logger.info(f"Answering query: {query}")
        
        # Retrieve relevant documents
        results = self.indexer.search(query, top_k=top_k)
        
        if not results:
            return {
                "answer": "I couldn't find any relevant documents to answer your question.",
                "citations": [],
                "retrieved_context": []
            }
        
        # Extract documents (keep original with LaTeX for UI display)
        context_docs = [r['document'] for r in results]
        
        # Build prompt (uses processed LaTeX internally for LLM understanding)
        prompt = self._build_prompt(query, context_docs)
        
        # Generate answer with config settings
        from .config import config as cfg
        
        logger.info("Generating answer with LLM...")
        response = self.llm(
            prompt,
            max_tokens=cfg.MAX_TOKENS,
            temperature=cfg.TEMPERATURE,
            top_p=cfg.TOP_P,
            repeat_penalty=cfg.REPEAT_PENALTY,
            stop=["Citations", "---CITATIONS---","---QUESTION---", "---ANSWER---", "User Question:", "References:", "\nReferences:", "\n\nReferences:"],
            echo=False
        )
        
        answer = response['choices'][0]['text'].strip()
        
        # Extract citations with ORIGINAL titles and authors
        citations = [
            {
                "doc_id": doc['id'],
                "title": doc['title'], 
                "authors": doc.get('authors', 'Unknown')
            }
            for doc in context_docs
        ]
        
        # Extract retrieved context with abstracts
        retrieved_context = [doc['abstract'] for doc in context_docs]

        logger.info("Answer generated successfully")
        
        return {
            "answer": answer,
            "citations": citations,
            "retrieved_context": retrieved_context
        }

    async def stream_answer(self, query: str, top_k: int = 5) -> AsyncGenerator[Dict, None]:
        """Stream answer generation (bonus feature)."""
        import asyncio
        
        # Retrieve documents
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
        
        # Extract documents
        context_docs = [r['document'] for r in results]
        
        # Send citations first (with authors field)
        yield {
            "type": "citations",
            "content": [
                {
                    "doc_id": doc['id'], 
                    "title": doc['title'],
                    "authors": doc.get('authors', 'Unknown')
                }
                for doc in context_docs
            ]
        }
        
        # Send retrieved context
        yield {
            "type": "context",
            "content": [doc['abstract'] for doc in context_docs]
        }
        
        # Build prompt
        prompt = self._build_prompt(query, context_docs)
        
        # Stream answer with config settings
        full_answer = ""
        from .config import config
        
        # Use a token counter to yield control to event loop periodically
        token_count = 0
        for chunk in self.llm(
            prompt,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            repeat_penalty=config.REPEAT_PENALTY,
            stop=["Citations", "---CITATIONS---", "---QUESTION---", "---ANSWER---", "User Question:", "References:", "\nReferences:", "\n\nReferences:"],
            stream=True
        ):
            token = chunk['choices'][0]['text']
            full_answer += token
            yield {
                "type": "token",
                "content": token
            }
            
            # Yield control to event loop periodically to keep the server responsive.
            token_count += 1
            if token_count % 5 == 0:
                await asyncio.sleep(0)
        
        # Send completion signal
        yield {
            "type": "complete",
            "content": full_answer.strip()
        }
