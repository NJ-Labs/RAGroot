"""
Agentic RAG System with Hebrew and English Agents.
Uses LangGraph for workflow orchestration, LangChain for agent building,
and DSPy for optimized prompt generation.

Architecture:
1. Language Detection Router - Detects query language (Hebrew/English)
2. Hebrew Agent - Processes Hebrew queries and responds in Hebrew
3. English Agent - Processes English queries and responds in English
4. Both agents use the same retrieval system but different prompts
"""

import logging
from typing import List, Dict, Any, Optional, TypedDict, Annotated, Literal
from dataclasses import dataclass
import operator
import re

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LangChain imports
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# DSPy imports
import dspy

logger = logging.getLogger(__name__)


# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class AgentState(TypedDict):
    """State passed between nodes in the graph."""
    query: str
    language: str  # "hebrew" or "english"
    retrieved_chunks: List[Dict]
    context: str
    answer: str
    citations: List[Dict]
    is_complete: bool


# =============================================================================
# DSPy SIGNATURES FOR CONCISE ANSWERS
# =============================================================================

class ConciseAnswerSignature(dspy.Signature):
    """Generate a concise, to-the-point answer based on retrieved context.
    
    The answer should be short and focused, only elaborating when the 
    question requires detailed explanation.
    """
    query: str = dspy.InputField(desc="The user's question")
    context: str = dspy.InputField(desc="Retrieved document context")
    language: str = dspy.InputField(desc="Response language: 'hebrew' or 'english'")
    answer: str = dspy.OutputField(desc="Concise, factual answer with citations [N]")


class HebrewAnswerSignature(dspy.Signature):
    """תשובה קצרה וממוקדת בעברית על בסיס ההקשר שאוחזר.
    
    התשובה צריכה להיות קצרה ולעניין, להרחיב רק כשנדרש.
    """
    query: str = dspy.InputField(desc="שאלת המשתמש")
    context: str = dspy.InputField(desc="הקשר מהמסמכים")
    answer: str = dspy.OutputField(desc="תשובה קצרה עם ציטוטים [N]")


class EnglishAnswerSignature(dspy.Signature):
    """Generate a concise, to-the-point answer in English.
    
    Keep it short and focused, elaborate only when necessary.
    """
    query: str = dspy.InputField(desc="The user's question")
    context: str = dspy.InputField(desc="Retrieved document context")
    answer: str = dspy.OutputField(desc="Concise answer with citations [N]")


# =============================================================================
# LANGUAGE DETECTOR
# =============================================================================

class LanguageDetector:
    """Detects the language of input text - Hebrew or English."""
    
    # Hebrew Unicode range: \u0590-\u05FF (Hebrew block)
    HEBREW_PATTERN = re.compile(r'[\u0590-\u05FF]')
    
    @staticmethod
    def detect(text: str) -> str:
        """
        Detect if text is primarily Hebrew or English.
        
        Args:
            text: Input text to analyze
            
        Returns:
            "hebrew" or "english"
        """
        # Count Hebrew characters
        hebrew_chars = len(LanguageDetector.HEBREW_PATTERN.findall(text))
        # Count Latin characters
        latin_chars = sum(1 for char in text if 'a' <= char.lower() <= 'z')
        
        # If more Hebrew characters than Latin, consider it Hebrew
        if hebrew_chars > latin_chars:
            return "hebrew"
        return "english"
    
    @staticmethod
    def get_language_name(lang_code: str) -> str:
        """Get human-readable language name."""
        return "עברית" if lang_code == "hebrew" else "English"


# =============================================================================
# AGENT PROMPTS
# =============================================================================

HEBREW_SYSTEM_PROMPT = """אתה עוזר מחקר מומחה. ענה בקצרה ותמציתית.

כללים:
1. תשובות קצרות - רק מה שנדרש
2. השתמש בציטוטים [1], [2] וכו' לכל טענה
3. הרחב רק כשהשאלה דורשת פירוט
4. אל תוסיף "מקורות:" בסוף
5. ענה בעברית בלבד

---מסמכים---
{context}

---שאלה---
{query}

---תשובה---"""

ENGLISH_SYSTEM_PROMPT = """You are an expert research assistant. Answer concisely and to the point.

Rules:
1. Keep answers short - only what's needed
2. Use citations [1], [2], etc. for every claim
3. Elaborate only when the question requires detail
4. DO NOT add "References:" section at the end
5. Answer in English only

---DOCUMENTS---
{context}

---QUESTION---
{query}

---ANSWER---"""


# =============================================================================
# RAG AGENTS
# =============================================================================

class BaseRAGAgent:
    """Base class for language-specific RAG agents."""
    
    def __init__(self, llm_backend, indexer, language: str):
        self.llm_backend = llm_backend
        self.indexer = indexer
        self.language = language
        self.language_detector = LanguageDetector()
        
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict]:
        """Retrieve relevant documents."""
        return self.indexer.search(query, top_k=top_k)
    
    def format_context(self, results: List[Dict]) -> str:
        """Format retrieved results into context string."""
        from .document_processor import DocumentChunk
        from .latex_utils import LaTeXMathHandler
        
        latex_handler = LaTeXMathHandler(preserve_structure=True)
        context_parts = []
        
        for i, result in enumerate(results, 1):
            chunk = result.get('chunk')
            doc = result.get('document', {})
            
            if chunk:
                content = chunk.content if isinstance(chunk, DocumentChunk) else chunk.get('content', '')
                section = chunk.section_title if isinstance(chunk, DocumentChunk) else chunk.get('section_title', '')
                chunk_metadata = chunk.metadata if isinstance(chunk, DocumentChunk) else chunk.get('metadata', {})
                
                processed_content = latex_handler.process_text(content)
                
                entry_parts = [f"[{i}]"]
                if doc.get('filename'):
                    entry_parts.append(f"Source: {doc.get('filename')}")
                if section:
                    entry_parts.append(f"Section: {section}")
                entry_parts.append(f"Content: {processed_content}")
                
                context_parts.append("\n".join(entry_parts))
            else:
                doc = result.get('document', result)
                processed_abstract = latex_handler.process_text(doc.get('abstract', ''))
                context_parts.append(
                    f"[{i}] Title: {doc.get('title', 'Unknown')}\n"
                    f"Content: {processed_abstract}"
                )
        
        return "\n\n".join(context_parts)
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate answer using LLM. Override in subclasses."""
        raise NotImplementedError
    
    def extract_citations(self, results: List[Dict]) -> List[Dict]:
        """Extract citations from results."""
        from .document_processor import DocumentChunk
        
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
        
        return citations


class HebrewAgent(BaseRAGAgent):
    """Agent specialized for Hebrew queries and responses."""
    
    def __init__(self, llm_backend, indexer):
        super().__init__(llm_backend, indexer, "hebrew")
        self.prompt_template = HEBREW_SYSTEM_PROMPT
        logger.info("Hebrew Agent initialized")
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate Hebrew answer."""
        prompt = self.prompt_template.format(context=context, query=query)
        
        # Don't include prompt template markers as stop sequences
        stop_sequences = ["מקורות:", "\nמקורות:", "---מסמכים---", "\n\n---"]
        
        from .config import config
        
        answer = self.llm_backend.generate(
            prompt,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            repeat_penalty=config.REPEAT_PENALTY,
            stop=stop_sequences,
        )
        
        return answer.strip()


class EnglishAgent(BaseRAGAgent):
    """Agent specialized for English queries and responses."""
    
    def __init__(self, llm_backend, indexer):
        super().__init__(llm_backend, indexer, "english")
        self.prompt_template = ENGLISH_SYSTEM_PROMPT
        logger.info("English Agent initialized")
    
    def generate_answer(self, query: str, context: str) -> str:
        """Generate English answer."""
        prompt = self.prompt_template.format(context=context, query=query)
        
        # Don't include prompt template markers as stop sequences
        stop_sequences = ["References:", "\nReferences:", "---DOCUMENTS---", "\n\n---"]
        
        from .config import config
        
        answer = self.llm_backend.generate(
            prompt,
            max_tokens=config.MAX_TOKENS,
            temperature=config.TEMPERATURE,
            top_p=config.TOP_P,
            repeat_penalty=config.REPEAT_PENALTY,
            stop=stop_sequences,
        )
        
        return answer.strip()


# =============================================================================
# LANGGRAPH WORKFLOW
# =============================================================================

class AgenticRAGPipeline:
    """
    Agentic RAG Pipeline using LangGraph for workflow orchestration.
    
    The workflow:
    1. Detect language of the query
    2. Route to appropriate agent (Hebrew or English)
    3. Retrieve relevant documents
    4. Generate concise answer in the detected language
    """
    
    def __init__(self, indexer, llm_backend=None):
        """
        Initialize the agentic RAG pipeline.
        
        Args:
            indexer: VectorIndexer instance for document retrieval
            llm_backend: Optional LLM backend (will be created if not provided)
        """
        self.indexer = indexer
        self.language_detector = LanguageDetector()
        
        # Initialize LLM backend if not provided
        if llm_backend is None:
            llm_backend = self._create_llm_backend()
        self.llm_backend = llm_backend
        
        # Initialize agents
        self.hebrew_agent = HebrewAgent(llm_backend, indexer)
        self.english_agent = EnglishAgent(llm_backend, indexer)
        
        # Build the workflow graph
        self.workflow = self._build_workflow()
        
        logger.info("AgenticRAGPipeline initialized with Hebrew and English agents")
    
    def _create_llm_backend(self):
        """Create LLM backend based on configuration."""
        from .config import config
        from .retriever import LlamaCppBackend, VLLMBackend
        import os
        
        llm_backend_type = getattr(config, 'LLM_BACKEND', os.getenv('LLM_BACKEND', 'llama_cpp'))
        
        if llm_backend_type == 'vllm':
            logger.info("Using vLLM backend for agents")
            return VLLMBackend(config)
        else:
            logger.info("Using llama.cpp backend for agents")
            return LlamaCppBackend(config)
    
    def _build_workflow(self) -> StateGraph:
        """Build the LangGraph workflow."""
        
        # Define the workflow graph
        workflow = StateGraph(AgentState)
        
        # Add nodes
        workflow.add_node("detect_language", self._detect_language_node)
        workflow.add_node("retrieve_documents", self._retrieve_documents_node)
        workflow.add_node("hebrew_agent", self._hebrew_agent_node)
        workflow.add_node("english_agent", self._english_agent_node)
        
        # Set entry point
        workflow.set_entry_point("detect_language")
        
        # Add edges
        workflow.add_edge("detect_language", "retrieve_documents")
        
        # Conditional routing based on language
        workflow.add_conditional_edges(
            "retrieve_documents",
            self._route_to_agent,
            {
                "hebrew": "hebrew_agent",
                "english": "english_agent"
            }
        )
        
        # Both agents lead to end
        workflow.add_edge("hebrew_agent", END)
        workflow.add_edge("english_agent", END)
        
        # Compile the workflow
        return workflow.compile()
    
    def _detect_language_node(self, state: AgentState) -> AgentState:
        """Node: Detect the language of the query."""
        query = state["query"]
        language = self.language_detector.detect(query)
        
        logger.info(f"Detected language: {language} for query: {query[:50]}...")
        
        return {
            **state,
            "language": language
        }
    
    def _retrieve_documents_node(self, state: AgentState) -> AgentState:
        """Node: Retrieve relevant documents."""
        query = state["query"]
        
        # Retrieve documents
        results = self.indexer.search(query, top_k=5)
        
        # Format context
        agent = self.hebrew_agent if state["language"] == "hebrew" else self.english_agent
        context = agent.format_context(results)
        citations = agent.extract_citations(results)
        
        logger.info(f"Retrieved {len(results)} documents for query")
        
        return {
            **state,
            "retrieved_chunks": results,
            "context": context,
            "citations": citations
        }
    
    def _route_to_agent(self, state: AgentState) -> str:
        """Router: Decide which agent to use based on language."""
        return state["language"]
    
    def _hebrew_agent_node(self, state: AgentState) -> AgentState:
        """Node: Process with Hebrew agent."""
        query = state["query"]
        context = state["context"]
        
        logger.info("Processing with Hebrew Agent...")
        answer = self.hebrew_agent.generate_answer(query, context)
        
        return {
            **state,
            "answer": answer,
            "is_complete": True
        }
    
    def _english_agent_node(self, state: AgentState) -> AgentState:
        """Node: Process with English agent."""
        query = state["query"]
        context = state["context"]
        
        logger.info("Processing with English Agent...")
        answer = self.english_agent.generate_answer(query, context)
        
        return {
            **state,
            "answer": answer,
            "is_complete": True
        }
    
    def answer_query(self, query: str, top_k: int = 5) -> Dict:
        """
        Answer a query using the agentic RAG pipeline.
        
        Args:
            query: User's question (Hebrew or English)
            top_k: Number of documents to retrieve
            
        Returns:
            Dict with answer, citations, and retrieved context
        """
        logger.info(f"AgenticRAG: Processing query: {query}")
        
        # Initialize state
        initial_state: AgentState = {
            "query": query,
            "language": "",
            "retrieved_chunks": [],
            "context": "",
            "answer": "",
            "citations": [],
            "is_complete": False
        }
        
        # Run the workflow
        final_state = self.workflow.invoke(initial_state)
        
        # Handle case where no documents found
        if not final_state.get("retrieved_chunks"):
            no_docs_message = (
                "לא נמצאו מסמכים רלוונטיים לשאלה שלך." 
                if final_state["language"] == "hebrew" 
                else "No relevant documents found for your query."
            )
            return {
                "answer": no_docs_message,
                "citations": [],
                "retrieved_context": [],
                "language": final_state["language"]
            }
        
        # Extract retrieved context
        retrieved_context = []
        for result in final_state["retrieved_chunks"]:
            chunk = result.get('chunk')
            if chunk:
                from .document_processor import DocumentChunk
                content = chunk.content if isinstance(chunk, DocumentChunk) else chunk.get('content', '')
                retrieved_context.append(content)
            else:
                doc = result.get('document', {})
                retrieved_context.append(doc.get('abstract', doc.get('content', '')))
        
        logger.info(f"AgenticRAG: Query completed with {final_state['language']} agent")
        
        return {
            "answer": final_state["answer"],
            "citations": final_state["citations"],
            "retrieved_context": retrieved_context,
            "language": final_state["language"]
        }
    
    async def stream_answer(self, query: str, top_k: int = 5):
        """
        Stream answer generation for the agentic RAG pipeline.
        
        Args:
            query: User's question
            top_k: Number of documents to retrieve
            
        Yields:
            Chunks of the response
        """
        import asyncio
        
        # Detect language
        language = self.language_detector.detect(query)
        
        # Retrieve documents
        results = self.indexer.search(query, top_k=top_k)
        
        if not results:
            no_docs_message = (
                "לא נמצאו מסמכים רלוונטיים לשאלה שלך."
                if language == "hebrew"
                else "No relevant documents found for your query."
            )
            yield {"type": "token", "content": no_docs_message}
            yield {"type": "complete", "content": no_docs_message}
            return
        
        # Get appropriate agent
        agent = self.hebrew_agent if language == "hebrew" else self.english_agent
        
        # Format context and citations
        context = agent.format_context(results)
        citations = agent.extract_citations(results)
        
        # Yield citations first
        yield {"type": "citations", "content": citations}
        
        # Yield retrieved context
        retrieved_context = []
        for result in results:
            chunk = result.get('chunk')
            if chunk:
                from .document_processor import DocumentChunk
                content = chunk.content if isinstance(chunk, DocumentChunk) else chunk.get('content', '')
                retrieved_context.append(content)
            else:
                doc = result.get('document', {})
                retrieved_context.append(doc.get('abstract', doc.get('content', '')))
        
        yield {"type": "context", "content": retrieved_context}
        
        # Build prompt
        prompt = agent.prompt_template.format(context=context, query=query)
        
        # Stream answer
        full_answer = ""
        # Note: Don't include prompt template markers as stop sequences
        # Only use sequences that indicate the model is done answering
        stop_sequences = (
            ["מקורות:", "\nמקורות:", "---מסמכים---", "\n\n---"]
            if language == "hebrew"
            else ["References:", "\nReferences:", "---DOCUMENTS---", "\n\n---"]
        )
        
        from .config import config
        
        token_count = 0
        try:
            for token in self.llm_backend.stream(
                prompt,
                max_tokens=config.MAX_TOKENS,
                temperature=config.TEMPERATURE,
                top_p=config.TOP_P,
                repeat_penalty=config.REPEAT_PENALTY,
                stop=stop_sequences,
            ):
                if token:  # Only yield non-empty tokens
                    full_answer += token
                    yield {"type": "token", "content": token}
                    
                    token_count += 1
                    if token_count % 5 == 0:
                        await asyncio.sleep(0)
            
            # If no tokens were streamed, fall back to non-streaming generation
            if token_count == 0:
                logger.warning("Streaming yielded no tokens, falling back to non-streaming generation")
                try:
                    full_answer = self.llm_backend.generate(
                        prompt,
                        max_tokens=config.MAX_TOKENS,
                        temperature=config.TEMPERATURE,
                        top_p=config.TOP_P,
                        repeat_penalty=config.REPEAT_PENALTY,
                        stop=stop_sequences,
                    )
                    logger.info(f"Non-streaming generation returned: {len(full_answer) if full_answer else 0} chars")
                    # Send the full answer as a single token
                    if full_answer:
                        yield {"type": "token", "content": full_answer}
                    else:
                        logger.error("Non-streaming generation returned empty response")
                        error_msg = "המודל לא החזיר תשובה. נסה שוב." if language == "hebrew" else "The model returned no response. Please try again."
                        yield {"type": "token", "content": error_msg}
                        full_answer = error_msg
                except Exception as gen_err:
                    logger.error(f"Non-streaming generation failed: {gen_err}", exc_info=True)
                    error_msg = "שגיאה בקבלת תשובה מהמודל." if language == "hebrew" else "Error getting response from model."
                    yield {"type": "token", "content": error_msg}
                    full_answer = error_msg
                    
        except Exception as e:
            logger.error(f"Error during streaming: {e}", exc_info=True)
            # Try non-streaming as fallback
            try:
                full_answer = self.llm_backend.generate(
                    prompt,
                    max_tokens=config.MAX_TOKENS,
                    temperature=config.TEMPERATURE,
                    top_p=config.TOP_P,
                    repeat_penalty=config.REPEAT_PENALTY,
                    stop=stop_sequences,
                )
                if full_answer:
                    yield {"type": "token", "content": full_answer}
            except Exception as e2:
                logger.error(f"Fallback generation also failed: {e2}", exc_info=True)
                error_msg = "לא ניתן לקבל תשובה מהמודל." if language == "hebrew" else "Unable to get response from the model."
                yield {"type": "token", "content": error_msg}
                full_answer = error_msg
        
        yield {"type": "complete", "content": full_answer.strip()}


# =============================================================================
# DSPY OPTIMIZED MODULE (Optional - for advanced optimization)
# =============================================================================

class DSPyRAGModule(dspy.Module):
    """
    DSPy module for optimized RAG responses.
    Can be used for few-shot learning and prompt optimization.
    """
    
    def __init__(self, language: str = "english"):
        super().__init__()
        self.language = language
        
        if language == "hebrew":
            self.answer_generator = dspy.ChainOfThought(HebrewAnswerSignature)
        else:
            self.answer_generator = dspy.ChainOfThought(EnglishAnswerSignature)
    
    def forward(self, query: str, context: str) -> str:
        """Generate an answer using DSPy."""
        result = self.answer_generator(query=query, context=context)
        return result.answer


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_agentic_rag_pipeline(indexer) -> AgenticRAGPipeline:
    """
    Factory function to create an AgenticRAGPipeline.
    
    Args:
        indexer: VectorIndexer instance
        
    Returns:
        Configured AgenticRAGPipeline
    """
    return AgenticRAGPipeline(indexer)

