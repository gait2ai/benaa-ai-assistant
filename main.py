# main.py (updated with model_router integration and memory optimization)

import os
import json
import logging
from typing import List, Dict, Any
import time
from datetime import datetime
import gc
from cachetools import TTLCache, LRUCache

from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv
from langdetect import detect

# Import the model router module
from model_router import router as model_router

# Load environment variables
load_dotenv()

# Check for OPENROUTER_API_KEY only
if not os.getenv("OPENROUTER_API_KEY"):
    raise ValueError("Please set OPENROUTER_API_KEY environment variable.")

# Create FastAPI app
app = FastAPI(title="Benaa Association AI Assistant")

# Set up CORS to allow connection with frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # You can restrict this later to your site's domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(model_router)

# Path to embeddings and data
EMBEDDINGS_FOLDER = "./embeddings"
DATA_FOLDER = "./data"
CONFIG_FOLDER = "./config"
STATIC_FOLDER = "./static"
SYSTEM_PROMPT_PATH = os.path.join(CONFIG_FOLDER, "system_prompt.txt")

# Global variables to store loaded models and data
global_index = None
global_metadata = []
global_embedding_model = None
global_model_name = None

# Cache configuration
CACHE_MAX_SIZE = 100  # Maximum number of cached responses
CACHE_TTL = 3600      # Cache time-to-live in seconds (1 hour)
response_cache = TTLCache(maxsize=CACHE_MAX_SIZE, ttl=CACHE_TTL)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# User message model
class UserMessage(BaseModel):
    message: str

# Response model
class RAGResponse(BaseModel):
    answer: str

def clean_json_text(text: str) -> str:
    """Clean JSON text for better readability in responses.
    
    Args:
        text: Text that might contain JSON
        
    Returns:
        Cleaned text or formatted JSON
    """
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            data = json.loads(text)
            # Return formatted JSON with indentation
            return json.dumps(data, indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            # Not valid JSON, return as is
            pass
    return text


class FaissRetriever:
    """Class for retrieving relevant documents using FAISS."""
    
    def __init__(self, index=None, metadata=None, embedding_model=None, model_name=None):
        """Initialize the retriever with pre-loaded components or load them if needed."""
        self.index = index
        self.metadata = metadata
        self.embedding_model = embedding_model
        self.model_name = model_name
        
        # Use global components if available, otherwise load them
        if self.index is None or self.metadata is None or self.embedding_model is None:
            global global_index, global_metadata, global_embedding_model, global_model_name
            
            if global_index is not None and global_metadata and global_embedding_model is not None:
                self.index = global_index
                self.metadata = global_metadata
                self.embedding_model = global_embedding_model
                self.model_name = global_model_name
            else:
                # This should not happen as components should be loaded at startup
                logger.warning("Models not loaded globally, loading them now (this should not happen)")
                self._load_components()
    
    def _load_components(self):
        """Load FAISS index, metadata, and embedding model."""
        try:
            # Define paths
            index_path = os.path.join(EMBEDDINGS_FOLDER, "index.faiss")
            metadata_path = os.path.join(EMBEDDINGS_FOLDER, "metadata.json")
            model_info_path = os.path.join(EMBEDDINGS_FOLDER, "model_info.json")
            
            # Check if index and metadata files exist
            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                logger.error(f"FAISS index or metadata file not found at {EMBEDDINGS_FOLDER}")
                raise FileNotFoundError(f"FAISS index or metadata not found. Run embed_documents.py first.")
            
            # Load FAISS index
            logger.info(f"Loading FAISS index from {index_path}")
            self.index = faiss.read_index(index_path)
            logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            
            # Load document metadata
            logger.info(f"Loading document metadata from {metadata_path}")
            with open(metadata_path, 'r', encoding='utf-8') as f:
                self.metadata = json.load(f)
            logger.info(f"Loaded metadata for {len(self.metadata)} document chunks")
            
            # Load model info
            with open(model_info_path, 'r', encoding='utf-8') as f:
                model_info = json.load(f)
                self.model_name = model_info.get("model_name")
            
            # Load embedding model
            logger.info(f"Loading embedding model: {self.model_name}")
            start_time = time.time()
            self.embedding_model = SentenceTransformer(self.model_name)
            logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Error loading retriever components: {e}")
            raise
    
    def retrieve(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the most relevant document chunks for a query.
        
        Args:
            query: The user query
            top_k: Number of relevant chunks to retrieve
            
        Returns:
            List of document chunks with text and metadata
        """
        try:
            # Create query embedding
            query_embedding = self.embedding_model.encode([query])[0]
            query_embedding = query_embedding.reshape(1, -1)
            
            # Normalize for cosine similarity - CRITICAL for accurate similarity calculation
            faiss.normalize_L2(query_embedding)
            
            # Search the index - retrieve more than needed for filtering duplicates
            # Ensure we retrieve enough candidates (at least 50 or top_k*3, whichever is larger)
            retrieve_count = min(max(top_k * 3, 50), self.index.ntotal)
            distances, indices = self.index.search(query_embedding, retrieve_count)
            
            # Track seen texts to avoid duplicates
            seen_texts = set()
            
            # Get the corresponding documents with proper similarity calculation
            results = []
            for i, doc_idx in enumerate(indices[0]):
                if doc_idx != -1 and doc_idx < len(self.metadata):  # Valid index
                    chunk = self.metadata[doc_idx].copy()  # Use copy to avoid modifying original metadata
                    
                    # Clean text if it appears to be JSON
                    chunk_text = clean_json_text(chunk.get("text", "").strip())
                    chunk["text"] = chunk_text
                    
                    # Skip duplicates based on text content
                    text_key = chunk_text[:100]  # Use prefix for deduplication to save memory
                    if text_key in seen_texts:
                        logger.debug(f"Skipped duplicate chunk: {text_key[:50]}...")
                        continue
                    seen_texts.add(text_key)
                    
                    # Proper similarity calculation for L2-normalized vectors:
                    # For normalized vectors, L2 distance of 0 means identical (similarity=1)
                    # L2 distance of 2 means opposite (similarity=0)
                    # Therefore similarity = 1 - (distance/2)
                    raw_similarity = 1.0 - (distances[0][i] / 2.0)
                    
                    # Ensure similarity score is within [0,1] range
                    chunk["similarity"] = max(0.0, min(1.0, raw_similarity))
                    
                    # Add retrieval timestamp for recency filtering if needed
                    chunk["retrieval_timestamp"] = datetime.now().isoformat()
                    
                    results.append(chunk)
                    
                    # Break if we have enough non-duplicate results
                    if len(results) >= top_k:
                        break
            
            # Sort by similarity score to ensure highest relevance first
            results = sorted(results, key=lambda x: x.get("similarity", 0.0), reverse=True)
            
            if not results:
                logger.warning(f"No relevant documents found for query: {query}")
                return [{"text": "No relevant information found.", "source_file": "", "chunk_id": -1, "similarity": 0.0}]
                
            return results[:top_k]  # Ensure we only return top_k after deduplication
            
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            return [{"text": "Error occurred while searching the knowledge base.", "source_file": "", "chunk_id": -1, "similarity": 0.0}]


def filter_results(chunks: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """Filter retrieved chunks based on various criteria before sending to LLM.
    
    Args:
        chunks: List of document chunks with text and metadata
        top_k: Number of chunks to return after filtering
        
    Returns:
        Filtered list of document chunks
    """
    # If we have fewer chunks than top_k, return all
    if len(chunks) <= top_k:
        return chunks
    
    # Initialize with base priority scores (similarity score)
    for chunk in chunks:
        chunk["priority_score"] = chunk["similarity"]
    
    # Boost priority based on file extension
    file_type_boost = {
        ".docx": 0.2,  # Prefer Word documents
        ".pdf": 0.15,  # Then PDFs
        ".md": 0.1,    # Then Markdown
        ".txt": 0.05   # Plain text gets lowest boost
    }
    
    for chunk in chunks:
        source_file = chunk.get("source_file", "")
        _, ext = os.path.splitext(source_file.lower())
        # Apply file type boost
        if ext in file_type_boost:
            chunk["priority_score"] += file_type_boost[ext]
    
    # Sort chunks by priority score in descending order
    sorted_chunks = sorted(chunks, key=lambda x: x.get("priority_score", 0), reverse=True)
    
    # Return the top_k chunks with highest priority
    return sorted_chunks[:top_k]


# Dependency to get retriever instance
def get_retriever():
    """Dependency to get the FAISS retriever with pre-loaded models."""
    global global_index, global_metadata, global_embedding_model, global_model_name
    return FaissRetriever(
        index=global_index,
        metadata=global_metadata,
        embedding_model=global_embedding_model,
        model_name=global_model_name
    )


def load_system_prompt():
    """Load system prompt from external file or use default."""
    # Default system prompt to use if file cannot be loaded
    default_prompt = """You are a helpful assistant for Benaa Association. 
You must ONLY answer based on the following provided information:

{context}

If you cannot find an answer, politely say you don't have sufficient information and advise the user to contact support.
Always mention the source file when referencing specific information.

{language_instruction}"""

    try:
        # Check if config directory and prompt file exist
        if os.path.exists(SYSTEM_PROMPT_PATH):
            logger.info(f"Loading system prompt from {SYSTEM_PROMPT_PATH}")
            with open(SYSTEM_PROMPT_PATH, 'r', encoding='utf-8') as f:
                custom_prompt = f.read().strip()
                
            # Ensure the custom prompt contains the {context} placeholder
            if "{context}" not in custom_prompt:
                logger.warning("Custom system prompt is missing {context} placeholder. Appending it.")
                custom_prompt += "\n\n{context}"
            
            # Add language instruction placeholder if not present
            if "{language_instruction}" not in custom_prompt:
                custom_prompt += "\n\n{language_instruction}"
                
            return custom_prompt
        else:
            logger.info("System prompt file not found. Using default prompt.")
            return default_prompt
    except Exception as e:
        logger.error(f"Error loading system prompt: {e}. Using default prompt.")
        return default_prompt


def detect_language(text: str) -> str:
    """Detect the language of the input text.
    
    Args:
        text: Text to detect language from
        
    Returns:
        Language code or 'en' as fallback
    """
    try:
        return detect(text)
    except Exception as e:
        logger.warning(f"Language detection failed: {e}. Defaulting to English.")
        return "en"


def get_language_instruction(lang_code: str) -> str:
    """Generate language instruction based on detected language.
    
    Args:
        lang_code: Language code from langdetect
        
    Returns:
        Instruction for the LLM to respond in the appropriate language
    """
    language_map = {
        "ar": "Arabic",
        "en": "English",
        "fr": "French",
        "es": "Spanish",
        "de": "German",
        # Add more languages as needed
    }
    
    language_name = language_map.get(lang_code, "the same language as the user's message")
    return f"Always respond in {language_name}."


def get_cache_key(message: str) -> str:
    """Generate a consistent cache key from user message.
    Trims and normalizes the message to improve cache hit rate.
    
    Args:
        message: The user's query
        
    Returns:
        A cache key string
    """
    # Basic normalization: lowercase and strip whitespace
    normalized = message.lower().strip()
    # Truncate very long messages to avoid excessive memory usage in cache keys
    if len(normalized) > 200:
        normalized = normalized[:200]
    return normalized


@app.post("/api/chat", response_model=RAGResponse)
async def process_message(user_message: UserMessage, retriever: FaissRetriever = Depends(get_retriever)):
    """Process user message using RAG with FAISS retrieval."""
    try:
        # Generate cache key
        cache_key = get_cache_key(user_message.message)
        
        # Check if this query is cached
        if cache_key in response_cache:
            logger.info(f"Cache hit for query: {cache_key[:30]}{'...' if len(cache_key) > 30 else ''}")
            return RAGResponse(answer=response_cache[cache_key])
            
        # Detect language
        detected_lang = detect_language(user_message.message)
        language_instruction = get_language_instruction(detected_lang)
        logger.info(f"Detected language: {detected_lang}")
        
        # Search for relevant documents using FAISS
        relevant_chunks = retriever.retrieve(user_message.message)
        
        # Filter chunks based on priority before sending to LLM
        filtered_chunks = filter_results(relevant_chunks)
        logger.info(f"Retrieved {len(relevant_chunks)} chunks, filtered to {len(filtered_chunks)}")
        
        # Add source information for each chunk
        context_with_sources = []
        for chunk in filtered_chunks:
            source_info = f"[Source: {chunk['source_file']}]"
            context_with_sources.append(f"{chunk['text']}\n{source_info}")
        
        # Load system prompt and format with context and language instruction
        system_prompt_template = load_system_prompt()
        system_message = system_prompt_template.format(
            context=chr(10).join(context_with_sources),
            language_instruction=language_instruction
        )

        # Use the model_router to generate a response
        from model_router import generate_response
        assistant_response = await generate_response(
            prompt=user_message.message,
            system_prompt=system_message,
            max_tokens=1000
        )
        
        # Cache the response
        response_cache[cache_key] = assistant_response
        
        return RAGResponse(answer=assistant_response)

    except Exception as e:
        logger.error(f"Internal error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Handle favicon request
@app.get("/favicon.ico")
async def get_favicon():
    """Serve the favicon."""
    favicon_path = os.path.join(STATIC_FOLDER, "favicon.ico")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path)
    # Return a 204 No Content if favicon doesn't exist
    return Response(status_code=204)


def load_models_and_data():
    """Load all models and data needed for the application."""
    global global_index, global_metadata, global_embedding_model, global_model_name
    
    try:
        # Define paths
        index_path = os.path.join(EMBEDDINGS_FOLDER, "index.faiss")
        metadata_path = os.path.join(EMBEDDINGS_FOLDER, "metadata.json")
        model_info_path = os.path.join(EMBEDDINGS_FOLDER, "model_info.json")
        
        # Check if files exist
        if not os.path.exists(index_path) or not os.path.exists(metadata_path):
            logger.error(f"FAISS index or metadata file not found at {EMBEDDINGS_FOLDER}")
            raise FileNotFoundError(f"FAISS index or metadata not found. Run embed_documents.py first.")
        
        # Load FAISS index
        logger.info(f"Loading FAISS index from {index_path}")
        start_time = time.time()
        global_index = faiss.read_index(index_path)
        logger.info(f"Loaded FAISS index with {global_index.ntotal} vectors in {time.time() - start_time:.2f} seconds")
        
        # Load document metadata
        logger.info(f"Loading document metadata from {metadata_path}")
        with open(metadata_path, 'r', encoding='utf-8') as f:
            global_metadata = json.load(f)
        logger.info(f"Loaded metadata for {len(global_metadata)} document chunks")
        
        # Load model info
        with open(model_info_path, 'r', encoding='utf-8') as f:
            model_info = json.load(f)
            global_model_name = model_info.get("model_name")
        
        # Load embedding model
        logger.info(f"Loading embedding model: {global_model_name}")
        start_time = time.time()
        global_embedding_model = SentenceTransformer(global_model_name)
        logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
        
        # Force garbage collection after loading
        gc.collect()
        
        return True
    except Exception as e:
        logger.error(f"Error loading models and data: {e}")
        return False


@app.on_event("startup")
async def startup_event():
    """Run startup tasks."""
    # Ensure data, embeddings, and config folders exist
    os.makedirs(DATA_FOLDER, exist_ok=True)
    os.makedirs(EMBEDDINGS_FOLDER, exist_ok=True)
    os.makedirs(CONFIG_FOLDER, exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)
    
    # Load all models and data at startup
    logger.info("Loading models and data at startup...")
    if load_models_and_data():
        logger.info("Startup complete: All models and data loaded successfully")
    else:
        logger.error("Startup incomplete: Failed to load models and data")


# Mount static files AFTER defining API endpoints to prevent route overriding
app.mount("/", StaticFiles(directory=STATIC_FOLDER, html=True), name="static")


@app.get("/", response_class=FileResponse)
async def read_index():
    index_path = os.path.join("static", "index.html")
    return index_path


# Add health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint to verify models are loaded."""
    global global_index, global_embedding_model
    
    if global_index is None or global_embedding_model is None:
        # Models not loaded yet
        return {"status": "initializing", "message": "Models are still loading"}
    
    return {"status": "healthy", "message": "Service is running with all models loaded"}


def clear_cache():
    """Clear the response cache."""
    global response_cache
    response_cache.clear()
    gc.collect()
    logger.info("Response cache cleared and garbage collected")


# Add endpoint to clear cache (for admin use)
@app.post("/api/clear-cache")
async def api_clear_cache():
    """Admin endpoint to clear the response cache."""
    clear_cache()
    return {"status": "success", "message": "Cache cleared successfully"}


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
    )
