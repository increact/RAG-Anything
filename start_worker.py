#!/usr/bin/env python3
"""
Start RQ worker for document processing
"""
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from rq import Worker, Queue
from redis import Redis
import logging

# Load environment variables
load_dotenv(dotenv_path=".env", override=False)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize Redis connection
redis_host = os.getenv("REDIS_HOST", "localhost")
redis_port = int(os.getenv("REDIS_PORT", 6379))
redis_db = int(os.getenv("REDIS_DB", 0))
redis_password = os.getenv("REDIS_PASSWORD")

redis_conn = Redis(
    host=redis_host,
    port=redis_port,
    db=redis_db,
    password=redis_password,
    decode_responses=False,
)

# Initialize RAG instance for worker
from raganything import RAGAnything, RAGAnythingConfig
from raganything.document_worker import set_rag_instance

logger.info("Initializing RAG instance for worker...")

# Check if LightRAG is enabled
enable_lightrag = os.getenv("ENABLE_LIGHTRAG", "true").lower() in ("true", "1", "yes")

try:
    # Create configuration
    config = RAGAnythingConfig(
        working_dir=os.getenv("WORKING_DIR", "./rag_storage"),
        parser=os.getenv("PARSER", "mineru"),
        parse_method=os.getenv("PARSE_METHOD", "auto"),
        enable_image_processing=True,
        enable_table_processing=True,
        enable_equation_processing=True,
    )
    
    if enable_lightrag:
        try:
            from lightrag.llm.openai import openai_complete_if_cache, openai_embed
            from lightrag.utils import EmbeddingFunc
            
            # Get API key from environment
            api_key = os.getenv("OPENAI_API_KEY") or os.getenv("LLM_BINDING_API_KEY")
            base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("LLM_BINDING_HOST")
            
            if not api_key:
                logger.warning("No API key found. Some features may not work.")
                api_key = "dummy-key"
            
            # Define LLM function
            def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
                return openai_complete_if_cache(
                    os.getenv("LLM_MODEL", "gpt-4o-mini"),
                    prompt,
                    system_prompt=system_prompt,
                    history_messages=history_messages,
                    api_key=api_key,
                    base_url=base_url,
                    **kwargs,
                )
            
            # Define embedding function
            embedding_model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-large")
            embedding_dim = int(os.getenv("EMBEDDING_DIM", "3072"))
            
            embedding_func = EmbeddingFunc(
                embedding_dim=embedding_dim,
                max_token_size=8192,
                func=lambda texts: openai_embed(
                    texts,
                    model=embedding_model,
                    api_key=api_key,
                    base_url=base_url,
                ),
            )
            
            # Initialize RAG-Anything with LightRAG
            rag_instance = RAGAnything(
                config=config,
                llm_model_func=llm_model_func,
                embedding_func=embedding_func,
            )
            logger.info("✅ RAG-Anything initialized with LightRAG")
        except ImportError:
            logger.warning("LightRAG not available, initializing without LightRAG")
            rag_instance = RAGAnything(
                config=config,
                llm_model_func=None,
                embedding_func=None,
            )
            rag_instance.lightrag = None
            logger.info("✅ RAG-Anything initialized (parsing only)")
    else:
        rag_instance = RAGAnything(
            config=config,
            llm_model_func=None,
            embedding_func=None,
        )
        rag_instance.lightrag = None
        logger.info("✅ RAG-Anything initialized (parsing only, LightRAG disabled)")
    
    # Set RAG instance for worker
    set_rag_instance(rag_instance)
    logger.info("✅ Document worker ready")
    
except Exception as e:
    logger.error(f"❌ Failed to initialize RAG-Anything: {str(e)}")
    logger.exception(e)
    sys.exit(1)

# Create queue
queue = Queue("document-processing", connection=redis_conn)

# Start worker
logger.info(f"Starting RQ worker for queue 'document-processing' (Redis: {redis_host}:{redis_port})")
logger.info(f"Max concurrent files: {os.getenv('MAX_CONCURRENT_FILES', '1')}")

if __name__ == "__main__":
    # In newer versions of rq, Connection is not needed
    # Worker can be created directly with the queue and connection
    worker = Worker(
        [queue],
        connection=redis_conn,
        name="document-processing-worker",
    )
    worker.work()

