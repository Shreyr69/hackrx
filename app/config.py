import os
from dotenv import load_dotenv

load_dotenv()

# Authorization token required by the API per spec
REQUIRED_BEARER_TOKEN = (
    "043dc79bbd910f6e4ea9b57b6705a94ee0677b8b3c80080823b643987dd73fe0"
)

# OpenAI API key must be provided via environment variable
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# OpenAI model configuration - Using the latest and most powerful model
OPENAI_MODEL = "gpt-4o"  # Latest GPT-4o model for best accuracy
OPENAI_MAX_TOKENS = 2000  # Increased for more detailed responses
OPENAI_TEMPERATURE = 0.3  # Balanced temperature for accuracy and creativity

# Retrieval / chunking defaults - OPTIMIZED for better accuracy and speed
DEFAULT_CHUNK_WORDS = 100  # Reduced from 150 for more precise retrieval
DEFAULT_CHUNK_OVERLAP_WORDS = 20  # Reduced from 30 for efficiency
TOP_K = 12  # Increased from 8 for better coverage and accuracy

# Timeouts - OPTIMIZED
HTTP_TIMEOUT_SECS = 30  # Reduced from 60 for faster failure detection

# Port for deployment (Render sets PORT env var)
PORT = int(os.getenv("PORT", 8000))

# NEW: Performance optimizations
MAX_CONCURRENT_LLM_CALLS = 3  # Limit concurrent LLM calls
CHUNK_SIMILARITY_THRESHOLD = 0.25  # Balanced threshold for better retrieval coverage

# NEW: Caching configuration
ENABLE_CACHING = True  # Enable response and embedding caching
CACHE_SIZE_LIMIT = 1000  # Maximum cache entries
CACHE_TTL_HOURS = 24  # Cache time-to-live in hours
