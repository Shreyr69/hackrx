import os
from dotenv import load_dotenv

load_dotenv()

# Authorization token required by the API per spec
REQUIRED_BEARER_TOKEN = (
    "043dc79bbd910f6e4ea9b57b6705a94ee0677b8b3c80080823b643987dd73fe0"
)

# Gemini API key must be provided via environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Retrieval / chunking defaults - OPTIMIZED for better accuracy and speed
DEFAULT_CHUNK_WORDS = 100  # Reduced from 150 for more precise retrieval
DEFAULT_CHUNK_OVERLAP_WORDS = 20  # Reduced from 30 for efficiency
TOP_K = 8  # Increased from 6 for better coverage

# Timeouts - OPTIMIZED
HTTP_TIMEOUT_SECS = 30  # Reduced from 60 for faster failure detection

# Port for deployment (Render sets PORT env var)
PORT = int(os.getenv("PORT", 8000))

# NEW: Performance optimizations
MAX_CONCURRENT_LLM_CALLS = 3  # Limit concurrent LLM calls
CHUNK_SIMILARITY_THRESHOLD = 0.3  # Lowered from 0.7 for better retrieval coverage
