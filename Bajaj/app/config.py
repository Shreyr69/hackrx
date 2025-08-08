import os
from dotenv import load_dotenv

load_dotenv()

# Authorization token required by the API per spec
REQUIRED_BEARER_TOKEN = (
    "043dc79bbd910f6e4ea9b57b6705a94ee0677b8b3c80080823b643987dd73fe0"
)

# Gemini API key must be provided via environment variable
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

# Retrieval / chunking defaults
DEFAULT_CHUNK_WORDS = 150
DEFAULT_CHUNK_OVERLAP_WORDS = 30
TOP_K = 6

# Timeouts
HTTP_TIMEOUT_SECS = 60
