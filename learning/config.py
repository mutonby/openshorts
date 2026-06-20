import os

LEARNING_ENABLED = os.getenv("LEARNING_ENABLED", "true").lower() in ("1", "true", "yes")
LEARNING_DATA_DIR = os.getenv("LEARNING_DATA_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chromadb"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() in ("1", "true", "yes")
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")

RAG_TOP_K = int(os.getenv("RAG_TOP_K", "8"))
RAG_MIN_PROFILE_CLIPS = int(os.getenv("RAG_MIN_PROFILE_CLIPS", "3"))
RAG_SUCCESS_THRESHOLD = float(os.getenv("RAG_SUCCESS_THRESHOLD", "0.05"))

weights_str = os.getenv("RAG_RERANK_WEIGHTS", "0.4,0.3,0.3")
RAG_RERANK_WEIGHTS = [float(w) for w in weights_str.split(",")]

PROFILES_FILE = os.getenv(
    "LEARNING_PROFILES_FILE",
    os.path.join(os.path.dirname(LEARNING_DATA_DIR), "profiles.json"),
)
