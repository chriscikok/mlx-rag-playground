"""
Central config for MLX RAG stack - General purpose
"""
from pathlib import Path
import hashlib

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
STORAGE_DIR = BASE_DIR / "storage"
TEST_DIR = BASE_DIR / "test"

MANIFEST_FILE = STORAGE_DIR / "manifest.json"
NODES_CACHE_FILE = STORAGE_DIR / "nodes_cache.pkl"

# Models - change here to switch globally
LLM_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
GRADER_MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
EMBED_MODEL_ID = "BAAI/bge-m3"  # 8192 tokens max, 1024 dim

# Chunking
CHUNK_SIZE = 600
CHUNK_OVERLAP = 100
CHUNKING_MODE = "semantic"  # semantic | sentence

# Retrieval
TOP_K_VECTOR = 10
TOP_K_BM25 = 10
TOP_K_RERANK = 4

# Generation
MAX_TOKENS = 512
TEMPERATURE = 0.2
CTX_SIZE = 2048

# Graph
GRAPH_ENABLED = False

# Supported file types - THIS WAS MISSING
SUPPORTED_EXTS = {".pdf", ".docx", ".md", ".txt", ".html"}

def get_config_version() -> str:
    raw = f"{CHUNK_SIZE}_{CHUNK_OVERLAP}_{CHUNKING_MODE}_{EMBED_MODEL_ID}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]

CONFIG_VERSION = get_config_version()