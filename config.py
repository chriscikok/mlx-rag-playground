"""
Central config for MLX RAG stack - General purpose
"""
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
CHROMA_DIR = BASE_DIR / "chroma_db"
STORAGE_DIR = BASE_DIR / "storage"

# Models - change here to switch globally
LLM_MODEL_ID = "mlx-community/Qwen2.5-7B-Instruct-4bit"
GRADER_MODEL_ID = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
EMBED_MODEL_ID = "BAAI/bge-m3"  # 8192 tokens max, 1024 dim

# Chunking - Phase 1: SemanticSplitter + Table-aware
# bge-m3 handles 8192 tokens, but 512 tokens ~ 2000 chars is sweet spot for semantic chunks
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
CHUNKING_MODE = "semantic"  # semantic | sentence | hierarchical

# Retrieval
TOP_K_VECTOR = 15
TOP_K_BM25 = 15
TOP_K_RERANK = 8

# Generation
MAX_TOKENS = 768
TEMPERATURE = 0.1
CTX_SIZE = 2048

# Graph
GRAPH_ENABLED = False