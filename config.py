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

# Chunking - General purpose balanced for mixed docs
# For bge-m3, 512 tokens ~ 2000 chars is common, but smaller = higher precision
# 512 is good for general KB: preserves 2-3 paragraphs, tables, Q&A pairs
# If your docs are mostly short clauses (T&C), use 380-450
# If mostly long reports, use 600-800
CHUNK_SIZE = 180  # Increased from 100 to 180 to keep Online/Branch rows together (was splitting tables)
CHUNK_OVERLAP = 30

# Retrieval
TOP_K_VECTOR = 15  # More candidates for multi-hop (interest rate + min deposit are different chunks)
TOP_K_BM25 = 15
TOP_K_RERANK = 8  # Return 8 chunks to include table + conditions

# Generation
MAX_TOKENS = 768  # Longer for table reasoning
TEMPERATURE = 0.1  # Lower temp for precise number extraction
CTX_SIZE = 2048

# Graph
GRAPH_ENABLED = False