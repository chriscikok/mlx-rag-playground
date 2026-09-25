"""
Embeddings - torch-enabled version
Uses BAAI/bge-m3 on MPS (Apple Silicon) with torch, falls back to fastembed ONNX
Fixed SSL issue for macOS Python 3.14
"""
import config
from typing import List
import os
import ssl
import certifi

# Fix SSL cert issue on macOS Python.org installs
# This sets SSL cert path for HuggingFace downloads
try:
    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['REQUESTS_CA_BUNDLE'] = certifi.where()
    ssl_context = ssl.create_default_context(cafile=certifi.where())
except:
    pass

def get_embed_model():
    """
    Try HuggingFace BGE-M3 with torch MPS first (best quality),
    fall back to fastembed ONNX if torch not available
    """
    try:
        print(f"[Embeddings] Loading {config.EMBED_MODEL_ID} on mps (torch)...")
        from llama_index.embeddings.huggingface import HuggingFaceEmbedding
        return HuggingFaceEmbedding(
            model_name=config.EMBED_MODEL_ID,
            device="mps",
            trust_remote_code=True,
            cache_folder=str(config.BASE_DIR / "models_cache")
        )
    except Exception as e:
        print(f"[Embeddings] HuggingFaceEmbedding failed: {e}")
        print("[Embeddings] Falling back to fastembed ONNX...")
        try:
            from fastembed import TextEmbedding
            from llama_index.core.embeddings import BaseEmbedding

            class FastEmbedWrapper(BaseEmbedding):
                def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5"):
                    super().__init__()
                    self._model = TextEmbedding(model_name=model_name)
                def _get_text_embedding(self, text: str) -> List[float]:
                    return list(self._model.embed([text]))[0].tolist()
                def _get_query_embedding(self, query: str) -> List[float]:
                    return list(self._model.query_embed([query]))[0].tolist()
                def _get_text_embeddings(self, texts: List[str]) -> List[List[float]]:
                    return [e.tolist() for e in self._model.embed(texts)]
                async def _aget_text_embedding(self, text: str) -> List[float]:
                    return self._get_text_embedding(text)
                async def _aget_query_embedding(self, query: str) -> List[float]:
                    return self._get_query_embedding(query)

            return FastEmbedWrapper()
        except Exception as e2:
            print(f"[Embeddings] fastembed also failed: {e2}")
            raise e