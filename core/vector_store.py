"""
Vector store - Sustainable - fixes readonly 1032
"""
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
import config
from .embeddings import get_embed_model
import shutil, os, time
from pathlib import Path

COLLECTION = "rag_collection"

def _fix_permissions(path: Path):
    try:
        if not path.exists():
            return
        for root, dirs, files in os.walk(path):
            for d in dirs:
                try: os.chmod(os.path.join(root, d), 0o777)
                except: pass
            for f in files:
                try: os.chmod(os.path.join(root, f), 0o666)
                except: pass
        os.chmod(path, 0o777)
    except: pass

def _force_delete_chroma():
    print(f"[Chroma] Force deleting {config.CHROMA_DIR}")
    _fix_permissions(Path(config.CHROMA_DIR))
    # Kill wal/shm locks first
    for pat in ["*.sqlite3", "*.sqlite3-wal", "*.sqlite3-shm", "*.sqlite"]:
        for f in Path(config.CHROMA_DIR).rglob(pat):
            try:
                os.chmod(f, 0o666)
                f.unlink()
            except: pass
    shutil.rmtree(config.CHROMA_DIR, ignore_errors=True)
    time.sleep(0.3)
    Path(config.CHROMA_DIR).mkdir(parents=True, exist_ok=True)

def get_chroma_collection():
    chroma_dir = Path(config.CHROMA_DIR)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    _fix_permissions(chroma_dir)
    try:
        db = chromadb.PersistentClient(path=str(chroma_dir))
        # FIX: use name= kwarg for new chroma API
        try:
            return db.get_or_create_collection(name=COLLECTION)
        except TypeError:
            return db.get_or_create_collection(COLLECTION)
    except Exception as e:
        if "readonly" in str(e).lower() or "1032" in str(e):
            print(f"[Chroma] readonly detected, fixing perms and retrying: {e}")
            _fix_permissions(chroma_dir)
            # delete locks and retry
            for pat in ["*.sqlite3-wal", "*.sqlite3-shm"]:
                for f in chroma_dir.rglob(pat):
                    try: f.unlink()
                    except: pass
            db = chromadb.PersistentClient(path=str(chroma_dir))
            try:
                return db.get_or_create_collection(name=COLLECTION)
            except TypeError:
                return db.get_or_create_collection(COLLECTION)
        raise

def get_collection_count() -> int:
    try:
        return get_chroma_collection().count()
    except:
        return 0

def delete_by_file_name(file_name: str) -> int:
    try:
        col = get_chroma_collection()
        # Try where filter first
        try:
            res = col.get(where={"file_name": file_name})
            ids = res.get("ids", [])
            if ids:
                col.delete(ids=ids)
                return len(ids)
        except: pass
        # Fallback scan
        all_data = col.get()
        to_del = [all_data["ids"][i] for i, meta in enumerate(all_data.get("metadatas", [])) if meta and meta.get("file_name")==file_name]
        if to_del:
            col.delete(ids=to_del)
            return len(to_del)
        return 0
    except Exception as e:
        print(f"[Chroma] Delete failed {e}")
        return 0

def add_nodes_to_index(nodes, embed_model=None):
    if not nodes:
        return None
    embed_model = embed_model or get_embed_model()
    Settings.embed_model = embed_model

    # FIX 1032: ensure perms before add
    _fix_permissions(Path(config.CHROMA_DIR))

    for attempt in range(2):
        try:
            col = get_chroma_collection()
            vs = ChromaVectorStore(chroma_collection=col)
            sc = StorageContext.from_defaults(vector_store=vs)
            index = VectorStoreIndex(nodes, storage_context=sc)
            return index
        except Exception as e:
            if "readonly" in str(e).lower() or "1032" in str(e) and attempt==0:
                print(f"[Chroma] add_nodes failed readonly, fixing and retry: {e}")
                _fix_permissions(Path(config.CHROMA_DIR))
                for pat in ["*.sqlite3-wal", "*.sqlite3-shm"]:
                    for f in Path(config.CHROMA_DIR).rglob(pat):
                        try: f.unlink()
                        except: pass
                time.sleep(0.5)
                continue
            print(f"[Chroma] add_nodes failed: {e}")
            raise

def get_index(nodes=None, embed_model=None):
    embed_model = embed_model or get_embed_model()
    Settings.embed_model = embed_model
    col = get_chroma_collection()
    vs = ChromaVectorStore(chroma_collection=col)
    if nodes:
        sc = StorageContext.from_defaults(vector_store=vs)
        index = VectorStoreIndex(nodes, storage_context=sc, embed_model=embed_model)
    else:
        index = VectorStoreIndex.from_vector_store(vector_store=vs, embed_model=embed_model)
    return index

def force_delete_chroma():
    return _force_delete_chroma()

def fix_permissions():  # <-- THIS WAS MISSING
    _fix_permissions(Path(config.CHROMA_DIR))
    _fix_permissions(Path(config.STORAGE_DIR))

# Aliases for old code
def clear_nodes_cache():
    pass

def get_indexed_files_summary():
    return {"total_files": 0, "total_chunks": get_collection_count()}