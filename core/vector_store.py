
"""
Vector store module - Chroma wrapper
Fixed: Handles readonly DB error (1032) - robust version with no client open on delete
"""
import chromadb
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core import VectorStoreIndex, StorageContext, Settings
import config
from .embeddings import get_embed_model
import shutil
import os
import time
from pathlib import Path

def _fix_permissions(path: Path):
    """Fix readonly permissions recursively"""
    try:
        if not path.exists():
            return
        for root, dirs, files in os.walk(path):
            for d in dirs:
                try:
                    os.chmod(os.path.join(root, d), 0o777)
                except:
                    pass
            for f in files:
                try:
                    os.chmod(os.path.join(root, f), 0o666)
                except:
                    pass
        try:
            os.chmod(path, 0o777)
        except:
            pass
    except Exception as e:
        print(f"[Chroma] Permission fix failed: {e}")

def _on_rm_error(func, path, exc_info):
    """Error handler for shutil.rmtree - chmod and retry"""
    try:
        os.chmod(path, 0o777)
    except:
        pass
    try:
        func(path)
    except Exception as e:
        print(f"[Chroma] rm error handler failed for {path}: {e}")

def _force_delete_chroma():
    """Force delete chroma_db WITHOUT opening a client first (avoids lock)"""
    chroma_dir = Path(config.CHROMA_DIR)
    storage_dir = Path(config.STORAGE_DIR)
    
    print(f"[Chroma] Force deleting {chroma_dir} and {storage_dir}")
    
    # 1. Fix permissions first
    _fix_permissions(chroma_dir)
    _fix_permissions(storage_dir)
    
    # 2. Try to remove sqlite WAL files that cause readonly error
    # These are often locked: chroma.sqlite, chroma.sqlite-wal, chroma.sqlite-shm
    for target_dir in [chroma_dir, storage_dir]:
        if target_dir.exists():
            # Remove individual files first to break locks
            for pattern in ["*.sqlite", "*.sqlite-wal", "*.sqlite-shm", "*.bin", "*.pkl"]:
                for f in target_dir.rglob(pattern):
                    try:
                        os.chmod(f, 0o666)
                        f.unlink()
                        print(f"[Chroma] Deleted locked file {f}")
                    except Exception as e:
                        print(f"[Chroma] Could not delete {f}: {e}")
    
    # 3. Now delete directories with error handler
    for target_dir in [chroma_dir, storage_dir]:
        if target_dir.exists():
            try:
                shutil.rmtree(target_dir, onerror=_on_rm_error)
                print(f"[Chroma] Deleted {target_dir}")
            except Exception as e:
                print(f"[Chroma] Delete failed for {target_dir}: {e}, retrying with ignore_errors")
                try:
                    shutil.rmtree(target_dir, ignore_errors=True)
                except:
                    pass
            # Final check - if still exists, try os removal
            if target_dir.exists():
                try:
                    # Last resort: rename then delete
                    backup = target_dir.parent / f"{target_dir.name}_backup_{int(time.time())}"
                    target_dir.rename(backup)
                    shutil.rmtree(backup, ignore_errors=True)
                    print(f"[Chroma] Renamed and deleted {target_dir} as {backup}")
                except Exception as e:
                    print(f"[Chroma] Final delete failed: {e}")
    
    # Small delay to let OS release file handles
    time.sleep(0.5)

def get_chroma_collection():
    """Get or create collection - never tries to open client if directory is broken"""
    chroma_dir = Path(config.CHROMA_DIR)
    
    # Ensure dir exists
    try:
        chroma_dir.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"[Chroma] Could not create dir: {e}")
    
    _fix_permissions(chroma_dir)
    
    # Try normal path
    try:
        db = chromadb.PersistentClient(path=str(chroma_dir))
        collection = db.get_or_create_collection("mlx_kb")
        return collection
    except Exception as e:
        err_str = str(e).lower()
        is_readonly = "readonly" in err_str or "1032" in str(e) or "attempt to write a readonly database" in err_str
        
        if is_readonly:
            print(f"[Chroma] Readonly error detected: {e}")
            print("[Chroma] Force deleting and recreating...")
            _force_delete_chroma()
            chroma_dir.mkdir(parents=True, exist_ok=True)
            _fix_permissions(chroma_dir)
            try:
                db = chromadb.PersistentClient(path=str(chroma_dir))
                return db.get_or_create_collection("mlx_kb")
            except Exception as e2:
                print(f"[Chroma] Still failing after force delete: {e2}")
                # Last resort: use ephemeral client in memory (won't persist but won't crash)
                print("[Chroma] Using ephemeral client as fallback")
                db = chromadb.EphemeralClient()
                return db.get_or_create_collection("mlx_kb")
        elif "dimension" in err_str or "embedding" in err_str:
            print(f"[Chroma] Dimension mismatch: {e}, recreating...")
            _force_delete_chroma()
            chroma_dir.mkdir(parents=True, exist_ok=True)
            db = chromadb.PersistentClient(path=str(chroma_dir))
            return db.get_or_create_collection("mlx_kb")
        else:
            raise

def get_index(nodes=None, embed_model=None):
    embed_model = embed_model or get_embed_model()
    Settings.embed_model = embed_model
    
    try:
        collection = get_chroma_collection()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        
        if nodes:
            print(f"[Chroma] Indexing {len(nodes)} nodes...")
            storage_context = StorageContext.from_defaults(vector_store=vector_store)
            index = VectorStoreIndex(nodes, storage_context=storage_context)
            try:
                index.storage_context.persist(persist_dir=str(config.STORAGE_DIR))
            except Exception as e:
                if "readonly" in str(e).lower() or "1032" in str(e):
                    print(f"[Chroma] Readonly during persist: {e}")
                    _fix_permissions(Path(config.STORAGE_DIR))
                    # Try to persist again, if fails, skip persist (index is already in memory)
                    try:
                        index.storage_context.persist(persist_dir=str(config.STORAGE_DIR))
                    except Exception as e2:
                        print(f"[Chroma] Persist still failing, continuing with in-memory index: {e2}")
                else:
                    raise
        else:
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        
        return index
    except Exception as e:
        if "readonly" in str(e).lower() or "1032" in str(e):
            print(f"[Chroma] Readonly error in get_index: {e}")
            print("[Chroma] Force clearing and retrying...")
            _force_delete_chroma()
            try:
                collection = get_chroma_collection()
                vector_store = ChromaVectorStore(chroma_collection=collection)
                if nodes:
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    index = VectorStoreIndex(nodes, storage_context=storage_context)
                    try:
                        index.storage_context.persist(persist_dir=str(config.STORAGE_DIR))
                    except:
                        pass
                    return index
                else:
                    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
            except Exception as e2:
                print(f"[Chroma] Retry failed, using ephemeral: {e2}")
                db = chromadb.EphemeralClient()
                collection = db.get_or_create_collection("mlx_kb")
                vector_store = ChromaVectorStore(chroma_collection=collection)
                if nodes:
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)
                    return VectorStoreIndex(nodes, storage_context=storage_context)
                else:
                    return VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        raise

# Public aliases
def force_delete_chroma():
    return _force_delete_chroma()

def fix_permissions(path):
    return _fix_permissions(path)

__all__ = ["get_index", "get_chroma_collection", "_fix_permissions", "_force_delete_chroma", "force_delete_chroma", "fix_permissions"]