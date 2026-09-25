"""
Vector store module - Chroma wrapper
Fixed: Handles readonly DB error (1032) - robust version with no client open on delete
Supports incremental add/delete per file for sustainable knowledge management
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
    _fix_permissions(chroma_dir)
    _fix_permissions(storage_dir)
    for target_dir in [chroma_dir, storage_dir]:
        if target_dir.exists():
            for pattern in ["*.sqlite", "*.sqlite-wal", "*.sqlite-shm", "*.bin", "*.pkl"]:
                for f in target_dir.rglob(pattern):
                    try:
                        os.chmod(f, 0o666)
                        f.unlink()
                    except:
                        pass
    for target_dir in [chroma_dir, storage_dir]:
        if target_dir.exists():
            try:
                shutil.rmtree(target_dir, onerror=_on_rm_error)
            except:
                try:
                    shutil.rmtree(target_dir, ignore_errors=True)
                except:
                    pass
            if target_dir.exists():
                try:
                    backup = target_dir.parent / f"{target_dir.name}_backup_{int(time.time())}"
                    target_dir.rename(backup)
                    shutil.rmtree(backup, ignore_errors=True)
                except:
                    pass
    time.sleep(0.5)

def get_chroma_collection():
    chroma_dir = Path(config.CHROMA_DIR)
    try:
        chroma_dir.mkdir(parents=True, exist_ok=True)
    except:
        pass
    _fix_permissions(chroma_dir)
    try:
        db = chromadb.PersistentClient(path=str(chroma_dir))
        collection = db.get_or_create_collection("mlx_kb")
        return collection
    except Exception as e:
        err_str = str(e).lower()
        is_readonly = "readonly" in err_str or "1032" in str(e) or "attempt to write a readonly database" in err_str
        if is_readonly:
            print(f"[Chroma] Readonly error detected: {e}")
            _force_delete_chroma()
            chroma_dir.mkdir(parents=True, exist_ok=True)
            _fix_permissions(chroma_dir)
            try:
                db = chromadb.PersistentClient(path=str(chroma_dir))
                return db.get_or_create_collection("mlx_kb")
            except Exception as e2:
                print(f"[Chroma] Still failing, using ephemeral: {e2}")
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

def delete_by_file_name(file_name: str) -> int:
    """Delete all chunks for a given file_name from Chroma - for sustainable updates"""
    try:
        collection = get_chroma_collection()
        # Chroma stores metadata - query for file_name
        results = collection.get(where={"file_name": file_name})
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            print(f"[Chroma] Deleted {len(ids)} chunks for file {file_name}")
            return len(ids)
        # Try alternative metadata key
        results = collection.get(where={"file_name": {"$eq": file_name}})
        ids = results.get("ids", [])
        if ids:
            collection.delete(ids=ids)
            print(f"[Chroma] Deleted {len(ids)} chunks for file {file_name} (eq)")
            return len(ids)
        # Fallback: scan all and filter by metadata in python
        all_data = collection.get()
        to_delete = []
        for i, meta in enumerate(all_data.get("metadatas", [])):
            if meta and meta.get("file_name") == file_name:
                to_delete.append(all_data["ids"][i])
        if to_delete:
            collection.delete(ids=to_delete)
            print(f"[Chroma] Deleted {len(to_delete)} chunks for file {file_name} (scan)")
            return len(to_delete)
        print(f"[Chroma] No chunks found for {file_name} to delete")
        return 0
    except Exception as e:
        print(f"[Chroma] Delete by file failed for {file_name}: {e}")
        import traceback; traceback.print_exc()
        return 0

def add_nodes_to_index(nodes, embed_model=None):
    """Incrementally add nodes to existing index (no clear)"""
    if not nodes:
        return None
    embed_model = embed_model or get_embed_model()
    Settings.embed_model = embed_model
    try:
        collection = get_chroma_collection()
        vector_store = ChromaVectorStore(chroma_collection=collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex(nodes, storage_context=storage_context)
        try:
            index.storage_context.persist(persist_dir=str(config.STORAGE_DIR))
        except:
            pass
        print(f"[Chroma] Incrementally added {len(nodes)} nodes")
        return index
    except Exception as e:
        print(f"[Chroma] add_nodes failed: {e}")
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
                    _fix_permissions(Path(config.STORAGE_DIR))
                    try:
                        index.storage_context.persist(persist_dir=str(config.STORAGE_DIR))
                    except:
                        pass
                else:
                    raise
        else:
            index = VectorStoreIndex.from_vector_store(vector_store=vector_store, embed_model=embed_model)
        return index
    except Exception as e:
        if "readonly" in str(e).lower() or "1032" in str(e):
            print(f"[Chroma] Readonly error in get_index: {e}, force clearing...")
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

__all__ = ["get_index", "get_chroma_collection", "_fix_permissions", "_force_delete_chroma", "force_delete_chroma", "fix_permissions", "delete_by_file_name", "add_nodes_to_index"]