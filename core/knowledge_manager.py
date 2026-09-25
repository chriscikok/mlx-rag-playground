"""
Knowledge Manager - Sustainable incremental ingestion
Tracks file hashes, mtime, chunk counts in manifest.json
Supports incremental add/delete without full reindex
"""
import hashlib
import json
import time
from pathlib import Path
from typing import List, Dict, Tuple
import config

def get_file_hash(file_path: Path) -> str:
    """Fast hash of file for change detection"""
    h = hashlib.sha256()
    with open(file_path, 'rb') as f:
        # Read in chunks for large PDFs
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()[:16]

def get_manifest_path() -> Path:
    return Path(config.STORAGE_DIR) / "manifest.json"

def load_manifest() -> Dict:
    """Load manifest of indexed files"""
    mp = get_manifest_path()
    if not mp.exists():
        return {}
    try:
        with open(mp, 'r') as f:
            return json.load(f)
    except:
        return {}

def save_manifest(manifest: Dict):
    mp = get_manifest_path()
    mp.parent.mkdir(parents=True, exist_ok=True)
    with open(mp, 'w') as f:
        json.dump(manifest, f, indent=2)

def scan_data_dir() -> List[Dict]:
    """Scan data/ and return file status vs manifest"""
    data_dir = Path(config.DATA_DIR)
    manifest = load_manifest()
    
    supported = {".pdf", ".docx", ".md", ".txt", ".html"}
    files = []
    
    if not data_dir.exists():
        return []
    
    for fp in data_dir.rglob("*"):
        if not fp.is_file() or fp.suffix.lower() not in supported:
            continue
        # Skip hidden and temp
        if fp.name.startswith(".") or fp.name.startswith("~"):
            continue
        
        stat = fp.stat()
        fhash = get_file_hash(fp)
        rel_path = str(fp.relative_to(data_dir))
        
        existing = manifest.get(rel_path) or manifest.get(str(fp)) or manifest.get(fp.name)
        
        if not existing:
            status = "new"
        elif existing.get("hash") != fhash:
            status = "modified"
        else:
            status = "indexed"
        
        files.append({
            "name": fp.name,
            "relative_path": rel_path,
            "full_path": str(fp),
            "size": stat.st_size,
            "size_mb": round(stat.st_size / (1024*1024), 2),
            "mtime": stat.st_mtime,
            "mtime_str": time.strftime("%Y-%m-%d %H:%M", time.localtime(stat.st_mtime)),
            "hash": fhash,
            "status": status,
            "indexed_at": existing.get("indexed_at") if existing else None,
            "chunks": existing.get("chunks", 0) if existing else 0,
            "tables": existing.get("tables", 0) if existing else 0,
        })
    
    # Sort by mtime desc
    files.sort(key=lambda x: x["mtime"], reverse=True)
    return files

def get_indexed_files_summary():
    """Get summary stats from manifest"""
    manifest = load_manifest()
    total_chunks = sum(v.get("chunks",0) for v in manifest.values())
    total_tables = sum(v.get("tables",0) for v in manifest.values())
    return {
        "total_files": len(manifest),
        "total_chunks": total_chunks,
        "total_tables": total_tables,
        "files": manifest
    }

def update_manifest_entry(file_path: Path, chunks: int, tables: int = 0):
    manifest = load_manifest()
    data_dir = Path(config.DATA_DIR)
    try:
        rel = str(file_path.relative_to(data_dir))
    except:
        rel = file_path.name
    
    manifest[rel] = {
        "hash": get_file_hash(file_path),
        "full_path": str(file_path),
        "chunks": chunks,
        "tables": tables,
        "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "size": file_path.stat().st_size if file_path.exists() else 0
    }
    save_manifest(manifest)

def remove_manifest_entry(file_name: str):
    manifest = load_manifest()
    # Remove by key matching name or relative path
    keys_to_remove = [k for k in manifest.keys() if k == file_name or k.endswith(file_name) or Path(k).name == file_name]
    for k in keys_to_remove:
        del manifest[k]
    save_manifest(manifest)
    return len(keys_to_remove)

def get_files_to_process(mode="incremental") -> Tuple[List[Path], List[Dict]]:
    """Get list of files that need processing based on mode"""
    scanned = scan_data_dir()
    if mode == "full":
        return [Path(f["full_path"]) for f in scanned], scanned
    elif mode == "incremental":
        # Only new or modified
        to_process = [Path(f["full_path"]) for f in scanned if f["status"] in ("new", "modified")]
        return to_process, scanned
    else:
        return [], scanned