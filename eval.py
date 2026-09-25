"""
Run Phase 1 evaluation with external test set - Sustainable version
Test sets live in test/ folder (gitignored) - must pass json file every time
Now supports incremental index: reuse existing DB unless --full-reindex
"""
import argparse
from pathlib import Path
import time
import config
from core.evaluation import load_test_set, run_evaluation
from core.embeddings import get_embed_model
from core.vector_store import get_index, force_delete_chroma, add_nodes_to_index
from core.ingestion import load_and_chunk
from core.llm import MLXLLM
from core.pipeline import RAGPipeline
from core.knowledge_manager import get_files_to_process, update_manifest_entry, load_manifest

def main():
    parser = argparse.ArgumentParser(description="Phase 1 RAG Evaluation - sustainable incremental")
    parser.add_argument("-t", "--test-file", type=str, required=True,
                        help="Path to test set json file (e.g. test/msa_test.json) - REQUIRED")
    parser.add_argument("-o", "--output", type=str, default="eval_results_phase1.json",
                        help="Output json for results")
    parser.add_argument("--full-reindex", action="store_true",
                        help="Clear DB and reindex everything (default: incremental)")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Disable reranker for faster eval")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Custom data dir (default: config.DATA_DIR)")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"ERROR: Test file not found: {test_path}")
        return 1

    print("=== Phase 1 RAG Evaluation (Sustainable) ===")
    print(f"Test file: {test_path}")
    
    dataset = load_test_set(test_path)
    print(f"Dataset: {len(dataset)} questions")

    data_dir = Path(args.data_dir) if args.data_dir else Path(config.DATA_DIR)
    
    embed_model = get_embed_model()

    if args.full_reindex:
        print("\n[Eval] Full reindex requested - clearing DB...")
        try:
            force_delete_chroma()
            time.sleep(0.5)
        except Exception as e:
            print(f"Clear warning: {e}")
        docs, nodes = load_and_chunk(data_dir=data_dir, embed_model=embed_model)
        print(f"Loaded {len(docs)} docs -> {len(nodes)} chunks")
        index = get_index(nodes=nodes, embed_model=embed_model)
        # Update manifest for all files
        from core.knowledge_manager import scan_data_dir
        for f in scan_data_dir():
            update_manifest_entry(Path(f["full_path"]), f.get("chunks",0), f.get("tables",0))
        # Actually update with real counts per file
        for doc in docs:
            # We approximate - manifest will be corrected on next scan
            pass
        # Rebuild manifest correctly
        manifest = {}
        for node in nodes:
            fname = node.metadata.get("file_name", "unknown")
            # Count per file
            pass
        # Simple: rescan will fix hash but we need to set chunks per file
        from collections import Counter
        file_chunk_counts = Counter([n.metadata.get("file_name","") for n in nodes])
        file_table_counts = Counter([n.metadata.get("file_name","") for n in nodes if n.metadata.get("is_table")])
        from core.knowledge_manager import get_file_hash
        import json
        new_manifest = {}
        for fp in data_dir.rglob("*"):
            if fp.suffix.lower() in config.SUPPORTED_EXTS and fp.is_file():
                rel = str(fp.relative_to(data_dir))
                new_manifest[rel] = {
                    "hash": get_file_hash(fp),
                    "full_path": str(fp),
                    "chunks": file_chunk_counts.get(fp.name,0),
                    "tables": file_table_counts.get(fp.name,0),
                    "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "size": fp.stat().st_size
                }
        from core.knowledge_manager import save_manifest
        save_manifest(new_manifest)
        print(f"[Manifest] Saved {len(new_manifest)} files")
    else:
        # Incremental mode: only process new/modified files
        to_process, scanned = get_files_to_process(mode="incremental")
        if to_process:
            print(f"\n[Eval] Incremental: {len(to_process)} new/modified files to index:")
            for p in to_process:
                print(f"  - {p.name}")
            # Delete old chunks for modified files first
            from core.vector_store import delete_by_file_name
            for p in to_process:
                # Check if modified (exists in manifest)
                existing = [s for s in scanned if s["full_path"]==str(p) and s["status"]=="modified"]
                if existing:
                    delete_by_file_name(p.name)
            docs, nodes = load_and_chunk(specific_files=to_process, embed_model=embed_model)
            print(f"Loaded {len(docs)} new docs -> {len(nodes)} new chunks")
            if nodes:
                index = add_nodes_to_index(nodes, embed_model=embed_model)
                if index is None:
                    # Fallback to get_index
                    index = get_index(nodes=nodes, embed_model=embed_model)
                # Update manifest
                from collections import Counter
                file_chunk_counts = Counter([n.metadata.get("file_name","") for n in nodes])
                file_table_counts = Counter([n.metadata.get("file_name","") for n in nodes if n.metadata.get("is_table")])
                for p in to_process:
                    update_manifest_entry(p, file_chunk_counts.get(p.name,0), file_table_counts.get(p.name,0))
            else:
                print("[Eval] No new nodes, using existing index")
                index = get_index(embed_model=embed_model)
        else:
            print("\n[Eval] No new files - using existing index (sustainable)")
            print(f"Indexed files: {len(load_manifest())}")
            index = get_index(embed_model=embed_model)
            # Need nodes for BM25 - try to reconstruct from manifest? 
            # For BM25 we need actual nodes - load_and_chunk all files but don't reindex, just for retriever nodes
            # We will load nodes from data dir without indexing
            _, nodes = load_and_chunk(data_dir=data_dir, embed_model=embed_model)
            print(f"Loaded {len(nodes)} nodes for hybrid retriever (not reindexed)")

    llm = MLXLLM()
    # For pipeline we need nodes for BM25 - if incremental we still need all nodes
    if 'nodes' not in locals() or not nodes:
        _, nodes = load_and_chunk(data_dir=data_dir, embed_model=embed_model)
    
    pipeline = RAGPipeline(
        index=index, llm=llm, nodes=nodes,
        use_hybrid=True, use_crag=False, use_reranker=not args.no_reranker
    )

    summary, results = run_evaluation(pipeline, dataset=dataset, save_path=args.output)

    print("\nFailed questions:")
    for r in results:
        if not r.get("passed"):
            print(f"- {r['question_id']}: score {r.get('overall_score',0):.2f} miss {r.get('kw_miss',[])}")
            print(f"  Q: {r.get('question')}")
            print(f"  A: {r.get('generated_answer','')[:200]}...")

    print(f"\nDone. Results saved to {args.output}")
    return 0

if __name__ == "__main__":
    exit(main())