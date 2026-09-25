"""
Run Phase 1 evaluation with external test set
Test sets live in test/ folder (gitignored) - must pass json file every time

Usage:
  python eval.py --test-file test/msa_test.json
  python eval.py --test-file test/my_new_doc.json --output eval_results_my_doc.json
  python eval.py -t test/msa_test.json -o eval_results.json --no-clear-db

Test file format (JSON list):
[
  {
    "id": "q1",
    "question": "What is the HKD rate for Phase 1?",
    "expected_keywords": ["2.70% p.a", "HKD"],
    "expected_numbers": ["2.70%"],
    "category": "table-numeric",
    "difficulty": "medium",
    "phase1_focus": "Camelot table extraction"
  }
]

Template: see test/template.json
"""
import argparse
import json
from pathlib import Path
import time

import config
from core.evaluation import load_test_set, run_evaluation
from core.embeddings import get_embed_model
from core.vector_store import get_index, force_delete_chroma
from core.ingestion import load_and_chunk
from core.llm import MLXLLM
from core.pipeline import RAGPipeline

def main():
    parser = argparse.ArgumentParser(description="Phase 1 RAG Evaluation - requires external test json")
    parser.add_argument("-t", "--test-file", type=str, required=True,
                        help="Path to test set json file (e.g. test/msa_test.json) - REQUIRED")
    parser.add_argument("-o", "--output", type=str, default="eval_results_phase1.json",
                        help="Output json for results (default: eval_results_phase1.json)")
    parser.add_argument("--no-clear-db", action="store_true",
                        help="Skip clearing chroma_db (use existing index)")
    parser.add_argument("--no-reranker", action="store_true",
                        help="Disable reranker for faster eval")
    args = parser.parse_args()

    test_path = Path(args.test_file)
    if not test_path.exists():
        print(f"ERROR: Test file not found: {test_path}")
        print(f"Create test/ folder with your json file. See test/template.json")
        print(f"Example: python eval.py --test-file test/msa_test.json")
        return 1

    print("=== Phase 1 RAG Evaluation ===")
    print(f"Test file: {test_path}")
    
    dataset = load_test_set(test_path)
    print(f"Dataset: {len(dataset)} questions")

    if not args.no_clear_db:
        print("\n[Eval] Clearing old chroma_db to avoid stale chunks...")
        try:
            force_delete_chroma()
            time.sleep(0.5)
            print("[Eval] Old DB cleared")
        except Exception as e:
            print(f"[Eval] Clear DB warning: {e}")
    else:
        print("\n[Eval] Keeping existing chroma_db (--no-clear-db)")

    embed_model = get_embed_model()
    docs, nodes = load_and_chunk(embed_model=embed_model)
    print(f"Loaded {len(docs)} docs -> {len(nodes)} chunks")
    print(f"Table chunks: {sum(1 for n in nodes if n.metadata.get('is_table'))}")

    index = get_index(nodes=nodes, embed_model=embed_model)
    llm = MLXLLM()
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