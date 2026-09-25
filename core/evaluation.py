"""
Phase 1 Evaluation: Automatic RAG evaluation
Tests Camelot table extraction + SemanticSplitter + Retrieval + Generation
Test sets are externalized to test/ folder (gitignored) - pass json file at runtime
"""
from pathlib import Path
import json
import re
from typing import List, Dict

def load_test_set(json_path: str | Path) -> List[Dict]:
    """Load external test set json file"""
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Test file not found: {p.absolute()} - create test/ folder with your json")
    with open(p, 'r', encoding='utf-8') as f:
        data = json.load(f)
    # Support both list and dict with 'dataset' key
    if isinstance(data, dict) and 'dataset' in data:
        data = data['dataset']
    if not isinstance(data, list):
        raise ValueError(f"Test file must be a list of questions, got {type(data)}")
    print(f"[Eval] Loaded {len(data)} questions from {p}")
    # Validate basic schema
    for i, q in enumerate(data):
        if 'id' not in q or 'question' not in q:
            raise ValueError(f"Question {i} missing 'id' or 'question': {q}")
        q.setdefault('expected_keywords', [])
        q.setdefault('expected_numbers', [])
        q.setdefault('category', 'general')
    return data

def normalize_text(s: str) -> str:
    return re.sub(r'\s+', ' ', s.lower().strip())

def evaluate_answer(question: Dict, generated_answer: str, retrieved_nodes: List = None) -> Dict:
    ans_norm = normalize_text(generated_answer)
    expected_kw = question.get("expected_keywords", [])
    expected_nums = question.get("expected_numbers", [])
    
    kw_hits = []
    kw_miss = []
    for kw in expected_kw:
        if normalize_text(kw) in ans_norm:
            kw_hits.append(kw)
        else:
            kw_miss.append(kw)
    kw_recall = len(kw_hits) / len(expected_kw) if expected_kw else 1.0
    
    num_hits = []
    num_miss = []
    for num in expected_nums:
        if normalize_text(num) in ans_norm:
            num_hits.append(num)
        else:
            num_miss.append(num)
    num_recall = len(num_hits) / len(expected_nums) if expected_nums else 1.0
    
    overall = 0.6 * kw_recall + 0.4 * num_recall
    
    has_table = False
    avg_rerank = 0.0
    if retrieved_nodes:
        has_table = any(n.metadata.get('is_table') for n in retrieved_nodes if hasattr(n, 'metadata'))
        scores = [n.score for n in retrieved_nodes if hasattr(n, 'score')]
        avg_rerank = sum(scores)/len(scores) if scores else 0.0
    
    passed = overall >= 0.5
    
    return {
        "question_id": question["id"],
        "question": question["question"],
        "kw_recall": kw_recall,
        "kw_hits": kw_hits,
        "kw_miss": kw_miss,
        "num_recall": num_recall,
        "num_hits": num_hits,
        "num_miss": num_miss,
        "overall_score": overall,
        "passed": passed,
        "has_table_chunk": has_table,
        "avg_rerank_score": avg_rerank,
        "phase1_focus": question.get("phase1_focus"),
        "category": question.get("category"),
        "generated_answer": generated_answer[:1000]
    }

def run_evaluation(pipeline, dataset: List[Dict], save_path=None):
    if not dataset:
        raise ValueError("Dataset is empty - provide a json file with questions")
    results = []
    print(f"Running evaluation on {len(dataset)} questions...")
    
    for q in dataset:
        print(f"\n[Eval] {q['id']}: {q['question'][:60]}...")
        try:
            nodes = pipeline.retrieve(q["question"])
            answer_gen = pipeline.answer(q["question"], stream=False)
            answer = "".join(list(answer_gen))
            result = evaluate_answer(q, answer, nodes)
            results.append(result)
            status = "PASS" if result["passed"] else "FAIL"
            print(f"  -> {status} score={result['overall_score']:.2f} rerank={result['avg_rerank_score']:.3f} table={result['has_table_chunk']}")
            if result["kw_miss"]:
                print(f"     Missed keywords: {result['kw_miss'][:3]}")
        except Exception as e:
            print(f"  -> ERROR: {e}")
            import traceback; traceback.print_exc()
            results.append({
                "question_id": q["id"],
                "error": str(e),
                "passed": False,
                "overall_score": 0.0
            })
    
    total = len(results)
    passed = sum(1 for r in results if r.get("passed"))
    avg_score = sum(r.get("overall_score",0) for r in results)/total if total else 0
    avg_rerank = sum(r.get("avg_rerank_score",0) for r in results)/total if total else 0
    table_hit = sum(1 for r in results if r.get("has_table_chunk"))/total if total else 0
    
    summary = {
        "total": total,
        "passed": passed,
        "failed": total-passed,
        "pass_rate": passed/total if total else 0,
        "avg_score": avg_score,
        "avg_rerank": avg_rerank,
        "table_chunk_hit_rate": table_hit,
        "by_category": {},
    }
    
    from collections import defaultdict
    cat_groups = defaultdict(list)
    for r in results:
        cat_groups[r.get("category","unknown")].append(r)
    for cat, rs in cat_groups.items():
        summary["by_category"][cat] = {
            "count": len(rs),
            "pass_rate": sum(1 for x in rs if x.get("passed"))/len(rs) if rs else 0,
            "avg_score": sum(x.get("overall_score",0) for x in rs)/len(rs) if rs else 0
        }
    
    print("\n" + "="*60)
    print(f"EVALUATION SUMMARY")
    print(f"Total: {total}, Passed: {passed}, Pass Rate: {summary['pass_rate']:.1%}")
    print(f"Avg Score: {avg_score:.2f}, Avg Rerank: {avg_rerank:.3f}, Table Hit: {table_hit:.1%}")
    for cat, stats in summary["by_category"].items():
        print(f"  {cat}: {stats['pass_rate']:.0%} ({stats['count']} Qs)")
    print("="*60)
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump({"summary": summary, "results": results, "dataset": dataset}, f, indent=2)
        print(f"Saved to {save_path}")
    
    return summary, results