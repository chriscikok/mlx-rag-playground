"""
Corrective RAG: Relevance Grader using tiny MLX model
Fixed: Less strict, handles Qwen output better
"""
from mlx_lm import load, generate
import config

class RelevanceGrader:
    def __init__(self, model_id=None):
        model_id = model_id or config.GRADER_MODEL_ID
        print(f"Loading grader model: {model_id}")
        self.model, self.tokenizer = load(model_id)
    
    def is_relevant(self, query: str, chunk_text: str) -> bool:
        # Shorter prompt works better for 1.5B model
        prompt = f"""<|im_start|>user
Query: {query}
Chunk: {chunk_text[:800]}
Does this chunk help answer the query? Answer YES or NO.
<|im_end|>
<|im_start|>assistant
"""
        try:
            out = generate(
                self.model, self.tokenizer,
                prompt=prompt,
                max_tokens=5,
                verbose=False
            )
            # Qwen might output "YES" or "Yes" or "YES\n"
            out_upper = out.upper().strip()
            print(f"[Grader] Q:'{query[:30]}' -> {out_upper[:20]}")
            return "YES" in out_upper
        except Exception as e:
            print(f"[Grader] Error: {e}, assuming relevant")
            return True  # On error, keep the node
    
    def filter(self, query, nodes):
        if not nodes:
            return nodes
        good = []
        for n in nodes:
            if self.is_relevant(query, n.text):
                good.append(n)
        print(f"CRAG: {len(nodes)} -> {len(good)} relevant")
        # If all filtered out, return original (don't break RAG)
        if len(good) == 0 and len(nodes) > 0:
            print("[CRAG] All filtered, returning original to avoid empty KB")
            return nodes
        return good