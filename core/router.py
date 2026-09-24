"""
Adaptive RAG: Route query to right pipeline
"""
from mlx_lm import generate
import config

class QueryRouter:
    def __init__(self, llm):
        self.llm = llm  # MLXLLM instance
    
    def route(self, query: str) -> str:
        prompt = f"""Classify the query into one type:
- FACTUAL: simple fact lookup
- COMPARISON: compare two things
- MULTI_HOP: needs multiple steps or reasoning

Query: {query}
Type (only one word):"""
        out = self.llm.generate(prompt, max_tokens=10)
        out = out.upper()
        if "COMPARISON" in out:
            return "COMPARISON"
        if "MULTI_HOP" in out:
            return "MULTI_HOP"
        return "FACTUAL"
