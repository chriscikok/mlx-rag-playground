"""
MLX LLM wrapper - reusable across all RAG types
Supports streaming for UI
"""
from mlx_lm import load, generate
from mlx_lm.generate import stream_generate
import config

class MLXLLM:
    def __init__(self, model_id=None):
        model_id = model_id or config.LLM_MODEL_ID
        print(f"Loading LLM: {model_id}...")
        self.model, self.tokenizer = load(model_id)
    
    def generate(self, prompt: str, max_tokens=None, temp=None, verbose=False):
        return generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens or config.MAX_TOKENS,
            temp=temp or config.TEMPERATURE,
            verbose=verbose
        )
    
    def stream(self, prompt: str, max_tokens=None):
        # Yields tokens for Streamlit
        for token in stream_generate(
            self.model, self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens or config.MAX_TOKENS
        ):
            yield token.text
    
    def build_rag_prompt(self, query, context_nodes):
        # FIX: Sort by score descending so most relevant appears first
        # LLM tends to focus on first chunks in context
        sorted_nodes = sorted(context_nodes, key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)

        context_parts = []
        for i, n in enumerate(sorted_nodes):
            # Include score in context for debugging but LLM should ignore
            fname = n.metadata.get('file_name', 'doc')
            context_parts.append(f"[Chunk {i+1} from {fname} - relevance {n.score:.4f}]:\n{n.text}")

        context_str = "\n\n---\n\n".join(context_parts)

        return f"""<|im_start|>system
You are a precise local knowledge base assistant. You excel at extracting numbers and tables. 

THINKING PROCESS (do this internally, then answer):
1. Identify what the user asks - note if it specifies a channel/method (online, branch, mobile, etc.)
2. Scan Context for relevant information - pay attention to markdown tables with [TABLE_START] markers
3. For numbers: preserve exact values, currencies, percentages, date ranges
4. If multiple chunks contain parts, combine them logically
5. If table is split across chunks, reconstruct it

RULES:
- Answer ONLY using Context below. Be precise with numbers and preserve exact values.
- For tables: reproduce relevant rows as markdown table. 
- When Context contains a table with multiple rows, extract ALL matching rows, not just first one
- Cite sources like [Chunk 1] for each fact
- If Context has conflicting info, prioritize higher relevance (lower chunk number)
- If answer requires calculation (e.g., total interest period), show your reasoning briefly
- If information not in Context, say "I don't have that in the knowledge base - retrieved chunks don't contain [specific field]"
- Be concise but include all relevant numbers: rates, dates, min/max, currencies
<|im_end|>
<|im_start|>user
Context:
{context_str}

Question: {query}

Analyze step by step, then give final answer with exact numbers and citations.
<|im_end|>
<|im_start|>assistant
"""