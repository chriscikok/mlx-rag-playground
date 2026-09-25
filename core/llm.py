"""
MLX LLM wrapper - General purpose, table-aware but not document-specific
Phase 1: No hardcoded channel/branch logic
Fixed for mlx-lm 0.18+ API changes (temp -> sampler)
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
        max_toks = max_tokens or config.MAX_TOKENS
        temperature = temp or config.TEMPERATURE
        
        # mlx-lm API changed over versions: older uses temp=, newer uses temperature= or sampler
        # Try in order to be compatible
        try:
            return generate(
                self.model, self.tokenizer,
                prompt=prompt,
                max_tokens=max_toks,
                temp=temperature,
                verbose=verbose
            )
        except TypeError as e:
            if "temp" in str(e) or "unexpected keyword" in str(e):
                try:
                    # Newer API: temperature
                    return generate(
                        self.model, self.tokenizer,
                        prompt=prompt,
                        max_tokens=max_toks,
                        temperature=temperature,
                        verbose=verbose
                    )
                except TypeError:
                    # Even newer API: sampler based or no temp arg
                    try:
                        from mlx_lm.sample_utils import make_sampler
                        sampler = make_sampler(temp=temperature)
                        return generate(
                            self.model, self.tokenizer,
                            prompt=prompt,
                            max_tokens=max_toks,
                            sampler=sampler,
                            verbose=verbose
                        )
                    except Exception:
                        # Fallback: no temp at all
                        return generate(
                            self.model, self.tokenizer,
                            prompt=prompt,
                            max_tokens=max_toks,
                            verbose=verbose
                        )
            else:
                raise
    
    def stream(self, prompt: str, max_tokens=None):
        max_toks = max_tokens or config.MAX_TOKENS
        # Try with temp fallback as well
        try:
            for token in stream_generate(
                self.model, self.tokenizer,
                prompt=prompt,
                max_tokens=max_toks,
                temp=config.TEMPERATURE
            ):
                yield token.text
        except TypeError:
            try:
                for token in stream_generate(
                    self.model, self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_toks,
                    temperature=config.TEMPERATURE
                ):
                    yield token.text
            except TypeError:
                # Fallback no temp
                for token in stream_generate(
                    self.model, self.tokenizer,
                    prompt=prompt,
                    max_tokens=max_toks
                ):
                    yield token.text
    
    def build_rag_prompt(self, query, context_nodes):
        sorted_nodes = sorted(context_nodes, key=lambda x: x.score if hasattr(x, 'score') else 0, reverse=True)
        
        context_parts = []
        for i, n in enumerate(sorted_nodes):
            fname = n.metadata.get('file_name', 'doc')
            context_parts.append(f"[Chunk {i+1} from {fname} - relevance {n.score:.4f}]:\n{n.text}")
        
        context_str = "\n\n---\n\n".join(context_parts)
        
        return f"""<|im_start|>system
You are a precise knowledge base assistant. You excel at extracting structured information from documents.

THINKING PROCESS:
1. Identify what the user asks
2. Scan Context for relevant information, including tables marked [TABLE_START]
3. Preserve exact values: numbers, currencies, percentages, dates
4. If multiple chunks contain parts, combine them logically
5. If table is split, reconstruct it

RULES:
- Answer ONLY using Context below
- Be precise with numbers and preserve exact values
- For tables: reproduce relevant rows as markdown table
- If question specifies a filter, extract matching row(s)
- If Context contains multiple relevant rows, include all
- Cite sources like [Chunk 1] for each fact
- If info not in Context, say "I don't have that in the knowledge base"
- Include all relevant numbers: rates, dates, min/max, currencies
<|im_end|>
<|im_start|>user
Context:
{context_str}

Question: {query}

Analyze step by step, then give final answer with exact numbers and citations.
<|im_end|>
<|im_start|>assistant
"""