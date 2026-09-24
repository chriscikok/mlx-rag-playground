#!/usr/bin/env python3
"""
CLI example for quick testing without Streamlit
python cli.py "What is refund policy?"
"""
import sys
from core.vector_store import get_index
from core.llm import MLXLLM
from core.pipeline import RAGPipeline
from core.ingestion import load_and_chunk
from pathlib import Path
import config

def main():
    query = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else "What is in my knowledge base?"
    
    print("Loading...")
    llm = MLXLLM()
    
    # Auto-index if no storage
    if not Path(config.STORAGE_DIR).exists():
        docs, nodes = load_and_chunk()
        index = get_index(nodes=nodes)
    else:
        index = get_index()
        # need nodes for BM25
        _, nodes = load_and_chunk()
    
    pipeline = RAGPipeline(
        index=index,
        llm=llm,
        nodes=nodes,
        use_hybrid=True,
        use_crag=True,
        use_reranker=True
    )
    
    print(f"\nQ: {query}\n")
    nodes = pipeline.retrieve(query)
    for n in nodes:
        print(f"- {n.metadata.get('file_name')} | {n.text[:120]}...")
    
    print("\nA: ", end="")
    for token in pipeline.answer(query, stream=True):
        print(token, end="", flush=True)
    print()

if __name__ == "__main__":
    main()
