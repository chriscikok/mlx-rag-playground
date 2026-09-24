"""
Retrievers: Basic, Hybrid (Vector+BM25), Reranker - torch-enabled
Scores: Vector = cosine similarity (0-1), BM25 = BM25 score, Hybrid = RRF fused score
Reranker now returns sigmoid-normalized scores 0-1
"""
from llama_index.core.retrievers import VectorIndexRetriever, BaseRetriever
from llama_index.core.schema import BaseNode, NodeWithScore, QueryBundle
from llama_index.core.postprocessor import SentenceTransformerRerank
from typing import List, Union
import math
import config

def get_vector_retriever(index, top_k=None):
    return VectorIndexRetriever(index=index, similarity_top_k=top_k or config.TOP_K_VECTOR)

def _extract_query_text(query: Union[str, QueryBundle]) -> str:
    if isinstance(query, str):
        return query
    if hasattr(query, "query_str"):
        return query.query_str
    return str(query)

class RankBM25Retriever(BaseRetriever):
    """Pure Python BM25 using rank-bm25"""
    def __init__(self, nodes: List[BaseNode], top_k: int = 10):
        self.nodes = nodes
        self.top_k = top_k
        from rank_bm25 import BM25Okapi
        self.corpus = [n.text.lower().split() for n in nodes]
        self.bm25 = BM25Okapi(self.corpus)
        print(f"[BM25] Indexed {len(nodes)} nodes with rank-bm25")
    
    def _retrieve(self, query_bundle: Union[str, QueryBundle]):
        query_str = _extract_query_text(query_bundle)
        tokenized_q = query_str.lower().split()
        scores = self.bm25.get_scores(tokenized_q)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:self.top_k]
        result = []
        for idx in top_indices:
            result.append(NodeWithScore(node=self.nodes[idx], score=float(scores[idx])))
        return result

class HybridRetriever(BaseRetriever):
    """Hybrid Vector + BM25 with RRF - RRF max = 0.03278 (rank1 in both)"""
    def __init__(self, vector_retriever, bm25_retriever, top_k=10):
        self.vector_retriever = vector_retriever
        self.bm25_retriever = bm25_retriever
        self.top_k = top_k
    
    def _retrieve(self, query_bundle: Union[str, QueryBundle]):
        v_nodes = self.vector_retriever.retrieve(query_bundle)
        b_nodes = self.bm25_retriever.retrieve(query_bundle)
        
        if v_nodes:
            print(f"[Hybrid] Vector top1: {v_nodes[0].score:.4f}")
        if b_nodes:
            print(f"[Hybrid] BM25 top1: {b_nodes[0].score:.4f}")
        
        rrf_k = 60
        fused_scores = {}
        node_map = {}
        vector_scores = {}
        bm25_scores = {}
        
        for rank, nws in enumerate(v_nodes):
            nid = nws.node.node_id
            fused_scores[nid] = fused_scores.get(nid, 0) + 1.0 / (rrf_k + rank + 1)
            node_map[nid] = nws.node
            vector_scores[nid] = nws.score
        
        for rank, nws in enumerate(b_nodes):
            nid = nws.node.node_id
            fused_scores[nid] = fused_scores.get(nid, 0) + 1.0 / (rrf_k + rank + 1)
            if nid not in node_map:
                node_map[nid] = nws.node
            bm25_scores[nid] = nws.score
        
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)[:self.top_k]
        
        result = []
        for nid in sorted_ids:
            base_node = node_map[nid]
            try:
                if base_node.metadata is None:
                    base_node.metadata = {}
                base_node.metadata["_rrf_score"] = fused_scores[nid]
                base_node.metadata["_vector_score"] = vector_scores.get(nid, 0.0)
                base_node.metadata["_bm25_score"] = bm25_scores.get(nid, 0.0)
            except Exception as e:
                print(f"[Hybrid] Metadata set failed: {e}")
                if not hasattr(base_node, 'metadata') or base_node.metadata is None:
                    base_node.metadata = {}
                base_node.metadata["_rrf_score"] = fused_scores[nid]
            
            result.append(NodeWithScore(node=base_node, score=fused_scores[nid]))
        
        return result

def get_bm25_retriever(index=None, top_k=None, nodes=None):
    if nodes is None:
        raise ValueError("get_bm25_retriever requires nodes")
    return RankBM25Retriever(nodes=nodes, top_k=top_k or config.TOP_K_BM25)

def get_hybrid_retriever(index, nodes=None):
    vector = get_vector_retriever(index)
    if nodes is None:
        try:
            nodes = list(index.docstore.docs.values())
        except Exception:
            print("[Hybrid] No nodes for BM25, falling back to Vector only")
            return vector
    if not nodes:
        return vector
    bm25 = RankBM25Retriever(nodes=nodes, top_k=config.TOP_K_BM25)
    return HybridRetriever(vector_retriever=vector, bm25_retriever=bm25, top_k=config.TOP_K_VECTOR)

class SigmoidReranker(SentenceTransformerRerank):
    """Wraps cross-encoder and converts raw logits to 0-1 via sigmoid
    ms-marco-MiniLM outputs logits like -2.7, 5.2 - not 0-1
    Sigmoid: 1 / (1 + exp(-x)) -> maps to 0-1 probability
    """
    def postprocess_nodes(self, nodes, query_str=None, **kwargs):
        # Get raw reranked nodes from parent
        reranked = super().postprocess_nodes(nodes, query_str=query_str, **kwargs)
        
        # Apply sigmoid normalization
        for n in reranked:
            raw_score = n.score
            # Sigmoid normalization
            try:
                norm_score = 1 / (1 + math.exp(-raw_score))
            except OverflowError:
                norm_score = 1.0 if raw_score > 0 else 0.0
            
            # Store both raw and normalized
            if hasattr(n, 'node') and hasattr(n.node, 'metadata'):
                if n.node.metadata is None:
                    n.node.metadata = {}
                n.node.metadata["_reranker_raw"] = raw_score
                n.node.metadata["_reranker_score"] = norm_score
                n.node.metadata["_pre_rerank_score"] = n.node.metadata.get("_pre_rerank_score", 0)
            
            # Replace score with normalized version for UI and sorting
            n.score = norm_score
            print(f"[Reranker] Raw: {raw_score:.4f} -> Sigmoid: {norm_score:.4f} | {n.text[:60]}...")
        
        return reranked

def get_reranker(top_n=None):
    """Returns reranker with sigmoid-normalized 0-1 scores"""
    return SigmoidReranker(
        model="cross-encoder/ms-marco-MiniLM-L-6-v2",
        top_n=top_n or config.TOP_K_RERANK
    )