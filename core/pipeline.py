"""
Pipeline - Phase 1: General purpose, no hardcoded channel logic
"""
from .retrievers import get_hybrid_retriever, get_vector_retriever, get_reranker
from .grader import RelevanceGrader
from .router import QueryRouter

class RAGPipeline:
    def __init__(self, index, llm, nodes=None, use_hybrid=True, use_crag=True, use_reranker=True, use_adaptive=False):
        self.index = index
        self.llm = llm
        self.nodes = nodes
        if use_hybrid:
            self.retriever = get_hybrid_retriever(index, nodes=nodes)
        else:
            self.retriever = get_vector_retriever(index)
        self.reranker = get_reranker() if use_reranker else None
        self.grader = RelevanceGrader() if use_crag else None
        self.router = QueryRouter(llm) if use_adaptive else None
        self.use_crag = use_crag
        self.use_adaptive = use_adaptive
        self.use_reranker = use_reranker
    
    def retrieve(self, query):
        if self.use_adaptive and self.router:
            route_type = self.router.route(query)
            print(f"[Adaptive] Route: {route_type}")
        try:
            nodes = self.retriever.retrieve(query)
        except Exception as e:
            print(f"[Retriever] Error: {e}")
            import traceback; traceback.print_exc()
            nodes = self.index.as_retriever(similarity_top_k=10).retrieve(query)
        print(f"[Retriever] Got {len(nodes)} candidates for '{query[:50]}'")
        if not nodes:
            return []
        if self.use_crag and self.grader:
            try:
                filtered = self.grader.filter(query, nodes)
                if len(filtered) > 0:
                    nodes = filtered
            except Exception as e:
                print(f"[CRAG] Error, skipping: {e}")
        if self.reranker and nodes:
            try:
                for n in nodes:
                    if hasattr(n, 'node') and hasattr(n.node, 'metadata'):
                        n.node.metadata = n.node.metadata or {}
                        n.node.metadata["_pre_rerank_score"] = n.score
                reranked = self.reranker.postprocess_nodes(nodes, query_str=query)
                print(f"[Reranker] Reranked {len(nodes)} -> {len(reranked)}")
                for n in reranked:
                    if hasattr(n, 'node') and hasattr(n.node, 'metadata'):
                        n.node.metadata = n.node.metadata or {}
                        n.node.metadata["_reranker_score"] = n.score
                nodes = reranked
            except Exception as e:
                print(f"[Reranker] Error, skipping: {e}")
                import traceback; traceback.print_exc()
        return nodes
    
    def answer(self, query, stream=False):
        nodes = self.retrieve(query)
        if not nodes:
            yield "I don't have that in the knowledge base."
            return
        prompt = self.llm.build_rag_prompt(query, nodes)
        if stream:
            yield from self.llm.stream(prompt)
        else:
            ans = self.llm.generate(prompt)
            yield ans
    
    def agentic_answer(self, query, max_iters=2):
        context_nodes = []
        context_text = ""
        for i in range(max_iters):
            check_prompt = f"Context: {context_text}\nQuestion: {query}\nDo you have enough to answer? If NO, write SEARCH: <new query>. If YES, write READY."
            check = self.llm.generate(check_prompt, max_tokens=100)
            if "SEARCH:" in check:
                sub_q = check.split("SEARCH:")[1].strip().split("\n")[0]
                print(f"[Agentic] Searching: {sub_q}")
                new_nodes = self.retriever.retrieve(sub_q)
                context_nodes.extend(new_nodes)
                context_text += "\n".join([n.text for n in new_nodes[:2]])
            else:
                break
        if not context_nodes:
            context_nodes = self.retrieve(query)
        prompt = self.llm.build_rag_prompt(query, context_nodes)
        return self.llm.generate(prompt)