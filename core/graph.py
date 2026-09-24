"""
Graph RAG module - optional
Builds entity graph from docs
"""
import networkx as nx

class GraphRAG:
    def __init__(self, llm):
        self.llm = llm
        self.graph = nx.Graph()
    
    def build_from_nodes(self, nodes):
        # Extract triples using LLM
        # Prompt: Extract (subject, relation, object) from text
        print("Building knowledge graph from nodes...")
        for node in nodes:
            prompt = f"Extract up to 5 triples as (subject, relation, object) from:\n{node.text[:800]}\nTriples:"
            out = self.llm.generate(prompt, max_tokens=300)
            # Simple parse - in production use structured output
            for line in out.split("\n"):
                if "," in line or "->" in line:
                    # naive parse, improve with regex
                    parts = line.replace("(", "").replace(")", "").split(",")
                    if len(parts) >= 3:
                        s, r, o = parts[0].strip(), parts[1].strip(), parts[2].strip()
                        self.graph.add_edge(s, o, relation=r, source=node.metadata.get("file_name"))
        print(f"Graph built: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
    
    def retrieve(self, query, k=5):
        # Extract entities from query, traverse 2 hops
        # For demo, simple keyword match on graph nodes
        related = []
        for node in self.graph.nodes:
            if node.lower() in query.lower():
                # get neighbors
                neighbors = list(self.graph.neighbors(node))[:k]
                related.extend(neighbors)
        return related

# For LlamaIndex KG, you can also use:
# from llama_index.core import KnowledgeGraphIndex
# kg_index = KnowledgeGraphIndex.from_documents(...)
