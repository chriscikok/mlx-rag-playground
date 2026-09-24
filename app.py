
"""
Streamlit App - General RAG with proper score display
Fixed: No slider, keyword highlighting
"""
import streamlit as st
from pathlib import Path
import config
from core.ingestion import load_and_chunk
from core.vector_store import get_index
from core.llm import MLXLLM
from core.pipeline import RAGPipeline
import re
import html

def highlight_keywords(text: str, query: str) -> str:
    """Highlight query keywords in text"""
    if not query or not text:
        return html.escape(text)
    
    highlighted = text
    keywords = re.findall(r'\w+', query.lower())
    stopwords = {'what','is','the','a','an','are','of','for','in','on','how','when','where','who','does','do','was','were','be','to','and','or','it','this','that'}
    keywords = [k for k in keywords if k not in stopwords and len(k) >= 2]
    keywords = sorted(set(keywords), key=len, reverse=True)
    
    for kw in keywords:
        regex = re.compile(f'({re.escape(kw)})', re.IGNORECASE)
        highlighted = regex.sub(r'__MARK_START__\1__MARK_END__', highlighted)
    
    highlighted_escaped = html.escape(highlighted)
    highlighted_escaped = highlighted_escaped.replace('__MARK_START__', "<mark style='background-color: #ffeb3b; color: #000; padding: 1px 4px; border-radius: 3px; font-weight: 600'>")
    highlighted_escaped = highlighted_escaped.replace('__MARK_END__', "</mark>")
    return highlighted_escaped

st.set_page_config(page_title="MLX Local RAG", layout="wide")
st.title("🔍 MLX Local RAG - Modular")

with st.sidebar:
    st.header("RAG Settings")
    use_hybrid = st.checkbox("Hybrid (Vector + BM25)", value=True, help="RRF max 0.0328 = rank1 in both")
    use_crag = st.checkbox("Corrective (CRAG Grader)", value=False)
    use_reranker = st.checkbox("Reranker (cross-encoder)", value=True, help="Sigmoid normalized 0-1, 0.5+ relevant")
    use_adaptive = st.checkbox("Adaptive Router", value=False)
    mode = st.selectbox("Mode", ["Standard", "Agentic (Self-RAG)"])
    
    st.divider()
    current_settings = (use_hybrid, use_crag, use_reranker, use_adaptive)
    if "last_settings" not in st.session_state:
        st.session_state.last_settings = current_settings
    
    if st.session_state.last_settings != current_settings:
        st.warning("⚠️ Settings changed! Click 'Load Models' to apply.")
    
    if st.button("🗑️ Clear DB"):
        # IMPORTANT: Clear cache FIRST to release Chroma client lock
        st.cache_resource.clear()
        st.session_state.pipeline = None
        import time
        time.sleep(0.5)  # Let file handles release
        try:
            from core.vector_store import force_delete_chroma
            force_delete_chroma()
        except ImportError:
            from core.vector_store import _force_delete_chroma
            _force_delete_chroma()
        st.session_state.last_settings = current_settings
        st.success("Cleared! Now Re-index")
    
    if st.button("Re-index data/ folder"):
        with st.spinner("Loading and chunking..."):
            try:
                # Clear old pipeline cache first to release lock
                if "pipeline" in st.session_state and st.session_state.pipeline is not None:
                    st.cache_resource.clear()
                    st.session_state.pipeline = None
                    import time
                    time.sleep(0.5)
                docs, nodes = load_and_chunk()
                get_index(nodes=nodes)
                st.success(f"Indexed {len(nodes)} chunks - avg {sum(len(n.text) for n in nodes)//len(nodes) if nodes else 0} chars per chunk")
            except Exception as e:
                if "readonly" in str(e).lower() or "1032" in str(e):
                    st.error(f"Readonly DB error: {e}")
                    st.warning("Force clearing DB and retrying...")
                    st.cache_resource.clear()
                    st.session_state.pipeline = None
                    import time
                    time.sleep(0.5)
                    try:
                        from core.vector_store import force_delete_chroma, fix_permissions
                        from pathlib import Path as P
                        fix_permissions(P(config.CHROMA_DIR))
                        fix_permissions(P(config.STORAGE_DIR))
                        force_delete_chroma()
                    except ImportError:
                        from core.vector_store import _force_delete_chroma, _fix_permissions
                        from pathlib import Path as P
                        _fix_permissions(P(config.CHROMA_DIR))
                        _fix_permissions(P(config.STORAGE_DIR))
                        _force_delete_chroma()
                    time.sleep(0.5)
                    try:
                        docs, nodes = load_and_chunk()
                        get_index(nodes=nodes)
                        st.success(f"Fixed! Indexed {len(nodes)} chunks")
                    except Exception as e2:
                        st.error(f"Still failing after force delete: {e2}")
                        st.markdown("**Manual fix:** Stop Streamlit (Ctrl+C) and run:\n```\nrm -rf chroma_db storage\nchmod -R 777 .\n```\nThen restart.")
                else:
                    st.error(f"Re-index failed: {e}")
                    import traceback; st.code(traceback.format_exc())
    
    st.divider()
    st.caption(f"LLM: {config.LLM_MODEL_ID}\nEmbed: {config.EMBED_MODEL_ID}")
    st.caption(f"Chunk: {config.CHUNK_SIZE} chars, Overlap: {config.CHUNK_OVERLAP}")

@st.cache_resource(show_spinner=False)
def load_stack(use_hybrid, use_crag, use_reranker, use_adaptive):
    print(f"[Load] hybrid={use_hybrid}, reranker={use_reranker}")
    llm = MLXLLM()
    docs, nodes = load_and_chunk()
    print(f"Loaded {len(nodes) if nodes else 0} nodes")
    if nodes:
        index = get_index(nodes=nodes)
    else:
        index = get_index()
        nodes = None
    pipeline = RAGPipeline(index=index, llm=llm, nodes=nodes, use_hybrid=use_hybrid, use_crag=use_crag, use_reranker=use_reranker, use_adaptive=use_adaptive)
    return pipeline

if "pipeline" not in st.session_state:
    st.session_state.pipeline = None

if st.sidebar.button("Load Models", type="primary"):
    with st.spinner("Loading models... ~30s"):
        if st.session_state.last_settings != (use_hybrid, use_crag, use_reranker, use_adaptive):
            st.cache_resource.clear()
        st.session_state.pipeline = load_stack(use_hybrid, use_crag, use_reranker, use_adaptive)
        st.session_state.last_settings = (use_hybrid, use_crag, use_reranker, use_adaptive)
        st.success(f"Ready! Reranker={'ON' if st.session_state.pipeline.reranker else 'OFF'}")

if st.session_state.pipeline is None:
    st.markdown("""
    ### 👋 Welcome - Click Load Models in sidebar to start
    **Tip:** For better scores on T&C, try smaller chunks (400) and specific queries
    """)
    st.stop()

if prompt := st.chat_input("Ask your knowledge base..."):
    if not st.session_state.pipeline:
        st.session_state.pipeline = load_stack(use_hybrid, use_crag, use_reranker, use_adaptive)
    
    st.chat_message("user").write(prompt)
    
    with st.chat_message("assistant"):
        pipe = st.session_state.pipeline
        st.caption(f"Pipeline: {pipe.retriever.__class__.__name__}, Reranker={'ON' if pipe.reranker else 'OFF'} | ✅")
        
        nodes = pipe.retrieve(prompt)
        
        if not nodes:
            st.write("I don't have that in the knowledge base.")
        else:
            with st.expander(f"📚 Sources ({len(nodes)} chunks) - Sorted by relevance", expanded=True):
                for i, n in enumerate(nodes):
                    meta = n.metadata if hasattr(n, 'metadata') and n.metadata else {}
                    rrf = meta.get('_rrf_score', 0)
                    vec = meta.get('_vector_score', 0)
                    raw = meta.get('_reranker_raw', None)
                    rerank = meta.get('_reranker_score', n.score if pipe.reranker else None)
                    
                    if pipe.reranker and rerank is not None:
                        if rerank > 0.7:
                            badge = "🟢 High"
                        elif rerank > 0.3:
                            badge = "🟡 Medium"
                        else:
                            badge = "🔵 Low"
                        st.markdown(f"**{i+1}. {n.metadata.get('file_name','unknown')}** {badge}")
                        if raw is not None:
                            st.markdown(f"`Reranker: {rerank:.3f} (raw {raw:.2f}) | Vector: {vec:.3f}`")
                        else:
                            st.markdown(f"`Reranker: {rerank:.3f} | Vector: {vec:.3f}`")
                    else:
                        badge = "🟢 Excellent" if rrf > 0.03 else "🟡 Good" if rrf > 0.02 else "🔵 OK"
                        st.markdown(f"**{i+1}. {n.metadata.get('file_name','unknown')}** {badge}")
                        st.markdown(f"`RRF: {rrf:.4f} | Vector: {vec:.3f} | Final: {n.score:.4f}`")
                    
                    preview_text = n.text[:600] + ("..." if len(n.text) > 600 else "")
                    highlighted_html = highlight_keywords(preview_text, prompt)
                    st.markdown(f"<div style='background: rgba(255,255,255,0.05); padding: 8px 12px; border-radius: 6px; border-left: 3px solid #4CAF50; font-size: 0.9em; line-height: 1.5'>{highlighted_html}</div>", unsafe_allow_html=True)
                    st.divider()
            
            placeholder = st.empty()
            full = ""
            for token in st.session_state.pipeline.answer(prompt, stream=True):
                full += token
                placeholder.write(full)