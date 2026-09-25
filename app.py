"""
Streamlit App - Sustainable Knowledge Management RAG
Tabs: Chat | Knowledge Manager | Evaluate | Test Creator
Fixed: Incremental ingestion, manifest tracking, upload, evaluation
"""
import streamlit as st
from pathlib import Path
import config
from core.ingestion import load_and_chunk
from core.vector_store import get_index, force_delete_chroma, delete_by_file_name, add_nodes_to_index
from core.llm import MLXLLM
from core.pipeline import RAGPipeline
from core.knowledge_manager import scan_data_dir, get_indexed_files_summary, update_manifest_entry, remove_manifest_entry, load_manifest, save_manifest, get_file_hash, get_files_to_process
from core.evaluation import load_test_set, run_evaluation
from core.embeddings import get_embed_model
import re
import html
import json
import time
import shutil
from collections import Counter

st.set_page_config(page_title="MLX Local RAG", layout="wide", initial_sidebar_state="expanded")
st.title("🔍 MLX Local RAG - Sustainable")

def highlight_keywords(text: str, query: str) -> str:
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

# --- Sidebar ---
with st.sidebar:
    st.header("RAG Settings")
    use_hybrid = st.checkbox("Hybrid (Vector + BM25)", value=True, help="RRF max 0.0328 = rank1 in both")
    use_crag = st.checkbox("Corrective (CRAG Grader)", value=False)
    use_reranker = st.checkbox("Reranker (cross-encoder)", value=True, help="Sigmoid 0-1, 0.5+ relevant")
    use_adaptive = st.checkbox("Adaptive Router", value=False)
    mode = st.selectbox("Mode", ["Standard", "Agentic (Self-RAG)"])
    
    st.divider()
    st.subheader("📁 Data Folder")
    data_path = Path(config.DATA_DIR)
    data_path.mkdir(exist_ok=True)
    files_in_data = list(data_path.rglob("*"))
    files_in_data = [f for f in files_in_data if f.is_file() and not f.name.startswith(".")]
    st.caption(f"{len(files_in_data)} files in data/")
    
    # Quick actions
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear DB", use_container_width=True):
            st.cache_resource.clear()
            if "pipeline" in st.session_state:
                st.session_state.pipeline = None
            time.sleep(0.5)
            force_delete_chroma()
            # Clear manifest
            manifest_path = Path(config.STORAGE_DIR) / "manifest.json"
            if manifest_path.exists():
                manifest_path.unlink()
            st.success("DB + manifest cleared!")
            st.rerun()
    with col2:
        if st.button("🔄 Rescan", use_container_width=True):
            st.rerun()
    
    st.divider()
    st.caption(f"LLM: {config.LLM_MODEL_ID}\nEmbed: {config.EMBED_MODEL_ID}")
    st.caption(f"Chunk: {config.CHUNK_SIZE}, Overlap: {config.CHUNK_OVERLAP}, Mode: {config.CHUNKING_MODE}")

@st.cache_resource(show_spinner=False)
def load_stack(use_hybrid, use_crag, use_reranker, use_adaptive, _manifest_mtime=0):
    print(f"[Load] hybrid={use_hybrid}, reranker={use_reranker}, manifest_mtime={_manifest_mtime}")
    llm = MLXLLM()
    embed_model = get_embed_model()
    # Load all docs for BM25 retriever nodes (even if index already exists)
    docs, nodes = load_and_chunk(embed_model=embed_model)
    print(f"Loaded {len(nodes) if nodes else 0} nodes for retriever")
    if nodes:
        # If index exists, get_index will use it; if not, it will index
        # But we want to ensure index exists
        index = get_index(nodes=nodes, embed_model=embed_model)
    else:
        index = get_index(embed_model=embed_model)
        nodes = None
    pipeline = RAGPipeline(index=index, llm=llm, nodes=nodes, use_hybrid=use_hybrid, use_crag=use_crag, use_reranker=use_reranker, use_adaptive=use_adaptive)
    return pipeline

def get_manifest_mtime():
    mp = Path(config.STORAGE_DIR) / "manifest.json"
    return mp.stat().st_mtime if mp.exists() else 0

# Initialize pipeline state
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "last_settings" not in st.session_state:
    st.session_state.last_settings = None

# --- Tabs ---
tab_chat, tab_knowledge, tab_evaluate, tab_test_creator = st.tabs(["💬 Chat", "📂 Knowledge Manager", "📊 Evaluate", "✏️ Test Creator"])

# ========== TAB 1: CHAT ==========
with tab_chat:
    col_load, col_status = st.columns([1,3])
    with col_load:
        if st.button("🚀 Load Models", type="primary", use_container_width=True):
            with st.spinner("Loading models... ~30s first time"):
                current = (use_hybrid, use_crag, use_reranker, use_adaptive)
                if st.session_state.last_settings != current:
                    st.cache_resource.clear()
                try:
                    st.session_state.pipeline = load_stack(use_hybrid, use_crag, use_reranker, use_adaptive, _manifest_mtime=get_manifest_mtime())
                    st.session_state.last_settings = current
                    st.success(f"Ready! Reranker={'ON' if st.session_state.pipeline.reranker else 'OFF'}")
                except Exception as e:
                    st.error(f"Load failed: {e}")
                    import traceback; st.code(traceback.format_exc())
    with col_status:
        if st.session_state.pipeline:
            st.success(f"✅ Models loaded | Retriever: {st.session_state.pipeline.retriever.__class__.__name__} | Reranker: {'ON' if st.session_state.pipeline.reranker else 'OFF'}")
            manifest = load_manifest()
            st.caption(f"📦 Knowledge: {len(manifest)} files indexed | {sum(v.get('chunks',0) for v in manifest.values())} chunks")
        else:
            st.warning("👈 Click Load Models in sidebar or above to start")

    st.divider()

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display history
    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

    if prompt := st.chat_input("Ask your knowledge base...", key="chat_input"):
        if not st.session_state.pipeline:
            # Auto load if not loaded
            try:
                st.session_state.pipeline = load_stack(use_hybrid, use_crag, use_reranker, use_adaptive, _manifest_mtime=get_manifest_mtime())
            except Exception as e:
                st.error(f"Failed to auto-load models: {e}")
                st.stop()
        
        st.session_state.chat_history.append(("user", prompt))
        st.chat_message("user").write(prompt)
        
        with st.chat_message("assistant"):
            pipe = st.session_state.pipeline
            st.caption(f"Pipeline: {pipe.retriever.__class__.__name__}, Reranker={'ON' if pipe.reranker else 'OFF'}")
            
            nodes = pipe.retrieve(prompt)
            
            if not nodes:
                response = "I don't have that in the knowledge base."
                st.write(response)
                st.session_state.chat_history.append(("assistant", response))
            else:
                with st.expander(f"📚 Sources ({len(nodes)} chunks) - Sorted by relevance", expanded=True):
                    for i, n in enumerate(nodes):
                        meta = n.metadata if hasattr(n, 'metadata') and n.metadata else {}
                        rrf = meta.get('_rrf_score', 0)
                        vec = meta.get('_vector_score', 0)
                        raw = meta.get('_reranker_raw', None)
                        rerank = meta.get('_reranker_score', n.score if pipe.reranker else None)
                        fname = meta.get('file_name','unknown')
                        is_table = "📊" if meta.get('is_table') else "📄"
                        
                        if pipe.reranker and rerank is not None:
                            badge = "🟢 High" if rerank > 0.7 else "🟡 Medium" if rerank > 0.3 else "🔵 Low"
                            st.markdown(f"**{i+1}. {is_table} {fname}** {badge}")
                            if raw is not None:
                                st.markdown(f"`Reranker: {rerank:.3f} (raw {raw:.2f}) | Vector: {vec:.3f}`")
                            else:
                                st.markdown(f"`Reranker: {rerank:.3f} | Vector: {vec:.3f}`")
                        else:
                            badge = "🟢 Excellent" if rrf > 0.03 else "🟡 Good" if rrf > 0.02 else "🔵 OK"
                            st.markdown(f"**{i+1}. {is_table} {fname}** {badge}")
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
                st.session_state.chat_history.append(("assistant", full))

# ========== TAB 2: KNOWLEDGE MANAGER ==========
with tab_knowledge:
    st.header("📂 Sustainable Knowledge Manager")
    st.markdown("Upload new docs, track changes via hash manifest, incremental indexing without full reindex.")
    
    # Upload section
    with st.expander("📤 Upload New Knowledge", expanded=True):
        uploaded_files = st.file_uploader("Drop PDFs, DOCX, MD, TXT, HTML", accept_multiple_files=True, type=["pdf","docx","md","txt","html"])
        if uploaded_files:
            for uf in uploaded_files:
                dest = Path(config.DATA_DIR) / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                st.success(f"Saved {uf.name} ({uf.size/1024/1024:.2f} MB) to data/")
            if st.button("🔄 Scan after upload"):
                st.rerun()

    # Scan current status
    scanned = scan_data_dir()
    summary = get_indexed_files_summary()
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total files in data/", len(scanned))
    col2.metric("Indexed (manifest)", summary["total_files"])
    col3.metric("Total chunks", summary["total_chunks"])
    col4.metric("Total table chunks", summary["total_tables"])

    if not scanned:
        st.info("No files in data/ folder. Upload some PDFs to start.")
    else:
        # Show table
        import pandas as pd
        df_data = []
        for f in scanned:
            df_data.append({
                "File": f["name"],
                "Status": f["status"],
                "Size MB": f["size_mb"],
                "Chunks": f["chunks"],
                "Tables": f["tables"],
                "Modified": f["mtime_str"],
                "Indexed At": f["indexed_at"] or "-",
            })
        df = pd.DataFrame(df_data)
        
        # Color status
        def color_status(val):
            if val == "new": return "background-color: #ffeb3b; color: #000"
            elif val == "modified": return "background-color: #ff9800; color: #fff"
            elif val == "indexed": return "background-color: #4caf50; color: #fff"
            return ""
        
        st.dataframe(df.style.applymap(color_status, subset=["Status"]), use_container_width=True, height=300)
        
        # Actions
        st.subheader("🔧 Indexing Actions")
        c1, c2, c3 = st.columns(3)
        with c1:
            to_process, _ = get_files_to_process(mode="incremental")
            st.metric("New/Modified to index", len(to_process))
            if st.button("⚡ Incremental Index (only new/changed)", type="primary", use_container_width=True, disabled=len(to_process)==0):
                with st.spinner(f"Indexing {len(to_process)} files..."):
                    try:
                        # Clear cache to release lock
                        st.cache_resource.clear()
                        if "pipeline" in st.session_state:
                            st.session_state.pipeline = None
                        # Delete old chunks for modified
                        for p in to_process:
                            # Check if modified
                            existing = [s for s in scanned if s["full_path"]==str(p) and s["status"]=="modified"]
                            if existing:
                                deleted = delete_by_file_name(p.name)
                                st.caption(f"Deleted {deleted} old chunks for {p.name}")
                        embed_model = get_embed_model()
                        docs, nodes = load_and_chunk(specific_files=to_process, embed_model=embed_model)
                        st.write(f"Extracted {len(nodes)} chunks from {len(docs)} docs")
                        if nodes:
                            # Add incrementally
                            file_chunk_counts = Counter([n.metadata.get("file_name","") for n in nodes])
                            file_table_counts = Counter([n.metadata.get("file_name","") for n in nodes if n.metadata.get("is_table")])
                            add_nodes_to_index(nodes, embed_model=embed_model)
                            for p in to_process:
                                update_manifest_entry(p, file_chunk_counts.get(p.name,0), file_table_counts.get(p.name,0))
                            st.success(f"✅ Incrementally indexed {len(nodes)} chunks from {len(to_process)} files!")
                        else:
                            st.warning("No chunks extracted")
                        # Always refresh status display
                        st.cache_resource.clear()
                        time.sleep(0.3)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Incremental index failed: {e}")
                        import traceback; st.code(traceback.format_exc())
        with c2:
            if st.button("🔄 Full Reindex (all files)", use_container_width=True):
                with st.spinner("Full reindex - clearing and rebuilding..."):
                    try:
                        st.cache_resource.clear()
                        if "pipeline" in st.session_state:
                            st.session_state.pipeline = None
                        time.sleep(0.5)
                        force_delete_chroma()
                        manifest_path = Path(config.STORAGE_DIR) / "manifest.json"
                        if manifest_path.exists():
                            manifest_path.unlink()
                        embed_model = get_embed_model()
                        docs, nodes = load_and_chunk(embed_model=embed_model)
                        get_index(nodes=nodes, embed_model=embed_model)
                        # Build manifest
                        file_chunk_counts = Counter([n.metadata.get("file_name","") for n in nodes])
                        file_table_counts = Counter([n.metadata.get("file_name","") for n in nodes if n.metadata.get("is_table")])
                        manifest = {}
                        for fp in Path(config.DATA_DIR).rglob("*"):
                            if fp.suffix.lower() in config.SUPPORTED_EXTS and fp.is_file():
                                rel = str(fp.relative_to(Path(config.DATA_DIR)))
                                manifest[rel] = {
                                    "hash": get_file_hash(fp),
                                    "full_path": str(fp),
                                    "chunks": file_chunk_counts.get(fp.name,0),
                                    "tables": file_table_counts.get(fp.name,0),
                                    "indexed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                                    "size": fp.stat().st_size
                                }
                        save_manifest(manifest)
                        st.cache_resource.clear()
                        st.cache_data.clear() if hasattr(st, 'cache_data') else None
                        st.success(f"Full reindex {len(nodes)} chunks. Cache rebuilt.")
                        time.sleep(0.5)
                        st.rerun()
                    except Exception as e:
                        st.error(f"Full reindex failed: {e}")
                        import traceback; st.code(traceback.format_exc())
        with c3:
            st.markdown("**Delete file from KB:**")
            file_to_delete = st.selectbox("Select file to remove", options=[f["name"] for f in scanned], key="delete_select")
            if st.button(f"🗑️ Delete {file_to_delete} from index", use_container_width=True):
                try:
                    deleted = delete_by_file_name(file_to_delete)
                    removed = remove_manifest_entry(file_to_delete)
                    # Also delete file from data/ ?
                    # Keep file but remove from index
                    st.success(f"Deleted {deleted} chunks for {file_to_delete}, removed {removed} manifest entries")
                    st.cache_resource.clear()
                    if "pipeline" in st.session_state:
                        st.session_state.pipeline = None
                except Exception as e:
                    st.error(f"Delete failed: {e}")

# ========== TAB 3: EVALUATE ==========
with tab_evaluate:
    st.header("📊 Evaluate RAG Performance")
    st.markdown("Run evaluation against test sets in `test/` folder (gitignored). Uses current index - no need to clear unless you want full reindex.")
    
    test_dir = Path(config.TEST_DIR)
    test_dir.mkdir(exist_ok=True)
    test_files = list(test_dir.glob("*.json"))
    # Filter out template? Keep but show
    test_files = sorted(test_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not test_files:
        st.warning("No test files in test/ folder. Create one in Test Creator tab.")
    else:
        selected_test = st.selectbox("Select test set", options=[str(f) for f in test_files], format_func=lambda x: Path(x).name)
        
        if selected_test:
            try:
                with open(selected_test, 'r') as f:
                    data = json.load(f)
                    if isinstance(data, dict) and 'dataset' in data:
                        data = data['dataset']
                st.caption(f"{len(data)} questions in {Path(selected_test).name}")
                with st.expander(f"Preview {Path(selected_test).name}"):
                    st.json(data[:2])  # Show first 2
            except Exception as e:
                st.error(f"Failed to load {selected_test}: {e}")
        
        col_eval1, col_eval2 = st.columns(2)
        with col_eval1:
            use_reranker_eval = st.checkbox("Use Reranker", value=True, key="eval_reranker")
        with col_eval2:
            output_name = st.text_input("Output results file", value=f"eval_results_{Path(selected_test).stem if selected_test else 'test'}.json" if selected_test else "eval_results.json")
        
        if st.button("🚀 Run Evaluation", type="primary", use_container_width=True, disabled=not selected_test):
            if not st.session_state.pipeline:
                st.warning("Loading models first...")
                try:
                    st.session_state.pipeline = load_stack(use_hybrid, use_crag, use_reranker, use_adaptive, _manifest_mtime=get_manifest_mtime())
                except Exception as e:
                    st.error(f"Failed to load models: {e}")
                    st.stop()
            
            with st.spinner(f"Evaluating {Path(selected_test).name}..."):
                try:
                    dataset = load_test_set(selected_test)
                    # Use current pipeline
                    pipe = st.session_state.pipeline
                    summary, results = run_evaluation(pipe, dataset=dataset, save_path=output_name)
                    
                    st.success(f"Evaluation done! Pass rate: {summary['pass_rate']:.1%}")
                    
                    # Show summary
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total", summary["total"])
                    c2.metric("Passed", summary["passed"])
                    c3.metric("Pass Rate", f"{summary['pass_rate']:.1%}")
                    c4.metric("Avg Score", f"{summary['avg_score']:.2f}")
                    
                    # By category
                    st.subheader("By Category")
                    cat_data = []
                    for cat, stats in summary.get("by_category", {}).items():
                        cat_data.append({"Category": cat, "Count": stats["count"], "Pass Rate": f"{stats['pass_rate']:.0%}", "Avg Score": f"{stats['avg_score']:.2f}"})
                    if cat_data:
                        import pandas as pd
                        st.dataframe(pd.DataFrame(cat_data), use_container_width=True)
                    
                    # Failed questions
                    failed = [r for r in results if not r.get("passed")]
                    if failed:
                        st.subheader(f"❌ Failed ({len(failed)})")
                        for r in failed[:10]:
                            with st.expander(f"{r['question_id']}: {r.get('overall_score',0):.2f} - {r.get('question','')[:60]}"):
                                st.markdown(f"**Question:** {r.get('question')}")
                                st.markdown(f"**Missed:** {r.get('kw_miss',[])}")
                                st.markdown(f"**Answer:** {r.get('generated_answer','')[:500]}")
                    else:
                        st.balloons()
                        st.success("All questions passed! 🎉")
                    
                    # Save results display
                    st.info(f"Results saved to {output_name}")
                    with open(output_name, 'r') as f:
                        st.download_button("📥 Download Results JSON", data=f.read(), file_name=output_name, mime="application/json")
                
                except Exception as e:
                    st.error(f"Evaluation failed: {e}")
                    import traceback; st.code(traceback.format_exc())

# ========== TAB 4: TEST CREATOR ==========
with tab_test_creator:
    st.header("✏️ Test Creator - Raise New Questions on Knowledge")
    st.markdown("Create new evaluation sets from your current knowledge base. Questions are saved to `test/` (gitignored).")
    
    if not st.session_state.pipeline:
        st.warning("Load models in Chat tab first to use Test Creator")
        st.stop()
    
    # Manual creation
    with st.expander("📝 Manual Question Creator", expanded=True):
        q_id = st.text_input("Question ID", value=f"q{int(time.time())%10000}")
        question = st.text_area("Question", placeholder="What was Apple's total net sales in 2025?")
        kw_input = st.text_input("Expected Keywords (comma separated)", placeholder="416,161, Total net sales")
        num_input = st.text_input("Expected Numbers (comma separated)", placeholder="416,161, 2025")
        category = st.selectbox("Category", ["table-numeric", "table-multi-row", "text-numeric", "comparison", "definition", "date", "general"])
        difficulty = st.selectbox("Difficulty", ["easy", "medium", "hard"])
        focus = st.text_input("Phase Focus", value="Camelot table extraction")
        
        if st.button("➕ Add to Current Test"):
            if not question:
                st.error("Question required")
            else:
                new_q = {
                    "id": q_id,
                    "question": question,
                    "expected_keywords": [k.strip() for k in kw_input.split(",") if k.strip()],
                    "expected_numbers": [k.strip() for k in num_input.split(",") if k.strip()],
                    "category": category,
                    "difficulty": difficulty,
                    "phase1_focus": focus
                }
                if "custom_test" not in st.session_state:
                    st.session_state.custom_test = []
                st.session_state.custom_test.append(new_q)
                st.success(f"Added {q_id} to custom test set ({len(st.session_state.custom_test)} total)")
    
    # Show current custom test
    if "custom_test" in st.session_state and st.session_state.custom_test:
        st.subheader(f"📋 Custom Test Set ({len(st.session_state.custom_test)} questions)")
        import pandas as pd
        df_custom = pd.DataFrame([{"ID": q["id"], "Question": q["question"][:60], "Category": q["category"]} for q in st.session_state.custom_test])
        st.dataframe(df_custom, use_container_width=True)
        
        col_save1, col_save2 = st.columns(2)
        with col_save1:
            save_name = st.text_input("Save as file in test/", value="my_new_test.json")
            if st.button("💾 Save to test/ folder", type="primary"):
                save_path = Path(config.TEST_DIR) / save_name
                save_path.parent.mkdir(exist_ok=True)
                with open(save_path, 'w') as f:
                    json.dump(st.session_state.custom_test, f, indent=2)
                st.success(f"Saved {len(st.session_state.custom_test)} questions to {save_path}")
        with col_save2:
            if st.button("🗑️ Clear custom test"):
                st.session_state.custom_test = []
                st.rerun()
    
    st.divider()
    
    # Auto-generate from knowledge using LLM
    with st.expander("🤖 Auto-Generate Questions from Knowledge (LLM)", expanded=False):
        st.markdown("Select a file from knowledge base, LLM will generate 5 test questions from it.")
        scanned_for_gen = scan_data_dir()
        if not scanned_for_gen:
            st.info("No files in data/")
        else:
            file_for_gen = st.selectbox("Select file to generate questions from", options=[f["name"] for f in scanned_for_gen], key="gen_file")
            num_q = st.slider("Number of questions", 1, 10, 5)
            
            if st.button("🤖 Generate Questions", type="primary"):
                with st.spinner(f"Generating {num_q} questions from {file_for_gen}..."):
                    try:
                        # Retrieve some chunks from that file
                        pipe = st.session_state.pipeline
                        # Get nodes for that file
                        # We need to load chunks for that file
                        from core.ingestion import load_pdf_with_fallback, _clean_text
                        target_file = None
                        for f in scanned_for_gen:
                            if f["name"] == file_for_gen:
                                target_file = Path(f["full_path"])
                                break
                        if not target_file:
                            st.error("File not found")
                        else:
                            # Extract text (first 3000 chars) to generate questions
                            if target_file.suffix.lower() == ".pdf":
                                text = load_pdf_with_fallback(target_file)[:8000]
                            else:
                                text = target_file.read_text()[:8000]
                            
                            gen_prompt = f"""<|im_start|>system
You are a test question generator for RAG evaluation.
Given a document excerpt, generate {num_q} diverse evaluation questions that test RAG accuracy.
Focus on numbers, tables, dates, definitions.
Return JSON list only, no explanation.
Each item: {{"id": "gen_q1", "question": "...", "expected_keywords": ["keyword1","keyword2"], "expected_numbers": ["number"], "category": "table-numeric", "difficulty": "medium", "phase1_focus": "Camelot extraction"}}
<|im_end|>
<|im_start|>user
Document: {text[:6000]}

Generate {num_q} questions as JSON list.
<|im_end|>
<|im_start|>assistant
"""
                            gen_text = pipe.llm.generate(gen_prompt, max_tokens=1500)
                            # Try to extract JSON
                            # Find [ ... ]
                            import re
                            json_match = re.search(r'\[.*\]', gen_text, re.DOTALL)
                            if json_match:
                                try:
                                    gen_questions = json.loads(json_match.group(0))
                                    st.success(f"Generated {len(gen_questions)} questions!")
                                    st.json(gen_questions)
                                    
                                    if "custom_test" not in st.session_state:
                                        st.session_state.custom_test = []
                                    st.session_state.custom_test.extend(gen_questions)
                                    st.info(f"Added to custom test set - now {len(st.session_state.custom_test)} questions total")
                                except Exception as e:
                                    st.error(f"Failed to parse JSON: {e}")
                                    st.code(gen_text)
                            else:
                                st.code(gen_text)
                                st.warning("Could not extract JSON list, check output above")
                    except Exception as e:
                        st.error(f"Generation failed: {e}")
                        import traceback; st.code(traceback.format_exc())