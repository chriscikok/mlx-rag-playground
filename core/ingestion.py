
"""
Ingestion: General purpose + Table-aware
- Keeps markdown tables intact (no split)
- Handles borderless tables via pdfplumber text strategy
- No hardcoded keywords - uses structural heuristics
"""
from pathlib import Path
from llama_index.core import Document
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
import config
import re

def _is_garbled(text: str) -> bool:
    if not text or len(text.strip()) < 50:
        return True
    readable = len(re.findall(r'[a-zA-Z0-9 .,;:()\-/%$\n]+', text))
    total = len(text)
    if total == 0:
        return True
    ratio = readable / total
    if ratio < 0.55:
        return True
    control_chars = len(re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', text))
    if control_chars > len(text) * 0.05:
        return True
    return False

def _clean_text(text: str) -> str:
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{3,}', ' ', text)
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('’', "'").replace('‘', "'")
    text = text.replace('“', '"').replace('”', '"').replace('–', '-').replace('—', '-')
    text = re.sub(r'(?i)Page\s+\d+\s+of\s+\d+', '', text)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'\.{4,}', ' ', text)
    return text.strip()

def _table_to_markdown(table):
    """Convert list of lists to markdown - generic"""
    if not table or len(table) < 1:
        return ""
    cleaned = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]
    cleaned = [row for row in cleaned if any(c for c in row)]
    if not cleaned:
        return ""
    # Require at least 2 columns to be a real table
    if len(cleaned[0]) < 2:
        return ""
    md = []
    md.append("| " + " | ".join(cleaned[0]) + " |")
    md.append("| " + " | ".join(["---"] * len(cleaned[0])) + " |")
    for row in cleaned[1:]:
        if len(row) < len(cleaned[0]):
            row = row + [""] * (len(cleaned[0]) - len(row))
        md.append("| " + " | ".join(row[:len(cleaned[0])]) + " |")
    return "\n".join(md)

def _is_table_like_block(block: str) -> bool:
    """
    General heuristic to detect table-like text blocks
    No hardcoded keywords - uses structure:
    - Multiple lines with similar column counts (split by 2+ spaces or |)
    - Lines containing numbers/currencies/percentages
    - Consistent delimiters
    """
    lines = [l.strip() for l in block.split('\n') if l.strip()]
    if len(lines) < 2:
        return False
    
    # Count lines that look tabular
    tabular_lines = 0
    for line in lines:
        # Split by 2+ spaces (common in PDFs) or | or tabs
        cols = re.split(r'\s{2,}|\t|\|', line)
        cols = [c.strip() for c in cols if c.strip()]
        
        # Heuristics for tabular line:
        # - 2+ columns
        # - Contains number/currency/percent
        # - Not a full sentence (no ending period + many words)
        # General: any digit, currency symbol, % or 3-letter code like USD/HKD/EUR
        has_number = bool(re.search(r'\d|\$|%|€|£|¥|\b[A-Z]{3}\b', line))
        has_multiple_cols = len(cols) >= 2
        
        # Avoid sentences: tabular lines are short per column
        avg_col_len = sum(len(c) for c in cols) / len(cols) if cols else 100
        is_short_cols = avg_col_len < 40
        
        if has_multiple_cols and (has_number or is_short_cols):
            tabular_lines += 1
    
    # If >60% lines are tabular, consider it a table block
    return tabular_lines >= max(2, len(lines) * 0.6)

def _extract_pdf_with_tables(pdf_path: Path) -> str:
    full_text = []
    tables_found = 0
    
    try:
        import pdfplumber
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                text = page.extract_text() or ""
                
                # Try multiple table extraction strategies - generic
                tables = []
                try:
                    tables = page.extract_tables() or []
                except:
                    tables = []
                
                if not tables:
                    try:
                        tables = page.extract_tables(table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                            "snap_tolerance": 3,
                            "join_tolerance": 3,
                            "text_x_tolerance": 3,
                            "text_y_tolerance": 3
                        }) or []
                    except:
                        tables = []
                
                if not tables:
                    try:
                        for tbl in page.find_tables():
                            try:
                                tables.append(tbl.extract())
                            except:
                                continue
                    except:
                        pass
                
                if tables:
                    # Keep original text plus tables as markdown
                    if text:
                        full_text.append(text)
                    for table in tables:
                        md_table = _table_to_markdown(table)
                        if md_table and len(md_table) > 30:
                            # Filter out noise tables (1 row or all empty)
                            if len(table) >= 2:
                                full_text.append(f"\n[TABLE_START page={page_num+1}]\n{md_table}\n[TABLE_END]\n")
                                tables_found += 1
                else:
                    # No lattice tables found - keep text as-is
                    # Our chunking will later group table-like lines together
                    if text:
                        full_text.append(text)
        
        print(f"[Ingestion] pdfplumber: tables_found={tables_found} from {pdf_path.name}")
        result = "\n\n".join(full_text)
        if result and len(result.strip()) > 100:
            print(f"[Ingestion] pdfplumber total: {len(result)} chars (tables: {tables_found})")
            return _clean_text(result)
            
    except ImportError as e:
        print(f"[Ingestion] pdfplumber not installed: {e}")
    except Exception as e:
        print(f"[Ingestion] pdfplumber failed: {e}")
    
    # Fallback to fitz
    try:
        import fitz
        doc = fitz.open(str(pdf_path))
        fitz_text = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            t = page.get_text("text")
            if t:
                fitz_text.append(t)
        doc.close()
        result = "\n\n".join(fitz_text)
        if result:
            print(f"[Ingestion] fitz fallback: {len(result)} chars")
            return _clean_text(result)
    except Exception as e:
        print(f"[Ingestion] fitz failed: {e}")
    
    return ""

def _extract_pdf_pypdf(pdf_path: Path) -> str:
    try:
        from pypdf import PdfReader
        reader = PdfReader(str(pdf_path))
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text()
                if t:
                    texts.append(t)
            except:
                continue
        return "\n\n".join(texts)
    except Exception as e:
        print(f"[Ingestion] pypdf failed: {e}")
        return ""

def load_pdf_with_fallback(pdf_path: Path) -> str:
    text = _extract_pdf_with_tables(pdf_path)
    if text and not _is_garbled(text) and len(text.strip()) > 200:
        return _clean_text(text)
    text2 = _extract_pdf_pypdf(pdf_path)
    if text2 and not _is_garbled(text2) and len(text2) > len(text or ""):
        return _clean_text(text2)
    if text:
        return _clean_text(text)
    return ""

def _split_preserve_tables(text: str):
    """
    General purpose split: preserve [TABLE_START] blocks and also detect table-like blocks
    without markers (borderless tables)
    """
    # First, extract explicit table markers
    table_pattern = r'\[TABLE_START.*?\].*?\[TABLE_END\]'
    parts = []
    last_end = 0
    
    for m in re.finditer(table_pattern, text, re.DOTALL):
        before = text[last_end:m.start()].strip()
        if before:
            parts.append(('text', before))
        parts.append(('table', m.group(0).strip()))
        last_end = m.end()
    
    after = text[last_end:].strip()
    if after:
        parts.append(('text', after))
    
    # If no explicit markers, try to detect markdown tables
    if len(parts) == 1 and parts[0][0] == 'text':
        md_table_pattern = r'(\|.*\|\n\|\s*---.*\n(?:\|.*\|\n?)+)'
        new_parts = []
        last_end = 0
        for m in re.finditer(md_table_pattern, parts[0][1]):
            before = parts[0][1][last_end:m.start()].strip()
            if before:
                new_parts.append(('text', before))
            new_parts.append(('table', m.group(0).strip()))
            last_end = m.end()
        remaining = parts[0][1][last_end:].strip()
        if remaining:
            new_parts.append(('text', remaining))
        if len(new_parts) > 1:
            parts = new_parts
    
    # Fallback: treat whole text as one part
    if not parts:
        parts = [('text', text)]
    
    return parts

def load_and_chunk(data_dir=None):
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    print(f"Loading docs from {data_dir}...")
    
    docs = []
    supported_exts = {".pdf", ".docx", ".md", ".txt", ".html"}
    
    for file_path in data_dir.rglob("*"):
        if not file_path.is_file():
            continue
        if file_path.suffix.lower() not in supported_exts:
            continue
        
        print(f"[Ingestion] Processing {file_path.name}...")
        text = ""
        
        if file_path.suffix.lower() == ".pdf":
            text = load_pdf_with_fallback(file_path)
        else:
            try:
                from llama_index.core import SimpleDirectoryReader
                file_docs = SimpleDirectoryReader(input_files=[str(file_path)]).load_data()
                if file_docs:
                    text = "\n\n".join([d.text for d in file_docs])
                    text = _clean_text(text)
            except Exception as e:
                print(f"[Ingestion] Failed {file_path}: {e}")
                continue
        
        if not text or len(text.strip()) < 50:
            print(f"[Ingestion] Skip {file_path.name} - no text")
            continue
        
        parts = _split_preserve_tables(text)
        print(f"[Ingestion] Split into {len(parts)} parts (tables: {sum(1 for t,_ in parts if t=='table')})")
        
        for part_type, part_text in parts:
            if not part_text or len(part_text.strip()) < 20:
                continue
            
            doc = Document(
                text=part_text,
                metadata={
                    "file_name": file_path.name,
                    "file_path": str(file_path),
                    "file_type": file_path.suffix,
                    "char_count": len(part_text),
                    "is_table": part_type == 'table'
                }
            )
            docs.append(doc)
            if part_type == 'table':
                print(f"[Ingestion] Table doc: {len(part_text)} chars")
    
    print(f"Loaded {len(docs)} docs (tables: {sum(1 for d in docs if d.metadata.get('is_table'))})")
    
    parser = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        paragraph_separator="\n\n",
        secondary_chunking_regex="[^,.;。]+[,.;。]?"
    )
    
    all_nodes = []
    for doc in docs:
        if doc.metadata.get('is_table'):
            # Keep tables intact unless very large
            if len(doc.text) > config.CHUNK_SIZE * 6:
                nodes = parser.get_nodes_from_documents([doc])
                all_nodes.extend(nodes)
            else:
                node = TextNode(text=doc.text, metadata=doc.metadata)
                all_nodes.append(node)
        else:
            nodes = parser.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)
    
    print(f"-> {len(all_nodes)} chunks (avg {sum(len(n.text) for n in all_nodes)//len(all_nodes) if all_nodes else 0} chars)")
    table_chunks = [n for n in all_nodes if n.metadata.get('is_table')]
    print(f"[Ingestion] {len(table_chunks)} table chunks preserved (general purpose, no hardcoded keywords)")
    
    if all_nodes:
        print(f"[Ingestion] Sample: {all_nodes[0].text[:300]}...")
    
    return docs, all_nodes