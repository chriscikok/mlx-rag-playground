"""
Ingestion: Phase 1 - Camelot for borderless tables + SemanticSplitter for better chunking
General purpose, no hardcoded keywords
"""
from pathlib import Path
from llama_index.core import Document
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
    if readable / total < 0.55:
        return True
    if len(re.findall(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', text)) > len(text) * 0.05:
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
    if not table or len(table) < 1:
        return ""
    cleaned = [[str(cell).strip() if cell is not None else "" for cell in row] for row in table]
    cleaned = [row for row in cleaned if any(c for c in row)]
    if not cleaned or len(cleaned[0]) < 2:
        return ""
    md = []
    md.append("| " + " | ".join(cleaned[0]) + " |")
    md.append("| " + " | ".join(["---"] * len(cleaned[0])) + " |")
    for row in cleaned[1:]:
        if len(row) < len(cleaned[0]):
            row = row + [""] * (len(cleaned[0]) - len(row))
        md.append("| " + " | ".join(row[:len(cleaned[0])]) + " |")
    return "\n".join(md)

def _extract_with_camelot(pdf_path: Path):
    """Phase 1: Camelot stream flavor - best for borderless tables"""
    try:
        import camelot
        tables_by_page = {}
        try:
            tables = camelot.read_pdf(str(pdf_path), pages='all', flavor='stream', strip_text='\n')
            for t in tables:
                try:
                    page = int(t.page) if hasattr(t, 'page') else 1
                    df = t.df
                    if df.shape[1] >= 2 and df.shape[0] >= 2:
                        cleaned = df.astype(str).values.tolist()
                        cleaned = [r for r in cleaned if any(c.strip() for c in r)]
                        if len(cleaned) >= 2:
                            md = _table_to_markdown(cleaned)
                            if md and len(md) > 30:
                                tables_by_page.setdefault(page, []).append(md)
                except:
                    continue
        except Exception as e:
            print(f"[Camelot] read_pdf failed: {e}")
        total = sum(len(v) for v in tables_by_page.values())
        if total:
            print(f"[Ingestion] Camelot stream: found {total} tables from {pdf_path.name}")
        return tables_by_page
    except ImportError:
        print("[Ingestion] Camelot not installed - pip install camelot-py")
        return {}
    except Exception as e:
        print(f"[Ingestion] Camelot failed: {e}")
        return {}

def _extract_with_tabula(pdf_path: Path):
    try:
        import tabula
        tables_by_page = {}
        dfs = tabula.read_pdf(str(pdf_path), pages='all', multiple_tables=True, silent=True)
        for df in dfs:
            try:
                if df.shape[1] >= 2 and df.shape[0] >= 2:
                    cleaned = df.astype(str).values.tolist()
                    cleaned = [r for r in cleaned if any(c.strip() for c in r)]
                    if len(cleaned) >= 2:
                        md = _table_to_markdown(cleaned)
                        if md and len(md) > 30:
                            tables_by_page.setdefault(1, []).append(md)
            except:
                continue
        if tables_by_page:
            print(f"[Ingestion] Tabula: found {sum(len(v) for v in tables_by_page.values())} tables")
        return tables_by_page
    except:
        return {}

def _extract_with_ocr(pdf_path: Path) -> str:
    """OCR fallback using tesseract for scanned PDFs - Uses PyMuPDF, no poppler needed"""
    try:
        import fitz
        import pytesseract
        from PIL import Image
        print(f"[Ingestion] Trying OCR with tesseract (fitz) for {pdf_path.name}...")
        doc = fitz.open(str(pdf_path))
        ocr_texts = []
        for i in range(len(doc)):
            try:
                page = doc[i]
                pix = page.get_pixmap(dpi=300)
                img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                txt = pytesseract.image_to_string(img)
                if txt and len(txt.strip()) > 50:
                    ocr_texts.append(f"[OCR Page {i+1}]\n{txt}")
            except Exception as e:
                print(f"[OCR] Page {i+1} failed: {e}")
                continue
        doc.close()
        result = "\n\n".join(ocr_texts)
        if result:
            print(f"[Ingestion] OCR extracted {len(result)} chars from {len(ocr_texts)} pages")
        return result
    except ImportError as e:
        print(f"[Ingestion] OCR deps missing: {e}")
        return ""
    except Exception as e:
        print(f"[Ingestion] OCR failed (brew install tesseract): {e}")
        return ""

def _extract_pdf_with_tables(pdf_path: Path) -> str:
    full_text = []
    tables_found = 0
    camelot_tables = {}
    try:
        import pdfplumber
        camelot_tables = _extract_with_camelot(pdf_path)
        with pdfplumber.open(str(pdf_path)) as pdf:
            for page_num, page in enumerate(pdf.pages):
                page_idx = page_num + 1
                text = page.extract_text() or ""
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
                    if text:
                        full_text.append(text)
                    for table in tables:
                        md_table = _table_to_markdown(table)
                        if md_table and len(md_table) > 30 and len(table) >= 2:
                            full_text.append(f"\n[TABLE_START page={page_idx}]\n{md_table}\n[TABLE_END]\n")
                            tables_found += 1
                else:
                    c_tables = camelot_tables.get(page_idx, [])
                    if c_tables:
                        if text:
                            full_text.append(text)
                        for md_table in c_tables:
                            full_text.append(f"\n[TABLE_START page={page_idx} camelot]\n{md_table}\n[TABLE_END]\n")
                            tables_found += 1
                    else:
                        if text:
                            full_text.append(text)
        print(f"[Ingestion] pdfplumber+camelot: tables_found={tables_found} from {pdf_path.name}")
        result = "\n\n".join(full_text)
        if result and len(result.strip()) > 100:
            return _clean_text(result)
    except ImportError as e:
        print(f"[Ingestion] pdfplumber not installed: {e}")
    except Exception as e:
        print(f"[Ingestion] pdfplumber failed: {e}")
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
            for page_idx, mds in camelot_tables.items():
                for md in mds:
                    result += f"\n\n[TABLE_START page={page_idx} camelot]\n{md}\n[TABLE_END]"
            print(f"[Ingestion] fitz fallback: {len(result)} chars")
            return _clean_text(result)
    except Exception as e:
        print(f"[Ingestion] fitz failed: {e}")
    return ""

def load_pdf_with_fallback(pdf_path: Path) -> str:
    text = _extract_pdf_with_tables(pdf_path)
    if text and not _is_garbled(text) and len(text.strip()) > 200:
        return _clean_text(text)
    ocr_text = _extract_with_ocr(pdf_path)
    if ocr_text and len(ocr_text.strip()) > 200:
        return _clean_text(ocr_text)
    return text or ""

def _split_preserve_tables(text: str):
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
    if not parts:
        parts = [('text', text)]
    return parts

def get_semantic_splitter(embed_model=None):
    try:
        from llama_index.core.node_parser import SemanticSplitterNodeParser
        from llama_index.core import Settings
        em = embed_model or Settings.embed_model
        if em is None:
            raise ValueError("No embed model")
        print(f"[Ingestion] Using SemanticSplitter with {config.CHUNK_SIZE}")
        return SemanticSplitterNodeParser(
            buffer_size=1,
            breakpoint_percentile_threshold=95,
            embed_model=em
        )
    except Exception as e:
        print(f"[Ingestion] SemanticSplitter not available ({e}), falling back to SentenceSplitter")
        from llama_index.core.node_parser import SentenceSplitter
        return SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )

def load_and_chunk(data_dir=None, embed_model=None):
    data_dir = Path(data_dir) if data_dir else config.DATA_DIR
    print(f"Loading docs from {data_dir}... mode={config.CHUNKING_MODE}")
    docs = []
    supported_exts = {".pdf", ".docx", ".md", ".txt", ".html"}
    for file_path in data_dir.rglob("*"):
        if not file_path.is_file() or file_path.suffix.lower() not in supported_exts:
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
    print(f"Loaded {len(docs)} docs (tables: {sum(1 for d in docs if d.metadata.get('is_table'))})")
    if config.CHUNKING_MODE == "semantic":
        parser = get_semantic_splitter(embed_model=embed_model)
    else:
        from llama_index.core.node_parser import SentenceSplitter
        parser = SentenceSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
        )
    all_nodes = []
    for doc in docs:
        if doc.metadata.get('is_table'):
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
    print(f"[Ingestion] {len(table_chunks)} table chunks preserved (mode={config.CHUNKING_MODE})")
    return docs, all_nodes