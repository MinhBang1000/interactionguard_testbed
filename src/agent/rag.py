# rag.py
import os
import json
import yaml
from glob import glob
from typing import List, Tuple, Dict, Optional

from docx import Document
from pdfminer.high_level import extract_text
from tqdm import tqdm

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src import settings

EMBED_MODEL = settings.EMBED_MODEL

# 1: benign, 2: poisoned_as, 3: tool_injection, 4: correlated_injection
MODE_MAP = settings.MODE_MAP


def load_jsonl_corpus(filepath: str) -> List[Tuple[str, Dict]]:
    """
    Load a JSONL corpus where each line is a document:
    {"_id": "...", "title": "...", "text": "...", "metadata": {...}}
    """
    docs = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            text = f"{obj.get('title', '')}\n{obj.get('text', '')}".strip()
            if not text:
                continue

            meta = obj.get("metadata") or {}
            meta.update(
                {
                    "doc_id": obj.get("_id"),
                    "title": obj.get("title"),
                    "source": filepath,
                    "type": "corpus_jsonl",
                }
            )
            docs.append((text, meta))
    return docs


def extract_text_from_file(filepath: str) -> str:
    ext = filepath.lower().split(".")[-1]
    try:
        if ext in ["txt", "md", "log", "csv", "tsv"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        if ext == "json":
            with open(filepath, "r", encoding="utf-8") as f:
                obj = json.load(f)
            return json.dumps(obj, indent=2, ensure_ascii=False)

        if ext in ["yaml", "yml"]:
            with open(filepath, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            return json.dumps(data, indent=2, ensure_ascii=False)

        if ext == "pdf":
            return extract_text(filepath)

        if ext == "docx":
            doc = Document(filepath)
            return "\n".join([p.text for p in doc.paragraphs])

        if ext in ["py", "js", "html", "css", "java", "cpp", "c", "ts", "go", "rs"]:
            with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()

        with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    except Exception as e:
        print(f"[RAG] Could not read {filepath}: {e}")
        return ""


def load_docs_from_folder(folder_path: str) -> List[Tuple[str, Dict]]:
    """
    Load docs from a single folder (e.g., docs/benign or docs/poisoned_as).
    Rules:
    - Skip queries.jsonl always
    - For *.jsonl: treat as corpus lines
    - Other files: fallback extract_text
    """
    patterns = ["**/*"]
    file_paths = []
    for pattern in patterns:
        file_paths.extend(glob(os.path.join(folder_path, pattern), recursive=True))

    docs: List[Tuple[str, Dict]] = []

    for fp in file_paths:
        if os.path.isdir(fp):
            continue

        filename = os.path.basename(fp)
        ext = fp.lower().split(".")[-1]

        # never embed queries file
        if filename == "queries.jsonl":
            continue

        # JSONL corpus
        if ext == "jsonl":
            docs.extend(load_jsonl_corpus(fp))
            continue

        # fallback for normal files
        text = extract_text_from_file(fp)
        if not text.strip():
            continue

        docs.append(
            (
                text,
                {
                    "source": fp,
                    "filename": filename,
                    "type": ext,
                },
            )
        )

    return docs


def setup_rag(
    mode: int = 1,
    chunk_size: int = None,
    chunk_overlap: int = None,
):
    """
    mode:
      1 -> benign
      2 -> poisoned_as
      3 -> tool_injection => But still load the benign chroma_db => Will add more modes here
      4 -> correlated_injection

    Will:
      - load existing data/vectorstore/<mode_name> if exists
      - else build from data/corpora/<mode_name>
    """
    if mode not in MODE_MAP:
        raise ValueError(f"mode must be one of {list(MODE_MAP.keys())}, got {mode}")

    chunk_size = settings.CHUNK_SIZE if chunk_size is None else chunk_size
    chunk_overlap = settings.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap

    mode_name = MODE_MAP[mode]
    if mode == 3: # Mode 3 = tool_injection: still load benign DB
        mode_name = MODE_MAP[1]

    docs_path = str(settings.corpus_dir(mode_name))
    persist_path = str(settings.vectorstore_dir(mode_name))
    os.makedirs(persist_path, exist_ok=True)

    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)

    # ----- Load if exists -----
    chroma_sqlite = os.path.join(persist_path, "chroma.sqlite3")
    if os.path.exists(chroma_sqlite):
        print(f"[RAG] Found existing Chroma DB ({mode_name}) at {persist_path}, loading...")
        db = Chroma(persist_directory=persist_path, embedding_function=embeddings)
        print("[RAG] Chroma DB loaded successfully")
        return db

    # ----- Build otherwise -----
    print(f"[RAG] No existing DB for mode={mode_name}. Building new index...")
    print(f"[RAG] Loading documents from {docs_path}")

    raw_docs = load_docs_from_folder(docs_path)
    if not raw_docs:
        raise ValueError(f"No readable documents inside: {docs_path}")

    print(f"[RAG] Loaded {len(raw_docs)} documents")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    all_chunks: List[str] = []
    all_metadatas: List[Dict] = []

    print("[RAG] Splitting documents into chunks...")
    for text, meta in tqdm(raw_docs, desc="Chunking docs"):
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        all_metadatas.extend([meta] * len(chunks))

    print(f"[RAG] Total chunks: {len(all_chunks)}")
    print("[RAG] Building Chroma vector database (embedding)...")

    db = Chroma.from_texts(
        texts=all_chunks,   # NOTE: must be list, not tqdm wrapper
        embedding=embeddings,
        metadatas=all_metadatas,
        persist_directory=persist_path,
    )
    db.persist()
    print(f"[RAG] ✅ Vector DB built and saved at {persist_path}")

    return db
