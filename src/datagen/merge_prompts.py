"""
datagen/merge_prompts.py

Merges the various prompt sources into two flat pools used for quick
sampling/inspection: a benign pool and a malicious pool.

This consolidates two previously separate scripts:
  - `build_index.py` (misleadingly named — it had nothing to do with the
    RAG/Chroma index; it merged benign prompts into
    `benign_user_prompts.jsonl`)
  - `collect_all_malicious_prompt.py` (merged malicious prompts into
    `malicious_user_prompts.jsonl`, with id/dedup handling)

`collect_all_benign_prompt.py` was empty in the original repo (dead file,
nothing to preserve).

Schemas are unchanged from the originals: the benign pool is
`{source, text}`; the malicious pool is `{id, source, attack_type, text}`.
"""

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from src import settings
from src.datagen.io_utils import read_jsonl, write_jsonl, norm_text, hash_text


# =========================
# Benign pool
# =========================

def merge_benign_prompts(out_path: Optional[Path] = None) -> Path:
    out_path = Path(out_path) if out_path else settings.BENIGN_PROMPT_POOL_PATH

    benign_prompts: List[Dict] = []

    for obj in read_jsonl(settings.DEEPSET_PROMPT_INJECTIONS_PATH):
        if obj.get("label") == 0:
            benign_prompts.append({"source": "deepset", "text": obj["text"].strip()})

    for obj in read_jsonl(settings.TOOL_INJECTION_DATASET_PATH):
        benign_prompts.append({"source": "tool_injection", "text": obj["user_prompt"].strip()})

    for obj in read_jsonl(settings.corpus_dir("benign") / "queries.jsonl"):
        benign_prompts.append({"source": "natural_questions", "text": obj["text"].strip()})

    write_jsonl(out_path, benign_prompts)
    print(f"Total benign prompts collected: {len(benign_prompts)} -> {out_path}")
    return out_path


# =========================
# Malicious pool
# =========================

def _make_id(attack_type: str, source: str, text: str) -> str:
    return f"{attack_type}_{hash_text(f'{attack_type}||{source}||{norm_text(text)}')[:16]}"


def _safe_get_text(obj: Dict, candidates: List[str]) -> str:
    for k in candidates:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


def _collect_prompt_injection(path: Path) -> List[Dict]:
    rows = []
    for obj in read_jsonl(path):
        if obj.get("label") != 1:
            continue  # only malicious
        text = _safe_get_text(obj, ["text", "prompt"])
        if not text:
            continue
        attack_type = obj.get("attack_type") or "prompt"
        source = obj.get("source") or "deepset"
        rid = obj.get("id") or _make_id(attack_type, source, text)
        rows.append({"id": rid, "source": source, "attack_type": attack_type, "text": text})
    return rows


def _collect_tool_injection(path: Path) -> List[Dict]:
    rows = []
    for obj in read_jsonl(path):
        text = _safe_get_text(obj, ["user_prompt", "text", "prompt", "query"])
        if not text:
            continue
        attack_type = obj.get("attack_type") or "tool"
        source = obj.get("source") or "tool_injection"
        rid = obj.get("id") or _make_id(attack_type, source, text)
        rows.append({"id": rid, "source": source, "attack_type": attack_type, "text": text})
    return rows


def _collect_queries_as_attack(path: Path, attack_type: str, source: str) -> List[Dict]:
    """rag / correlated: same NQ query pool, tagged with the given attack_type."""
    rows = []
    for obj in read_jsonl(path):
        text = _safe_get_text(obj, ["text", "query", "prompt", "question"])
        if not text:
            continue
        rid = obj.get("id") or _make_id(attack_type, source, text)
        rows.append({"id": rid, "source": source, "attack_type": attack_type, "text": text})
    return rows


def _dedup_rows(rows: Iterable[Dict], mode: str) -> List[Dict]:
    """mode: 'none' | 'text' | 'type_text' (default, keeps correlated even if text overlaps rag)."""
    if mode == "none":
        return list(rows)

    seen = set()
    out = []
    for r in rows:
        key = norm_text(r["text"]) if mode == "text" else (r["attack_type"], norm_text(r["text"]))
        if key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def merge_malicious_prompts(out_path: Optional[Path] = None, dedup_mode: str = "type_text") -> Path:
    out_path = Path(out_path) if out_path else settings.MALICIOUS_PROMPT_POOL_PATH
    benign_queries_path = settings.corpus_dir("benign") / "queries.jsonl"

    rows: List[Dict] = []
    rows += _collect_prompt_injection(settings.DEEPSET_PROMPT_INJECTIONS_PATH)
    rows += _collect_tool_injection(settings.TOOL_INJECTION_DATASET_PATH)
    rows += _collect_queries_as_attack(benign_queries_path, "rag", "poison_queries")
    rows += _collect_queries_as_attack(benign_queries_path, "correlated", "correlated_queries")

    before = len(rows)
    rows = _dedup_rows(rows, dedup_mode)
    after = len(rows)

    write_jsonl(out_path, rows)

    c1 = Counter(r["attack_type"] for r in rows)
    c2 = Counter(r["source"] for r in rows)
    print(f"Total collected: {before}")
    print(f"After dedup ({dedup_mode}): {after}")
    print("By attack_type:", dict(c1))
    print("By source:", dict(c2))
    print(f"Saved to: {out_path}")
    return out_path


def merge(kind: str, out_path: Optional[Path] = None, dedup_mode: str = "type_text") -> Path:
    """kind: 'benign' | 'malicious'"""
    if kind == "benign":
        return merge_benign_prompts(out_path)
    if kind == "malicious":
        return merge_malicious_prompts(out_path, dedup_mode)
    raise ValueError(f"Unknown kind: {kind}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["benign", "malicious"], required=True)
    parser.add_argument("--dedup", choices=["none", "text", "type_text"], default="type_text")
    args = parser.parse_args()

    merge(args.kind, dedup_mode=args.dedup)
