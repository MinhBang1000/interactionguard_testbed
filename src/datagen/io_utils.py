"""
datagen/io_utils.py

Small IO/text helpers shared across the data-generation scripts. These are
extracted verbatim (same behavior) from what used to be copy-pasted in
generate_poisoned_as.py, generate_correlated_injection.py,
preprocess_benign_data.py and collect_all_malicious_prompt.py.
"""

from pathlib import Path
from typing import Any, Dict, List, Union
import hashlib
import json
import re

PathLike = Union[str, Path]


def read_jsonl(path: PathLike) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
    return items


def write_jsonl(path: PathLike, items: List[Dict[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for obj in items:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def load_queries(path: PathLike) -> List[Dict[str, str]]:
    """Normalize a queries.jsonl file into [{"_id", "text"}, ...]."""
    qs: List[Dict[str, str]] = []
    for obj in read_jsonl(path):
        qid = obj.get("_id") or obj.get("query-id") or obj.get("id")
        qtext = obj.get("text") or obj.get("query") or obj.get("question")
        if qid and qtext:
            qs.append({"_id": qid, "text": qtext})
    return qs


_ws_re = re.compile(r"\s+")


def norm_text(text: str, lowercase: bool = True) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if lowercase:
        s = s.lower()
    return _ws_re.sub(" ", s)


def hash_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()
