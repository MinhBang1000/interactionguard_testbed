"""
datagen/reduce_corpus.py

Stage 0 of the pipeline: subsample the raw BEIR "nq" (Natural Questions)
dump down to a manageable benign corpus + the queries that still have a
relevant document in that sample.

Same algorithm as the original `docs/reduce_nq_dataset.py`
(random.sample with a fixed seed, qrels-based query filtering) — only the
paths changed: it now reads the raw BEIR layout directly
(`data/raw/nq/{corpus.jsonl,queries.jsonl,qrels/test.tsv}`) and writes the
final benign corpus/queries straight into `data/corpora/benign/`
(no more manual "new_queries.jsonl -> queries.jsonl" rename step).

The raw BEIR "nq" dataset (~1.5GB uncompressed) is NOT shipped in this
repo. Download it separately (e.g. from the BEIR benchmark, dataset id
"nq") and extract it so that `data/raw/nq/corpus.jsonl`,
`data/raw/nq/queries.jsonl` and `data/raw/nq/qrels/test.tsv` exist, or
point `RAW_NQ_DIR` in `.env` at wherever you extracted it.
"""

from pathlib import Path
from typing import Optional
import csv
import random

from src import settings
from src.datagen.io_utils import read_jsonl, write_jsonl


def reduce_corpus(
    raw_nq_dir: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    max_corpus: Optional[int] = None,
    seed: Optional[int] = None,
) -> None:
    raw_nq_dir = Path(raw_nq_dir) if raw_nq_dir else settings.RAW_NQ_DIR
    out_dir = Path(out_dir) if out_dir else settings.corpus_dir("benign")
    max_corpus = settings.REDUCE_MAX_CORPUS if max_corpus is None else max_corpus
    seed = settings.DATA_SEED if seed is None else seed

    corpus_path = raw_nq_dir / "corpus.jsonl"
    queries_path = raw_nq_dir / "queries.jsonl"
    qrels_path = raw_nq_dir / "qrels" / "test.tsv"

    for p in (corpus_path, queries_path, qrels_path):
        if not p.exists():
            raise FileNotFoundError(
                f"Missing raw BEIR nq file: {p}. Download/extract the BEIR 'nq' "
                f"dataset into {raw_nq_dir} first (see module docstring)."
            )

    random.seed(seed)

    print("[1] Loading corpus...")
    corpus = read_jsonl(corpus_path)
    print(f"    Total corpus docs: {len(corpus)}")

    print(f"[2] Sampling {max_corpus} corpus documents...")
    if len(corpus) > max_corpus:
        corpus_sample = random.sample(corpus, max_corpus)
    else:
        corpus_sample = corpus
    selected_doc_ids = {doc["_id"] for doc in corpus_sample}
    print(f"    Selected corpus docs: {len(corpus_sample)}")

    print("[3] Loading qrels...")
    selected_query_ids = set()
    with qrels_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            qid = row["query-id"]
            did = row["corpus-id"]
            score = int(row["score"])
            if score > 0 and did in selected_doc_ids:
                selected_query_ids.add(qid)
    print(f"    Queries linked to sampled corpus: {len(selected_query_ids)}")

    print("[4] Filtering queries...")
    filtered_queries = [
        obj for obj in read_jsonl(queries_path) if obj["_id"] in selected_query_ids
    ]
    print(f"    Final queries: {len(filtered_queries)}")

    out_corpus_path = out_dir / "corpus.jsonl"
    out_queries_path = out_dir / "queries.jsonl"

    print(f"[5] Writing {out_corpus_path}")
    write_jsonl(out_corpus_path, corpus_sample)

    print(f"[6] Writing {out_queries_path}")
    write_jsonl(out_queries_path, filtered_queries)

    print("\nDONE")
    print(f"   Corpus:  {out_corpus_path} ({len(corpus_sample)})")
    print(f"   Queries: {out_queries_path} ({len(filtered_queries)})")


if __name__ == "__main__":
    reduce_corpus()
