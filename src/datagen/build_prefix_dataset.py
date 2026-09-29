#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
datagen/build_prefix_dataset.py

Menu-based pipeline:
Benign Logs  -> (Split Final Answers) -> (Optional Mix Malicious Into Test)
            -> (Optional Split Mixed Test into VAL/TEST)
            -> (Prefix Samples) -> Train/Test JSONL
(for Layer-1 Anomaly Detection)

Modes:
1) Split-only:
   - output train_final.jsonl / test_final_benign.jsonl
   - optionally output test_final_mixed.jsonl (+ mix_report.json)
   - optionally output val_final_mixed.jsonl + test_final_mixed.json
2) Split-then-preprocess:
   - output train.jsonl / test.jsonl (prefix samples)
   - optionally output train_final.jsonl / test_final_benign.jsonl / test_final_mixed.jsonl
   - optionally output val_final_mixed.jsonl + test_final_mixed.json
   - optionally preprocess val/test separately (val.jsonl + test.jsonl)
3) Preprocess-only:
   - input one file -> output processed.jsonl (prefix samples)

Key:
- Conversation-level shuffle + split
- Prefix expansion (N nodes -> N samples)
- Lowercase + strip + normalize whitespace (NO truncate)
- Deduplicate samples:
  - within_split: dedup train and test separately
  - global_no_leak: ensure NO identical sample exists in both train and test

Unchanged from the original `preprocess_benign_data.py` — only the default
input/output directories now come from `settings.py` (`data/traces/` for
collected logs, `data/processed/` for the resulting train/val/test JSONL).
"""

import json
import random
import hashlib
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple, Optional

from src import settings

# =========================
# CONFIG: default collected-trace files
# =========================
DEFAULT_COLLECTED_DIR = settings.TRACES_DIR

DEFAULT_ATTACK_FILES = {
    "prompt": DEFAULT_COLLECTED_DIR / "prompt_injection_traces.jsonl",
    "rag": DEFAULT_COLLECTED_DIR / "rag_poison_traces.jsonl",
    "tool": DEFAULT_COLLECTED_DIR / "tool_injection_traces.jsonl",
    "correlated": DEFAULT_COLLECTED_DIR / "correlated_injection_traces.jsonl",
}


# =========================
# Data structure
# =========================

@dataclass
class PrefixSample:
    sample_id: str
    origin_id: str
    source: str
    content: str
    label: int
    meta: Dict[str, Any]


# =========================
# Builder class
# =========================

class BenignPrefixDatasetBuilder:
    def __init__(
        self,
        seed: int = 42,
        sep_token: str = "[SEP]",
        normalize_lowercase: bool = True,
    ):
        self.seed = seed
        self.sep_token = sep_token
        self.normalize_lowercase = normalize_lowercase
        random.seed(seed)

    # ---------- IO ----------

    def load_logs(self, path: Path) -> List[Dict[str, Any]]:
        """
        Supports:
        - JSON array file: [...]
        - JSONL file: one JSON object per line
        """
        if not path.exists():
            print(f"[WARN] File not found: {path}")
            return []

        text = path.read_text(encoding="utf-8", errors="replace").strip()
        if not text:
            return []
        if text.startswith("["):
            return json.loads(text)

        logs = []
        with path.open("r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line:
                    logs.append(json.loads(line))
        return logs

    def write_jsonl(self, items: List[Dict[str, Any]], path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for obj in items:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")

    def write_json_array(self, items: List[Dict[str, Any]], path: Path) -> None:
        """Write JSON array (not JSONL)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

    # ---------- Preprocess ----------

    def normalize(self, text: str) -> str:
        """
        Normalize whitespace, keep newlines structure lightly, NO truncate.
        """
        if text is None:
            text = ""
        text = str(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = text.strip()
        if self.normalize_lowercase:
            text = text.lower()
        return text

    def tag(self, name: str, text: str) -> str:
        return f"[{name}] {self.normalize(text)}"

    # ---------- Node extraction ----------

    def extract_nodes(self, conv: Dict[str, Any]) -> List[str]:
        """
        nodes order:
        1) [PROMPT] conv.prompt
        2) [MEMORY] retrieve step content
        3) [TOOL:<name>] tool result content
        4) [REASON] final_answer
        """
        nodes: List[str] = []

        if conv.get("prompt"):
            nodes.append(self.tag("PROMPT", conv["prompt"]))

        for step in conv.get("reasoning_steps", []):
            node = step.get("node", "")
            typ = step.get("type", "")
            content = step.get("content", "")

            if not content or not str(content).strip():
                continue

            if node == "retrieve":
                nodes.append(self.tag("MEMORY", content))
            elif typ == "tool":
                tool = step.get("tool", "unknown")
                nodes.append(self.tag(f"TOOL:{tool}", self._tool_text(content)))

        if conv.get("final_answer"):
            nodes.append(self.tag("REASON", conv["final_answer"]))

        return nodes

    def _tool_text(self, content: Any) -> str:
        if isinstance(content, dict):
            return str(content.get("result", content))
        if isinstance(content, str):
            s = content.strip()
            try:
                obj = json.loads(s)
                if isinstance(obj, dict):
                    return str(obj.get("result", obj))
            except Exception:
                pass
            return s
        return str(content)

    # ---------- Prefix expansion ----------

    def build_prefixes(self, conv: Dict[str, Any]) -> List[PrefixSample]:
        nodes = self.extract_nodes(conv)
        samples: List[PrefixSample] = []

        origin_id = conv.get("id", "unknown_id")
        source = conv.get("source", "unknown")

        for i in range(1, len(nodes) + 1):
            content = f"\n{self.sep_token}\n".join(nodes[:i])
            samples.append(
                PrefixSample(
                    sample_id=f"{origin_id}_p{i}",
                    origin_id=origin_id,
                    source=source,
                    content=content,
                    label=int(conv.get("label", 0)),
                    meta={
                        "prefix_index": i,
                        "num_nodes": len(nodes),
                        "trace": conv.get("trace", []),
                        "attack_type": conv.get("attack_type", None),
                    },
                )
            )
        return samples

    # ---------- Dedup ----------

    @staticmethod
    def _hash_text(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def deduplicate_samples(self, samples: List[PrefixSample]) -> List[PrefixSample]:
        seen = set()
        out = []
        for s in samples:
            h = self._hash_text(s.content)
            if h not in seen:
                seen.add(h)
                out.append(s)
        return out


# =========================
# Prompt normalization for overlap matching
# =========================
_ws_re = re.compile(r"\s+")

def norm_prompt(text: str, lowercase: bool = True) -> str:
    if text is None:
        return ""
    s = str(text).strip()
    if lowercase:
        s = s.lower()
    s = _ws_re.sub(" ", s)
    return s


# =========================
# Split utilities
# =========================

def split_logs(
    logs: List[Dict[str, Any]],
    test_ratio: float,
    seed: int,
    do_shuffle: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    logs = list(logs)
    rnd = random.Random(seed)
    if do_shuffle:
        rnd.shuffle(logs)

    if len(logs) == 0:
        return [], []

    n_test = int(len(logs) * test_ratio)
    if test_ratio > 0 and n_test == 0 and len(logs) >= 2:
        n_test = 1

    test_logs = logs[:n_test]
    train_logs = logs[n_test:]
    return train_logs, test_logs


def split_val_test_from_mixed(
    items: List[Dict[str, Any]],
    val_ratio: float,
    seed: int,
    do_shuffle: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a mixed test set into val + test (conversation-level)."""
    items = list(items)
    rng = random.Random(seed)
    if do_shuffle:
        rng.shuffle(items)

    if not items:
        return [], []

    n_val = int(len(items) * val_ratio)
    if val_ratio > 0 and n_val == 0 and len(items) >= 2:
        n_val = 1

    val_items = items[:n_val]
    test_items = items[n_val:]
    return val_items, test_items


# =========================
# Malicious mixing
# =========================

def load_attack_logs(builder: BenignPrefixDatasetBuilder, attack_files: Dict[str, Path]) -> Dict[str, List[Dict[str, Any]]]:
    attacks = {}
    for attack_type, path in attack_files.items():
        items = builder.load_logs(path)
        attacks[attack_type] = items
    return attacks

def index_by_prompt(builder: BenignPrefixDatasetBuilder, logs: List[Dict[str, Any]], attack_type: str) -> Dict[str, List[Dict[str, Any]]]:
    """
    norm_prompt -> list(records). We'll pop() for selection in replace-based attacks.
    """
    idx: Dict[str, List[Dict[str, Any]]] = {}
    for r in logs:
        p = norm_prompt(r.get("prompt", ""), lowercase=builder.normalize_lowercase)
        if not p:
            continue
        rec = dict(r)
        rec["label"] = 1
        rec["attack_type"] = attack_type
        idx.setdefault(p, []).append(rec)
    return idx

def compute_overlaps(builder: BenignPrefixDatasetBuilder, test_logs: List[Dict[str, Any]], attack_indexes: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, int]:
    test_prompts = set()
    for r in test_logs:
        p = norm_prompt(r.get("prompt", ""), lowercase=builder.normalize_lowercase)
        if p:
            test_prompts.add(p)

    overlaps = {}
    for atk, idx in attack_indexes.items():
        overlaps[atk] = sum(1 for p in test_prompts if p in idx and len(idx[p]) > 0)
    return overlaps

def flatten_attack_records(prompt_index: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    out = []
    for lst in prompt_index.values():
        out.extend(lst)
    return out

def mix_malicious_into_test(
    builder: BenignPrefixDatasetBuilder,
    test_benign: List[Dict[str, Any]],
    attack_indexes: Dict[str, Dict[str, List[Dict[str, Any]]]],
    picks: Dict[str, int],
    seed: int,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Mixing policy:
    - rag/tool/correlated: REPLACE by overlapping prompts
    - prompt injection: APPEND directly (no overlap required)
    """
    rng = random.Random(seed)

    # prompt -> list(test indices)
    prompt_to_indices: Dict[str, List[int]] = {}
    for i, rec in enumerate(test_benign):
        p = norm_prompt(rec.get("prompt", ""), lowercase=builder.normalize_lowercase)
        if p:
            prompt_to_indices.setdefault(p, []).append(i)

    replaced_indices = set()
    added_malicious: List[Dict[str, Any]] = []
    report: Dict[str, Any] = {}

    # ---------- 1) Replace-based attacks ----------
    for attack_type in ["rag", "tool", "correlated"]:
        idx = attack_indexes.get(attack_type, {})
        overlap_prompts = [p for p in prompt_to_indices if p in idx and idx[p]]

        want = int(picks.get(attack_type, 0))
        can = len(overlap_prompts)
        take = min(want, can)

        report[attack_type] = {
            "strategy": "replace",
            "overlap_prompts": can,
            "requested": want,
            "used": take,
        }

        rng.shuffle(overlap_prompts)
        for p in overlap_prompts[:take]:
            candidates = [j for j in prompt_to_indices[p] if j not in replaced_indices]
            if not candidates:
                continue
            j = candidates[0]
            replaced_indices.add(j)

            mal = idx[p].pop()
            mal = dict(mal)
            mal["label"] = 1
            mal["attack_type"] = attack_type
            mal["mixed_from_origin_id"] = test_benign[j].get("id")
            mal["mixed_from_prompt"] = test_benign[j].get("prompt")
            added_malicious.append(mal)

    # ---------- 2) Prompt injection: append directly ----------
    prompt_pick = int(picks.get("prompt", 0))
    prompt_idx = attack_indexes.get("prompt", {})
    flat_prompt_attacks = flatten_attack_records(prompt_idx)

    rng.shuffle(flat_prompt_attacks)
    take_prompt = min(prompt_pick, len(flat_prompt_attacks))

    for mal in flat_prompt_attacks[:take_prompt]:
        mal = dict(mal)
        mal["label"] = 1
        mal["attack_type"] = "prompt"
        mal["mixed_from_origin_id"] = None
        mal["mixed_from_prompt"] = None
        added_malicious.append(mal)

    report["prompt"] = {
        "strategy": "append" if take_prompt > 0 else "none",
        "available": len(flat_prompt_attacks),
        "requested": prompt_pick,
        "used": take_prompt,
    }

    # ---------- 3) Build final test ----------
    kept_benign = []
    for i, rec in enumerate(test_benign):
        if i in replaced_indices:
            continue
        r = dict(rec)
        r["label"] = 0
        r.pop("attack_type", None)
        kept_benign.append(r)

    final_test = kept_benign + added_malicious
    rng.shuffle(final_test)

    report["final_counts"] = {
        "benign": sum(int(r.get("label", 0)) == 0 for r in final_test),
        "malicious": sum(int(r.get("label", 0)) == 1 for r in final_test),
        "total": len(final_test),
    }

    return final_test, report


# =========================
# Sample dedup policies
# =========================

def dedup_global_no_leak(
    builder: BenignPrefixDatasetBuilder,
    train: List[PrefixSample],
    test: List[PrefixSample],
) -> Tuple[List[PrefixSample], List[PrefixSample], Dict[str, int]]:
    """
    Ensure NO identical sample content appears in both train and test.
    Policy:
    - Dedup train internally
    - Dedup test internally
    - Remove from test any sample whose hash already appears in train
    """
    train = builder.deduplicate_samples(train)
    test = builder.deduplicate_samples(test)

    train_hash = set(builder._hash_text(s.content) for s in train)
    before = len(test)
    test2 = []
    dropped = 0
    for s in test:
        h = builder._hash_text(s.content)
        if h in train_hash:
            dropped += 1
            continue
        test2.append(s)

    stats = {"test_before": before, "test_after": len(test2), "dropped_from_test": dropped}
    return train, test2, stats


# =========================
# Menu helpers
# =========================

def ask(prompt, cast=str, default=None):
    msg = f"{prompt}"
    if default is not None:
        msg += f" [{default}]"
    msg += ": "
    val = input(msg).strip()
    if not val and default is not None:
        return default
    return cast(val)

def ask_choice(title: str, choices: List[Tuple[str, str]]) -> str:
    print(f"\n{title}")
    for k, desc in choices:
        print(f"  {k}) {desc}")
    while True:
        v = input("Choose: ").strip()
        if any(v == k for k, _ in choices):
            return v
        print("Invalid choice. Try again.")

def ask_yes_no(msg: str, default: str = "y") -> bool:
    default = default.lower()
    while True:
        ans = input(f"{msg} (y/n) [{default}]: ").strip().lower()
        if not ans:
            ans = default
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        print("Please answer y or n.")

def ask_int(msg: str, default: int, min_v: int = 0, max_v: Optional[int] = None) -> int:
    while True:
        s = input(f"{msg} [{default}]: ").strip()
        if not s:
            v = default
        else:
            if not s.isdigit():
                print("Invalid integer.")
                continue
            v = int(s)
        if v < min_v:
            print(f"Must be >= {min_v}")
            continue
        if max_v is not None and v > max_v:
            print(f"Must be <= {max_v}")
            continue
        return v


# =========================
# Main
# =========================

def main():
    print("\n=== Dataset Pipeline (Final Answers + Prefix Samples + Optional Malicious Mix) ===\n")

    mode = ask_choice(
        "Select mode",
        [
            ("1", "Split-only (final answers) -> train_final.jsonl / test_final_benign.jsonl (+ optional mixed test)"),
            ("2", "Split-then-preprocess (prefix samples) -> train.jsonl / test.jsonl (+ optional mixed test)"),
            ("3", "Preprocess-only (one file -> processed.jsonl)"),
        ],
    )

    input_path = Path(ask("Path to input logs (.jsonl or .json)", str, str(DEFAULT_COLLECTED_DIR / "benign_traces.jsonl")))
    out_dir = Path(ask("Output directory", str, str(settings.PROCESSED_DIR)))
    seed = ask("Random seed", int, settings.DATA_SEED)
    do_shuffle = ask_yes_no("Shuffle before split?", "y")

    out_dir.mkdir(parents=True, exist_ok=True)

    builder = BenignPrefixDatasetBuilder(seed=seed)

    logs = builder.load_logs(input_path)
    print(f"\nLoaded {len(logs)} conversations")

    if len(logs) == 0:
        print("No logs found. Exiting.")
        return

    normalize_lowercase = ask_yes_no("Lowercase normalization?", "y")
    builder.normalize_lowercase = normalize_lowercase

    # ===== MODE 3: Preprocess-only =====
    if mode == "3":
        dedup = ask_yes_no("Deduplicate samples?", "y")
        out_path = out_dir / "processed.jsonl"

        all_samples: List[PrefixSample] = []
        for c in logs:
            all_samples.extend(builder.build_prefixes(c))

        if dedup:
            all_samples = builder.deduplicate_samples(all_samples)

        with out_path.open("w", encoding="utf-8") as f:
            for s in all_samples:
                f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")

        print("\n=== DONE (Preprocess-only) ===")
        print(f"Processed samples: {len(all_samples)} -> {out_path}")
        return

    # Common split params
    test_ratio = float(ask("Test ratio (conversation-level split)", float, 0.2))

    # ===== Split benign final answers =====
    train_logs, test_logs_benign = split_logs(logs, test_ratio=test_ratio, seed=seed, do_shuffle=do_shuffle)

    # enforce labels for split (benign baseline)
    for r in train_logs:
        r["label"] = 0
        r.pop("attack_type", None)
    for r in test_logs_benign:
        r["label"] = 0
        r.pop("attack_type", None)

    # always export split finals
    train_final_path = out_dir / "train_final.jsonl"
    test_final_benign_path = out_dir / "test_final_benign.jsonl"
    builder.write_jsonl(train_logs, train_final_path)
    builder.write_jsonl(test_logs_benign, test_final_benign_path)

    print("\n--- Split Result ---")
    print(f"Train final answers: {len(train_logs)} -> {train_final_path}")
    print(f"Test  final answers (benign): {len(test_logs_benign)} -> {test_final_benign_path}")

    # ===== Optional: mix malicious into test finals =====
    do_mix = ask_yes_no("\nMix malicious logs into TEST? (rag/tool/correlated replace, prompt append)", "y")

    test_logs_final = test_logs_benign
    mix_report = None

    # these are for the new "split mixed into val/test" step
    val_logs_final: Optional[List[Dict[str, Any]]] = None
    test_logs_final_split: Optional[List[Dict[str, Any]]] = None

    if do_mix:
        attack_dir = Path(ask("Attack logs directory", str, str(DEFAULT_COLLECTED_DIR)))
        attack_files = {
            "prompt": Path(ask("Prompt-attack traces file", str, str(attack_dir / DEFAULT_ATTACK_FILES["prompt"].name))),
            "rag": Path(ask("RAG-attack traces file", str, str(attack_dir / DEFAULT_ATTACK_FILES["rag"].name))),
            "tool": Path(ask("Tool-attack traces file", str, str(attack_dir / DEFAULT_ATTACK_FILES["tool"].name))),
            "correlated": Path(ask("Correlated-attack traces file", str, str(attack_dir / DEFAULT_ATTACK_FILES["correlated"].name))),
        }

        attacks = load_attack_logs(builder, attack_files)
        attack_indexes = {atk: index_by_prompt(builder, items, atk) for atk, items in attacks.items()}

        overlaps = compute_overlaps(builder, test_logs_benign, attack_indexes)

        prompt_available = len(flatten_attack_records(attack_indexes.get("prompt", {})))

        print("\n--- Overlap report (test prompts ∩ attack prompts) ---")
        for atk in ["rag", "tool", "correlated"]:
            print(f"{atk:10s}: {overlaps.get(atk, 0)} overlapping prompts (replace-based)")
        print(f"{'prompt':10s}: overlap={overlaps.get('prompt', 0)} (ignored), available_prompt_attacks={prompt_available} (append-based)")

        print("\nChoose how many attacks to mix into the TEST set:")

        picks = {}
        for atk in ["rag", "tool", "correlated"]:
            picks[atk] = ask_int(f"Replace with {atk} attacks", 0, min_v=0, max_v=overlaps.get(atk, 0))

        picks["prompt"] = ask_int("Append prompt-injection attacks (no overlap needed)", 0, min_v=0, max_v=prompt_available)

        test_logs_final, mix_report = mix_malicious_into_test(
            builder=builder,
            test_benign=test_logs_benign,
            attack_indexes=attack_indexes,
            picks=picks,
            seed=seed,
        )

        test_final_mixed_path = out_dir / "test_final_mixed.jsonl"
        builder.write_jsonl(test_logs_final, test_final_mixed_path)

        report_path = out_dir / "mix_report.json"
        report_path.write_text(json.dumps(mix_report, ensure_ascii=False, indent=2), encoding="utf-8")

        print("\n=== MIX DONE ===")
        print(f"Mixed test final answers: {len(test_logs_final)} -> {test_final_mixed_path}")
        print(f"Mix report: {report_path}")

        # ===== NEW: Split mixed test into val + test BEFORE prefix/preprocess =====
        do_split_mixed = ask_yes_no("\nSplit test_final_mixed.jsonl into val_final_mixed.jsonl + test_final_mixed.json before preprocessing?", "y")
        if do_split_mixed:
            val_ratio = float(ask("VAL ratio from mixed set", float, 0.5))
            split_shuffle = ask_yes_no("Shuffle mixed set before val/test split?", "y")

            val_logs_final, test_logs_final_split = split_val_test_from_mixed(
                items=test_logs_final,
                val_ratio=val_ratio,
                seed=seed,
                do_shuffle=split_shuffle,
            )

            val_path = out_dir / "val_final_mixed.jsonl"
            test_json_path = out_dir / "test_final_mixed.json"  # JSON array, as requested

            builder.write_jsonl(val_logs_final, val_path)
            builder.write_json_array(test_logs_final_split, test_json_path)

            vb = sum(int(x.get("label", 0)) == 0 for x in val_logs_final)
            vm = sum(int(x.get("label", 0)) == 1 for x in val_logs_final)
            tb = sum(int(x.get("label", 0)) == 0 for x in test_logs_final_split)
            tm = sum(int(x.get("label", 0)) == 1 for x in test_logs_final_split)

            print("\n=== SPLIT MIXED DONE ===")
            print(f"VAL : {len(val_logs_final)} -> {val_path} (benign={vb}, malicious={vm})")
            print(f"TEST: {len(test_logs_final_split)} -> {test_json_path} (benign={tb}, malicious={tm})")

    # ===== MODE 1: Split-only ends here =====
    if mode == "1":
        print("\n=== DONE (Split-only) ===")
        return

    # ===== MODE 2: Split-then-preprocess =====
    if mode == "2":
        dedup = ask_yes_no("Deduplicate samples?", "y")

        dedup_mode = ask_choice(
            "Dedup scope (to avoid leakage)",
            [
                ("1", "within_split (dedup train and test separately)"),
                ("2", "global_no_leak (drop any test sample identical to a train sample)"),
            ],
        )

        # Build train prefixes (always from train_logs)
        train_samples: List[PrefixSample] = []
        for c in train_logs:
            train_samples.extend(builder.build_prefixes(c))

        # Decide what to preprocess for evaluation side:
        # - If user split mixed -> preprocess val/test separately (optional)
        # - else -> preprocess test_logs_final as before
        preprocess_val_test_separately = False
        if val_logs_final is not None and test_logs_final_split is not None:
            preprocess_val_test_separately = ask_yes_no("Preprocess VAL and TEST separately (val.jsonl + test.jsonl)?", "y")

        if preprocess_val_test_separately:
            val_samples: List[PrefixSample] = []
            test_samples: List[PrefixSample] = []

            for c in val_logs_final:
                val_samples.extend(builder.build_prefixes(c))
            for c in test_logs_final_split:
                test_samples.extend(builder.build_prefixes(c))

            # Dedup policies (apply separately for val/test; no-leak against train if chosen)
            if dedup:
                if dedup_mode == "1":
                    train_samples = builder.deduplicate_samples(train_samples)
                    val_samples = builder.deduplicate_samples(val_samples)
                    test_samples = builder.deduplicate_samples(test_samples)
                else:
                    # no-leak vs train for val and test separately
                    train_samples, val_samples, stats_val = dedup_global_no_leak(builder, train_samples, val_samples)
                    # recompute train hash again (train unchanged) and drop leaks in test
                    train_samples, test_samples, stats_test = dedup_global_no_leak(builder, train_samples, test_samples)
                    print("\n--- global_no_leak stats (VAL) ---")
                    print(json.dumps(stats_val, ensure_ascii=False, indent=2))
                    print("\n--- global_no_leak stats (TEST) ---")
                    print(json.dumps(stats_test, ensure_ascii=False, indent=2))

            # Write outputs
            train_path = out_dir / "train.jsonl"
            val_path = out_dir / "val.jsonl"
            test_path = out_dir / "test.jsonl"

            with train_path.open("w", encoding="utf-8") as f:
                for s in train_samples:
                    f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")
            with val_path.open("w", encoding="utf-8") as f:
                for s in val_samples:
                    f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")
            with test_path.open("w", encoding="utf-8") as f:
                for s in test_samples:
                    f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")

            print("\n=== DONE (Split-then-preprocess, val/test separate) ===")
            print(f"Train samples: {len(train_samples)} -> {train_path}")
            print(f"VAL   samples: {len(val_samples)} -> {val_path}")
            print(f"TEST  samples: {len(test_samples)} -> {test_path}")
            return

        # -------- Default behavior (as before): preprocess single test set --------
        test_samples: List[PrefixSample] = []
        for c in test_logs_final:
            test_samples.extend(builder.build_prefixes(c))

        if dedup:
            if dedup_mode == "1":
                train_samples = builder.deduplicate_samples(train_samples)
                test_samples = builder.deduplicate_samples(test_samples)
            else:
                train_samples, test_samples, stats = dedup_global_no_leak(builder, train_samples, test_samples)
                print("\n--- global_no_leak stats ---")
                print(json.dumps(stats, ensure_ascii=False, indent=2))

        train_path = out_dir / "train.jsonl"
        test_path = out_dir / "test.jsonl"

        with train_path.open("w", encoding="utf-8") as f:
            for s in train_samples:
                f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")

        with test_path.open("w", encoding="utf-8") as f:
            for s in test_samples:
                f.write(json.dumps(s.__dict__, ensure_ascii=False) + "\n")

        print("\n=== DONE (Split-then-preprocess) ===")
        print(f"Train samples: {len(train_samples)} -> {train_path}")
        print(f"Test  samples: {len(test_samples)} -> {test_path}")
        return


if __name__ == "__main__":
    main()
