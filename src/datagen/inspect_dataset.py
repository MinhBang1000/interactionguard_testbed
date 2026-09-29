#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
datagen/inspect_dataset.py

Read-only diagnostics over collected traces: dedup ratio, trace-pattern
distribution, tool usage, prompt overlap between attack types. Same
analysis as the original `investigate_data.py` — only the default log
directory now comes from `settings.TRACES_DIR`.
"""

import json
from pathlib import Path
from collections import defaultdict, Counter
import hashlib

from src import settings


# =====================================
# Helper functions
# =====================================

def load_jsonl(path):
    """Load JSONL file"""
    data = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def hash_text(text):
    """Hash text for dedup check"""
    return hashlib.md5(str(text).lower().strip().encode()).hexdigest()


def extract_trace_pattern(trace):
    """Extract trace pattern for analysis"""
    return tuple(trace) if isinstance(trace, list) else None


def analyze_file(filepath, label_name):
    """Phân tích chi tiết một file"""
    print(f"\n{'=' * 80}")
    print(f"FILE: {filepath.name}")
    print(f"Label: {label_name}")
    print(f"{'=' * 80}")

    traces = load_jsonl(filepath)
    n_traces = len(traces)

    print(f"\n📊 TỔNG QUAN:")
    print(f"  - Số traces: {n_traces:,}")
    print(f"  - File size: {filepath.stat().st_size / 1024 / 1024:.2f} MB")

    # ===== Phân tích prompts =====
    prompts = [t.get("prompt", "") for t in traces]
    prompt_hashes = [hash_text(p) for p in prompts]
    unique_prompts = len(set(prompt_hashes))

    print(f"\n🎯 PROMPTS:")
    print(f"  - Unique prompts: {unique_prompts:,}")
    print(f"  - Duplicate ratio: {(1 - unique_prompts / n_traces) * 100:.1f}%")

    # Length stats
    prompt_lens = [len(str(p)) for p in prompts]
    print(f"  - Avg length: {sum(prompt_lens) / len(prompt_lens):.0f} chars")
    print(f"  - Min/Max: {min(prompt_lens)} / {max(prompt_lens)} chars")

    # ===== Phân tích trace patterns =====
    trace_patterns = [extract_trace_pattern(t.get("trace", [])) for t in traces]
    pattern_counter = Counter(trace_patterns)

    print(f"\n🔀 TRACE PATTERNS:")
    print(f"  - Unique patterns: {len(pattern_counter)}")
    print(f"  - Top 5 patterns:")
    for pattern, count in pattern_counter.most_common(5):
        print(f"      {list(pattern) if pattern else 'None'}: {count:,} ({count / n_traces * 100:.1f}%)")

    # Trace length
    trace_lens = [len(t.get("trace", [])) for t in traces]
    print(f"  - Avg trace length: {sum(trace_lens) / len(trace_lens):.1f} steps")
    print(f"  - Min/Max: {min(trace_lens)} / {max(trace_lens)} steps")

    # ===== Phân tích tools =====
    all_tools = []
    for t in traces:
        all_tools.extend(t.get("tools_used", []))

    tool_counter = Counter(all_tools)
    print(f"\n🛠️  TOOLS USED:")
    print(f"  - Total tool calls: {len(all_tools):,}")
    print(f"  - Unique tools: {len(tool_counter)}")
    if tool_counter:
        for tool, count in tool_counter.most_common():
            print(f"      {tool}: {count:,}")

    # ===== Phân tích reasoning steps =====
    reasoning_lens = [len(t.get("reasoning_steps", [])) for t in traces]
    print(f"\n💭 REASONING STEPS:")
    print(f"  - Avg steps: {sum(reasoning_lens) / len(reasoning_lens):.1f}")
    print(f"  - Min/Max: {min(reasoning_lens)} / {max(reasoning_lens)}")

    # ===== Phân tích sources (chỉ cho benign) =====
    if label_name == "benign":
        sources = [t.get("source", "unknown") for t in traces]
        source_counter = Counter(sources)
        print(f"\n📁 SOURCES (benign only):")
        for src, count in source_counter.most_common():
            print(f"      {src}: {count:,} ({count / n_traces * 100:.1f}%)")

    # ===== Phân tích attack types (chỉ cho attacks) =====
    if label_name != "benign":
        attack_types = [t.get("attack_type", "unknown") for t in traces]
        attack_counter = Counter(attack_types)
        print(f"\n🎭 ATTACK TYPES:")
        for atype, count in attack_counter.most_common():
            print(f"      {atype}: {count:,} ({count / n_traces * 100:.1f}%)")

    # ===== Errors =====
    n_errors = sum(1 for t in traces if t.get("error") is not None)
    print(f"\n❌ ERRORS:")
    print(f"  - Traces with errors: {n_errors} ({n_errors / n_traces * 100:.1f}%)")

    # ===== Kiểm tra final_answer =====
    has_answer = sum(1 for t in traces if t.get("final_answer"))
    print(f"\n✅ FINAL ANSWERS:")
    print(f"  - Traces with final_answer: {has_answer} ({has_answer / n_traces * 100:.1f}%)")

    return {
        "file": filepath.name,
        "label": label_name,
        "n_traces": n_traces,
        "unique_prompts": unique_prompts,
        "unique_trace_patterns": len(pattern_counter),
        "avg_trace_len": sum(trace_lens) / len(trace_lens),
        "tools": dict(tool_counter),
        "traces": traces,
        "prompt_hashes": prompt_hashes,
    }


def analyze_prompt_overlap(datasets):
    """Phân tích prompt overlap giữa các tập (đặc biệt RAG vs Correlated)"""
    print(f"\n{'=' * 80}")
    print("PHÂN TÍCH PROMPT OVERLAP")
    print(f"{'=' * 80}")

    # Build prompt hash -> datasets mapping
    prompt_to_datasets = defaultdict(set)
    prompt_to_attacks = defaultdict(list)

    for ds in datasets:
        label = ds["label"]
        for i, ph in enumerate(ds["prompt_hashes"]):
            prompt_to_datasets[ph].add(label)
            if label != "benign":
                prompt_to_attacks[ph].append({
                    "label": label,
                    "trace": ds["traces"][i]
                })

    # Find overlaps
    print(f"\n🔍 PROMPT OVERLAPS:")

    # RAG vs Correlated
    rag_prompts = set()
    corr_prompts = set()

    for ds in datasets:
        if ds["label"] == "rag_poison":
            rag_prompts.update(ds["prompt_hashes"])
        elif ds["label"] == "correlated":
            corr_prompts.update(ds["prompt_hashes"])

    rag_corr_overlap = rag_prompts & corr_prompts
    print(f"\n  RAG vs Correlated overlap:")
    print(f"    - RAG unique prompts: {len(rag_prompts):,}")
    print(f"    - Correlated unique prompts: {len(corr_prompts):,}")
    print(f"    - Shared prompts: {len(rag_corr_overlap):,}")

    if rag_corr_overlap:
        # Analyze a few shared prompts
        print(f"\n  📝 Sample shared prompt analysis:")
        for ph in list(rag_corr_overlap)[:3]:  # Show 3 examples
            attacks = prompt_to_attacks[ph]
            rag_traces = [a for a in attacks if a["label"] == "rag_poison"]
            corr_traces = [a for a in attacks if a["label"] == "correlated"]

            if rag_traces and corr_traces:
                rag_trace = rag_traces[0]["trace"]
                corr_trace = corr_traces[0]["trace"]

                print(f"\n    Prompt hash: {ph[:16]}...")
                print(f"      RAG trace: {rag_trace.get('trace', [])}")
                print(f"      RAG trace len: {len(rag_trace.get('trace', []))}")
                print(f"      Correlated trace: {corr_trace.get('trace', [])}")
                print(f"      Correlated trace len: {len(corr_trace.get('trace', []))}")

    # Cross-attack overlaps (tất cả các cặp)
    attack_labels = [ds["label"] for ds in datasets if ds["label"] != "benign"]
    print(f"\n  Cross-attack prompt overlaps:")

    for i, label1 in enumerate(attack_labels):
        for label2 in attack_labels[i + 1:]:
            prompts1 = set()
            prompts2 = set()

            for ds in datasets:
                if ds["label"] == label1:
                    prompts1.update(ds["prompt_hashes"])
                if ds["label"] == label2:
                    prompts2.update(ds["prompt_hashes"])

            overlap = prompts1 & prompts2
            if overlap:
                print(f"    {label1} ∩ {label2}: {len(overlap):,}")

    # Benign vs Attacks overlap
    benign_prompts = set()
    attack_prompts = set()

    for ds in datasets:
        if ds["label"] == "benign":
            benign_prompts.update(ds["prompt_hashes"])
        else:
            attack_prompts.update(ds["prompt_hashes"])

    benign_attack_overlap = benign_prompts & attack_prompts
    print(f"\n  Benign vs All Attacks:")
    print(f"    - Benign unique prompts: {len(benign_prompts):,}")
    print(f"    - Attack unique prompts: {len(attack_prompts):,}")
    print(f"    - Overlap: {len(benign_attack_overlap):,}")


def main(log_dir: Path = None):
    """Main analysis"""
    print("\n" + "=" * 80)
    print("ĐIỀU TRA DỮ LIỆU COLLECTED LOGS")
    print("=" * 80)

    log_dir = Path(log_dir) if log_dir else settings.TRACES_DIR

    files = [
        (log_dir / "benign_traces.jsonl", "benign"),
        (log_dir / "prompt_injection_traces.jsonl", "prompt_injection"),
        (log_dir / "rag_poison_traces.jsonl", "rag_poison"),
        (log_dir / "tool_injection_traces.jsonl", "tool_injection"),
        (log_dir / "correlated_injection_traces.jsonl", "correlated"),
    ]

    datasets = []
    for fpath, label in files:
        if fpath.exists():
            ds = analyze_file(fpath, label)
            datasets.append(ds)
        else:
            print(f"\n⚠️  File not found: {fpath}")

    # Analyze overlaps
    analyze_prompt_overlap(datasets)

    # ===== TỔNG KẾT =====
    print(f"\n{'=' * 80}")
    print("TỔNG KẾT DỮ LIỆU")
    print(f"{'=' * 80}")

    total_benign = sum(ds["n_traces"] for ds in datasets if ds["label"] == "benign")
    total_attacks = sum(ds["n_traces"] for ds in datasets if ds["label"] != "benign")

    print(f"\n📈 NUMBERS:")
    print(f"  - Total benign: {total_benign:,}")
    print(f"  - Total attacks: {total_attacks:,}")
    print(f"  - Total: {total_benign + total_attacks:,}")
    if (total_benign + total_attacks) > 0:
        print(f"  - Benign ratio: {total_benign / (total_benign + total_attacks) * 100:.1f}%")

    print(f"\n📊 BY ATTACK TYPE:")
    for ds in datasets:
        if ds["label"] != "benign" and total_attacks > 0:
            ratio = ds["n_traces"] / total_attacks * 100
            print(f"  - {ds['label']}: {ds['n_traces']:,} ({ratio:.1f}% of attacks)")

    print(f"\n{'=' * 80}")
    print("HOÀN TẤT ĐIỀU TRA")
    print(f"{'=' * 80}\n")


if __name__ == "__main__":
    main()
