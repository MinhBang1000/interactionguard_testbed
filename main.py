#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
main.py

Single entrypoint for the InteractionGuard data-generation testbed.
Presents a menu covering every stage of the pipeline:

  raw NQ dump -> benign corpus/RAG -> poisoned corpora (AS / CORR) ->
  tool-injection dataset -> prompt-pool merges -> agent trace collection ->
  prefix train/val/test dataset -> dataset inspection

Run with:  python main.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src import settings
from src.agent.rag import setup_rag
from src.datagen import (
    reduce_corpus,
    poison_as,
    poison_correlated,
    tool_injection_dataset,
    merge_prompts,
    collect_traces,
    build_prefix_dataset,
    inspect_dataset,
)


def _pause():
    input("\nPress Enter to return to the menu...")


def step_reduce_corpus():
    reduce_corpus.reduce_corpus()


def step_build_index(mode: int):
    setup_rag(mode=mode)


def step_poison_as():
    poison_as.generate()


def step_poison_correlated():
    poison_correlated.generate()


def step_tool_injection_dataset():
    n = tool_injection_dataset.ask_num_samples()
    tool_injection_dataset.generate(n)


def step_merge_prompts(kind: str):
    merge_prompts.merge(kind)


def step_collect_traces():
    print("\nCollect traces for which attack type?")
    print("  1) benign")
    print("  2) tool")
    print("  3) rag")
    print("  4) correlated")
    print("  5) all")
    choice = input("Choose [5]: ").strip() or "5"
    mapping = {"1": "benign", "2": "tool", "3": "rag", "4": "correlated", "5": "all"}
    attack_type = mapping.get(choice, "all")

    sample_raw = input("Sample how many prompts? (blank = all): ").strip()
    sample_n = int(sample_raw) if sample_raw else None
    shuffle = input("Shuffle prompts? (y/n) [n]: ").strip().lower() in ("y", "yes")

    collect_traces.collect(attack_type, sample_n=sample_n, shuffle=shuffle)


def step_build_prefix_dataset():
    build_prefix_dataset.main()


def step_inspect_dataset():
    inspect_dataset.main()


def step_full_pipeline():
    print("\n=== FULL PIPELINE: 1 -> 11 (using env/.env defaults) ===")
    step_reduce_corpus()
    step_build_index(1)
    step_poison_as()
    step_poison_correlated()
    step_build_index(2)
    step_build_index(4)
    step_tool_injection_dataset()
    step_merge_prompts("benign")
    step_merge_prompts("malicious")
    collect_traces.collect("all")
    step_build_prefix_dataset()
    print("\n=== FULL PIPELINE DONE ===")


MENU = [
    ("1", "Reduce raw NQ dataset -> benign corpus + queries", step_reduce_corpus),
    ("2", "Build/Load benign RAG index (mode=1)", lambda: step_build_index(1)),
    ("3", "Generate poisoned_as corpus (AS attack)", step_poison_as),
    ("4", "Generate correlated_injection corpus (CORR attack)", step_poison_correlated),
    ("5", "Build/Load poisoned_as / correlated_injection RAG index", lambda: (step_build_index(2), step_build_index(4))),
    ("6", "Generate tool_injection dataset", step_tool_injection_dataset),
    ("7", "Merge benign prompt pool", lambda: step_merge_prompts("benign")),
    ("8", "Merge malicious prompt pool", lambda: step_merge_prompts("malicious")),
    ("9", "Collect agent traces (benign / tool / rag / correlated / all)", step_collect_traces),
    ("10", "Build prefix train/val/test dataset", step_build_prefix_dataset),
    ("11", "Inspect collected data (stats)", step_inspect_dataset),
    ("12", "Run FULL pipeline (1 -> 10 end-to-end)", step_full_pipeline),
]


def print_menu():
    print("\n=== InteractionGuard Data-Gen Testbed ===")
    for key, label, _ in MENU:
        print(f" {key:>2}) {label}")
    print("  0) Exit")


def main():
    settings.ensure_data_dirs()

    if not settings.OPENAI_API_KEY:
        print("[WARN] OPENAI_API_KEY is not set in .env — generation steps that call an LLM will fail.")

    actions = {key: fn for key, _, fn in MENU}

    while True:
        print_menu()
        choice = input("Choose: ").strip()

        if choice == "0":
            print("Bye.")
            return

        fn = actions.get(choice)
        if fn is None:
            print("Invalid choice.")
            continue

        try:
            fn()
        except Exception as e:
            print(f"[ERROR] {type(e).__name__}: {e}")

        _pause()


if __name__ == "__main__":
    main()
