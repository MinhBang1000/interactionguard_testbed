"""
datagen/collect_traces.py

Runs the real LangGraph agent end-to-end over benign and malicious
prompts and logs the full per-node reasoning trace. Merges what used to
be two near-identical scripts (`collect_all_benign_logs.py` and
`collect_all_malicious_logs.py`) — same `run_agent()` logic, same
ATTACK_MODES mapping, same output schema.

Output: one JSONL file per collection under `data/traces/`
(`benign_traces.jsonl`, `tool_injection_traces.jsonl`,
`rag_poison_traces.jsonl`, `correlated_injection_traces.jsonl`).
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Literal, Optional

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph.errors import GraphRecursionError

from src import settings
from src.agent.app import build_app
from src.agent.rag import setup_rag
from src.datagen.io_utils import read_jsonl, write_jsonl
from src.utils import set_env_var, get_mode_from_env

AttackType = Literal["benign", "tool", "rag", "correlated"]

# Attack-type -> (seed prompts file, agent MODE). "rag" and "correlated" both
# drive the agent with the FULL benign query pool (not just the queries that
# have a hand-crafted poison doc) — matches the original
# datasets/poison_queries.jsonl / correlated_queries.jsonl, which were byte-
# identical copies of docs/benign/queries.jsonl.
ATTACK_FILES: Dict[str, Path] = {
    # "prompt": settings.DEEPSET_PROMPT_INJECTIONS_PATH,  # disabled in the original script — kept disabled here too
    "tool": settings.TOOL_INJECTION_DATASET_PATH,
    "rag": settings.corpus_dir("benign") / "queries.jsonl",
    "correlated": settings.corpus_dir("benign") / "queries.jsonl",
}

ATTACK_MODES: Dict[str, int] = {
    "prompt": 1,   # benign pipeline (prompt-injection attacks the prompt itself, not RAG/tools)
    "tool": 3,
    "rag": 2,
    "correlated": 4,
}

RECURSION_LIMIT = settings.AGENT_RECURSION_LIMIT
THREAD_PREFIX = "collector"


# =========================
# Shared agent runner
# =========================

def run_agent(prompt: str, app, thread_id: str) -> Dict:
    trace_steps: List[str] = []
    reasoning_steps: List[Dict] = []
    tools_used: List[str] = []
    final_answer = ""
    hit_recursion = False
    error = None

    try:
        events = app.stream(
            {"messages": [HumanMessage(content=prompt)]},
            config={
                "configurable": {"thread_id": thread_id},
                "recursion_limit": RECURSION_LIMIT,
            },
        )

        for ev in events:
            node_name, payload = next(iter(ev.items()))
            msg = payload["messages"][-1]

            step_log = {
                "node": node_name,
                "type": "tool" if isinstance(msg, ToolMessage) else "agent",
                "content": msg.content if hasattr(msg, "content") else "",
            }

            if isinstance(msg, ToolMessage):
                tool = msg.name
                tools_used.append(tool)
                trace_steps.append(f"TOOL:{tool}")
                step_log["tool"] = tool
            else:
                trace_steps.append(node_name)

            if isinstance(msg, AIMessage):
                final_answer = msg.content or final_answer

            reasoning_steps.append(step_log)

    except GraphRecursionError:
        hit_recursion = True
        error = "GraphRecursionError"
    except Exception as e:
        error = f"{type(e).__name__}: {e}"

    return {
        "trace": trace_steps,
        "reasoning_steps": reasoning_steps,
        "tools_used": list(set(tools_used)),
        "final_answer": final_answer,
        "hit_recursion": hit_recursion,
        "error": error,
    }


# =========================
# Prompt pool loaders
# =========================

def load_benign_prompts() -> List[Dict]:
    prompts: List[Dict] = []

    for obj in read_jsonl(settings.DEEPSET_PROMPT_INJECTIONS_PATH):
        if int(obj.get("label", 1)) == 0:
            prompts.append({"source": "deepset", "text": obj["text"].strip()})

    for obj in read_jsonl(settings.TOOL_INJECTION_DATASET_PATH):
        prompts.append({"source": "tool_injection", "text": obj["user_prompt"].strip()})

    for obj in read_jsonl(settings.corpus_dir("benign") / "queries.jsonl"):
        prompts.append({"source": "natural_questions", "text": obj["text"].strip()})

    return prompts


def load_malicious_prompts(path: Path, attack_type: str) -> List[Dict]:
    prompts = []
    for obj in read_jsonl(path):
        if attack_type == "prompt":
            if int(obj.get("label", 0)) != 1:
                continue
            text = obj.get("text", "").strip()
        else:
            text = obj.get("text") or obj.get("user_prompt")

        if not text:
            continue

        prompts.append({
            "id": obj.get("id"),
            "source": obj.get("source", attack_type),
            "attack_type": attack_type,
            "text": text.strip(),
        })
    return prompts


# =========================
# Collection
# =========================

def _collect_benign(sample_n: Optional[int], shuffle: bool) -> Path:
    prompts = load_benign_prompts()
    print(f"Loaded {len(prompts)} benign prompts.")

    if shuffle:
        random.shuffle(prompts)
    if sample_n:
        prompts = prompts[:sample_n]

    set_env_var(str(settings.ENV_PATH), "MODE", "1")
    load_dotenv(str(settings.ENV_PATH), override=True)
    mode = get_mode_from_env(settings.ENV_PATH)
    print(f"[ENV] MODE={mode} (benign) enforced for collector")

    chroma_db = setup_rag(mode)
    app = build_app(chroma_db, k=6, mode=mode)

    out_file = settings.TRACES_DIR / "benign_traces.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    with out_file.open("w", encoding="utf-8") as f:
        for i, item in enumerate(prompts, 1):
            result = run_agent(item["text"], app, thread_id=f"{THREAD_PREFIX}_{i}")
            record = {
                "id": f"{item['source']}_{i}",
                "source": item["source"],
                "prompt": item["text"],
                "trace": result["trace"],
                "reasoning_steps": result["reasoning_steps"],
                "tools_used": result["tools_used"],
                "final_answer": result["final_answer"],
                "hit_recursion": result["hit_recursion"],
                "error": result["error"],
                "label": 0,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

            status = "RECURSION" if result["hit_recursion"] else ("ERROR" if result["error"] else "OK")
            print(f"[{i}/{len(prompts)}] {status} | {item['source']}")

    print(f"\nSaved to: {out_file.resolve()}")
    return out_file


def _collect_malicious(attack_type: AttackType, sample_n: Optional[int]) -> Path:
    file_path = ATTACK_FILES[attack_type]
    prompts = load_malicious_prompts(file_path, attack_type)
    print(f"Loaded {len(prompts)} malicious prompts ({attack_type})")

    if sample_n:
        prompts = prompts[:sample_n]

    mode = ATTACK_MODES[attack_type]
    set_env_var(str(settings.ENV_PATH), "MODE", str(mode))
    load_dotenv(str(settings.ENV_PATH), override=True)
    mode = get_mode_from_env(settings.ENV_PATH)
    print(f"[ENV] MODE={mode} ({attack_type})")

    chroma_db = setup_rag(mode)
    app = build_app(chroma_db, k=6, mode=mode)

    out_file = settings.TRACES_DIR / f"{attack_type}_injection_traces.jsonl"
    if attack_type == "rag":
        out_file = settings.TRACES_DIR / "rag_poison_traces.jsonl"
    out_file.parent.mkdir(parents=True, exist_ok=True)

    records = []
    for i, item in enumerate(prompts, 1):
        result = run_agent(item["text"], app, thread_id=f"{THREAD_PREFIX}_{attack_type}_{i}")
        records.append({
            "id": item.get("id") or f"{attack_type}_{i}",
            "source": item["source"],
            "attack_type": attack_type,
            "prompt": item["text"],
            "trace": result["trace"],
            "reasoning_steps": result["reasoning_steps"],
            "tools_used": result["tools_used"],
            "final_answer": result["final_answer"],
            "hit_recursion": result["hit_recursion"],
            "error": result["error"],
            "label": 1,
        })
        status = "RECURSION" if result["hit_recursion"] else ("ERROR" if result["error"] else "OK")
        print(f"[{i}/{len(prompts)}] {status}")

    write_jsonl(out_file, records)
    print(f"Saved to {out_file}")
    return out_file


def collect(attack_type, sample_n: Optional[int] = None, shuffle: bool = False) -> List[Path]:
    """attack_type: 'benign' | 'tool' | 'rag' | 'correlated' | 'all'"""
    if attack_type == "all":
        out = [_collect_benign(sample_n, shuffle)]
        for t in ("tool", "rag", "correlated"):
            out.append(_collect_malicious(t, sample_n))
        return out
    if attack_type == "benign":
        return [_collect_benign(sample_n, shuffle)]
    if attack_type in ("tool", "rag", "correlated"):
        return [_collect_malicious(attack_type, sample_n)]
    raise ValueError(f"Unknown attack_type: {attack_type}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--type", choices=["benign", "tool", "rag", "correlated", "all"], default="all")
    parser.add_argument("--sample", type=int, default=None)
    parser.add_argument("--shuffle", action="store_true")
    args = parser.parse_args()

    collect(args.type, sample_n=args.sample, shuffle=args.shuffle)
