"""
settings.py

Central configuration for the InteractionGuard data-generation testbed.

Every path and tunable constant used across `src/agent/` and `src/datagen/`
is defined here, with defaults equal to the values that were previously
hardcoded across individual scripts. Override any of them via a `.env`
file at the project root (see `.env.example`) or real environment
variables — the generation *behavior* (seeds, templates, sample sizes)
stays identical to the original scripts unless you explicitly change it.
"""

from pathlib import Path
import os

from dotenv import load_dotenv

# --------------------------------------------------------------------
# Project root + .env loading
# --------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --------------------------------------------------------------------
# Data directories (all overridable, default = PROJECT_ROOT/data/<name>)
# --------------------------------------------------------------------

DATA_ROOT = Path(os.getenv("DATA_ROOT", PROJECT_ROOT / "data"))

RAW_DIR = Path(os.getenv("RAW_DIR", DATA_ROOT / "raw"))
CORPORA_DIR = Path(os.getenv("CORPORA_DIR", DATA_ROOT / "corpora"))
VECTORSTORE_DIR = Path(os.getenv("VECTORSTORE_DIR", DATA_ROOT / "vectorstore"))
SEED_DIR = Path(os.getenv("SEED_DIR", DATA_ROOT / "seed"))
PROMPTS_DIR = Path(os.getenv("PROMPTS_DIR", DATA_ROOT / "prompts"))
TRACES_DIR = Path(os.getenv("TRACES_DIR", DATA_ROOT / "traces"))
PROCESSED_DIR = Path(os.getenv("PROCESSED_DIR", DATA_ROOT / "processed"))

UPLOADS_DIR = Path(os.getenv("UPLOADS_DIR", PROJECT_ROOT / "uploads"))

# Raw external BEIR "nq" dump (corpus.jsonl / queries.jsonl / qrels/test.tsv).
# Not shipped in the repo (1.5GB+) — download separately, see README.
RAW_NQ_DIR = Path(os.getenv("RAW_NQ_DIR", RAW_DIR / "nq"))

# --------------------------------------------------------------------
# Per-mode corpus / vectorstore paths
# --------------------------------------------------------------------

MODE_MAP = {1: "benign", 2: "poisoned_as", 3: "tool_injection", 4: "correlated_injection"}


def corpus_dir(mode_name: str) -> Path:
    return CORPORA_DIR / mode_name


def vectorstore_dir(mode_name: str) -> Path:
    return VECTORSTORE_DIR / mode_name


# --------------------------------------------------------------------
# Seed / runtime datasets
# --------------------------------------------------------------------

DEEPSET_PROMPT_INJECTIONS_PATH = Path(
    os.getenv("DEEPSET_PROMPT_INJECTIONS_PATH", SEED_DIR / "deepset_prompt_injections_all.jsonl")
)
TOOL_INJECTION_DATASET_PATH = Path(
    os.getenv("TOOL_INJECTION_DATASET_PATH", SEED_DIR / "tool_injection.jsonl")
)

BENIGN_PROMPT_POOL_PATH = Path(os.getenv("BENIGN_PROMPT_POOL_PATH", PROMPTS_DIR / "benign_user_prompts.jsonl"))
MALICIOUS_PROMPT_POOL_PATH = Path(
    os.getenv("MALICIOUS_PROMPT_POOL_PATH", PROMPTS_DIR / "malicious_user_prompts.jsonl")
)

# --------------------------------------------------------------------
# Secrets / credentials (never commit these — see .gitignore)
# --------------------------------------------------------------------

ENV_PATH = Path(os.getenv("ENV_PATH", PROJECT_ROOT / ".env"))
CREDENTIALS_PATH = Path(os.getenv("CREDENTIALS_PATH", PROJECT_ROOT / "credentials.json"))
TOKEN_PATH = Path(os.getenv("TOKEN_PATH", PROJECT_ROOT / "token.pkl"))

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Point at any OpenAI-compatible endpoint (e.g. OpenRouter:
# https://openrouter.ai/api/v1). Leave unset to use the official OpenAI API.
# When set, remember model names may need a provider prefix
# (e.g. "openai/gpt-4o-mini" on OpenRouter) — see AGENT_MODEL /
# GEN_MODEL_DEFAULT / GEN_MODEL_TOOL_INJECTION below.
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL") or None

# --------------------------------------------------------------------
# Model / embedding config (defaults == original hardcoded values)
# --------------------------------------------------------------------

AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")                      # src/config.py (LLM)
AGENT_TEMPERATURE = float(os.getenv("AGENT_TEMPERATURE", "0.6"))

GEN_MODEL_DEFAULT = os.getenv("GEN_MODEL_DEFAULT", "gpt-4o-mini")          # poison_as.py / poison_correlated.py
GEN_MODEL_TOOL_INJECTION = os.getenv("GEN_MODEL_TOOL_INJECTION", "gpt-4o")  # tool_injection_dataset.py

EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))

# --------------------------------------------------------------------
# Data-generation constants (defaults == original hardcoded values)
# --------------------------------------------------------------------

DATA_SEED = int(os.getenv("DATA_SEED", "42"))
POISON_SAMPLE_SIZE = int(os.getenv("POISON_SAMPLE_SIZE", "100"))
POISON_TOPK_RETRIEVE = int(os.getenv("POISON_TOPK_RETRIEVE", "6"))
POISON_TOOL_NAME = os.getenv("POISON_TOOL_NAME", "get_information")

REDUCE_MAX_CORPUS = int(os.getenv("REDUCE_MAX_CORPUS", "500000"))

AGENT_RECURSION_LIMIT = int(os.getenv("AGENT_RECURSION_LIMIT", "25"))
DEFAULT_RETRIEVAL_K = int(os.getenv("DEFAULT_RETRIEVAL_K", "5"))


def ensure_data_dirs() -> None:
    """Create every data directory this module points at (mkdir -p)."""
    for d in (
        RAW_DIR,
        RAW_NQ_DIR,
        CORPORA_DIR,
        VECTORSTORE_DIR,
        SEED_DIR,
        PROMPTS_DIR,
        TRACES_DIR,
        PROCESSED_DIR,
        UPLOADS_DIR,
    ):
        d.mkdir(parents=True, exist_ok=True)
    for mode_name in MODE_MAP.values():
        corpus_dir(mode_name).mkdir(parents=True, exist_ok=True)
