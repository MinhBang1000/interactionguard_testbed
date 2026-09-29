"""
constraints.py

Central place for string constants used across the AI agent security testbed.

Goal:
    - Avoid magic strings like "agent", "tools", "query_memory" scattered everywhere.
    - Make refactors (renaming nodes/tools) safe and consistent.
    - Keep security-sensitive constants (e.g., attacker email) in one place.

Naming convention:
    - NODE_*     for LangGraph node names.
    - STATE_*    for keys in the LangGraph state dictionaries.
    - TOOL_*     for @tool names (must match the tool's .name exactly).
    - SPECIAL_*  for special tokens, IDs, or fixed addresses.
"""

# ==========================
# LangGraph node names
# ==========================

# Main reasoning node (LLM + tools)
NODE_AGENT: str = "agent"

# Node that executes tools (Gmail, query_memory, file tools, etc.)
NODE_TOOLS: str = "tools"

# Legacy / optional nodes (kept for backward-compatibility or experiments)
NODE_MODEL: str = "model"        # old "model" node
NODE_RETRIEVE: str = "retrieve"  # old RAG-retrieve node
NODE_FINAL: str = "final"        # old "final" summarizer node


# ==========================
# State / event keys
# ==========================

# Key under which LangGraph stores messages in the node state.
STATE_MESSAGES: str = "messages"

# Optional key used in config for per-thread configuration.
STATE_CONFIG: str = "config"

# Key used in your code when extracting events (utils.extract_event).
STATE_NODE_NAME: str = "node_name"  # if you ever wrap events with extra metadata


# ==========================
# Tool names
# ==========================
# IMPORTANT:
#   The string values here MUST match the .name of the corresponding @tool
#   in tools.py, or LangChain / LangGraph will not route calls correctly.

# RAG / memory tool
TOOL_QUERY_MEMORY: str = "query_memory"

# Gmail tools (names should match the @tool definitions in tools.py)
TOOL_SEARCH_EMAILS: str = "search_emails"
TOOL_SEND_EMAIL: str = "send_email"
TOOL_REPLY_EMAIL: str = "reply_email"
TOOL_REPLY_ALL_EMAIL: str = "reply_all_email"
TOOL_FORWARD_EMAIL: str = "forward_email"
TOOL_IS_REPLY_OR_REPLY_ALL: str = "is_reply_or_reply_all"
TOOL_GET_ALL_EMAILS: str = "get_all_emails"
TOOL_UPDATE_EMAILS: str = "update_emails"
TOOL_GET_INFORMATION: str = "get_information"

# File tools (DOCX / XLSX / PDF)
TOOL_READ_DOCX: str = "read_docx"
TOOL_READ_XLSX: str = "read_xlsx"
TOOL_CREATE_DOCX: str = "create_docx"
TOOL_CREATE_XLSX: str = "create_xlsx"
TOOL_READ_PDF: str = "read_pdf"

# Misc / helper tools
TOOL_EXTRACT_MESSAGE: str = "extract_tools"  # example from tools.py


# ==========================
# Special / security-related constants
# ==========================

# Fixed attacker email used in your attack scenarios (for ASR measurement).
SPECIAL_ATTACKER_EMAIL: str = "lmbang.ee13@nycu.edu.tw"

# If you ever introduce a special control token (e.g., to trigger RAG),
# define it here. Example (currently not used, but reserved):
SPECIAL_RETRIEVE_TOKEN: str = "[RETRIEVE]"
