"""
IGPO Tool Server - Tool Definitions (Web Search Only)
"""

from typing import Dict, Any

# Web Search Tool Definition
WEB_SEARCH_TOOL = {
    "name": "web_search",
    "description": "Search the web for information using Google or Bing.",
    "inputs": {
        "query": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of search queries (1-3 queries recommended)"
        }
    },
    "example": {"query": ["What is the capital of France?", "Paris population 2024"]}
}

# Local Hybrid Retrieval Tool Definition
# Backed by the Search-R1 hybrid retrieval server (BM25 + Dense, fused via RRF),
# served at /retrieve. See tools_server.search.search_api.local_retrieve.
LOCAL_RETRIEVE_TOOL = {
    "name": "local_retrieve",
    "description": "Retrieve passages from a local document corpus using hybrid (BM25 + dense) search.",
    "inputs": {
        "query": {
            "type": "array",
            "items": {"type": "string"},
            "description": "A list of retrieval queries (1-3 queries recommended)"
        }
    },
    "example": {"query": ["causes of the French Revolution", "Treaty of Versailles terms"]}
}

ALL_TOOLS = {
    "web_search": WEB_SEARCH_TOOL,
    "local_retrieve": LOCAL_RETRIEVE_TOOL,
}


def get_tools(config: Dict[str, Any] = None) -> Dict[str, Dict]:
    """Get available tools, gated by config['available_tools'] (defaults to web_search)."""
    names = (config or {}).get("available_tools") or ["web_search"]
    return {name: ALL_TOOLS[name] for name in names if name in ALL_TOOLS}


def get_tool_names(config: Dict[str, Any] = None) -> list:
    """Get list of available tool names."""
    return list(get_tools(config).keys())
