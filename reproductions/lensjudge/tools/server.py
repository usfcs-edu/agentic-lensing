"""Assemble the in-process ``lens`` MCP server from selected tools.

Returns the (mcp_servers, allowed_tools) pair for ClaudeAgentOptions. Heavy tools
that subprocess into other venvs (cnn_score, quick_lensmodel) and the spectroscopic
tools are imported lazily and only registered if requested, so the lean imaging
grader has no torch/JAX/DESI dependency.
"""
from __future__ import annotations

from claude_agent_sdk import create_sdk_mcp_server

from lensjudge import config
from lensjudge.tools.cutout import fetch_cutout
from lensjudge.tools.photometry import get_photometry
from lensjudge.tools.crossmatch import crossmatch_local

# always-available, dependency-light tools
_CORE = {
    "fetch_cutout": fetch_cutout,
    "get_photometry": get_photometry,
    "crossmatch_local": crossmatch_local,
}


def _lazy(name):
    """Import an optional heavy tool by name; return the SdkMcpTool or None."""
    try:
        if name == "cnn_score":
            from lensjudge.tools.cnn import cnn_score
            return cnn_score
        if name == "quick_lensmodel":
            from lensjudge.tools.quicklens import quick_lensmodel
            return quick_lensmodel
        if name in ("fetch_spectrum", "get_specfit"):
            from lensjudge.tools import spectrum as _sp
            return getattr(_sp, name)
        if name == "lens_representations":
            from lensjudge.tools.representations import lens_representations
            return lens_representations
        if name == "fetch_euclid_cutout":
            from lensjudge.tools.euclid_cutout import fetch_euclid_cutout
            return fetch_euclid_cutout
    except Exception:
        return None
    return None


def build(tool_names=None):
    """Build (mcp_servers, allowed_tools) for the given tool names.

    Defaults to the lean imaging set (fetch_cutout, get_photometry).
    """
    names = list(tool_names or ["fetch_cutout", "get_photometry"])
    tools, ok = [], []
    for n in names:
        t = _CORE.get(n) or _lazy(n)
        if t is not None:
            tools.append(t); ok.append(n)
    server = create_sdk_mcp_server(config.MCP_SERVER_NAME, "0.1.0", tools=tools)
    allowed = [f"mcp__{config.MCP_SERVER_NAME}__{n}" for n in ok]
    return {config.MCP_SERVER_NAME: server}, allowed
