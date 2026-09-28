"""MESA — MCP and Endpoint Sitemap generator for Agents.

Given a site URL, crawl it, propose the essential HTTP calls, derive one MCP
tool per HTTP call, let the owner curate each in a terminal UI, polish
descriptions/schemas with an LLM, approve, and emit repo-format
``sitemap.json`` + ``mcp_manifest.json``.
"""

__version__ = "1.0.0"
