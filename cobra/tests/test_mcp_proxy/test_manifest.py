"""Unit tests for the MCP tool-defs -> BRH manifest bridge."""

import unittest

from cobra.mcp_proxy.manifest import manifest_from_tools, server_map_for_tools

_TOOLS = [
    {"name": "get_product",
     "inputSchema": {"type": "object", "properties": {"product_id": {"type": "string"}}}},
    {"name": "place_order",
     "inputSchema": {"type": "object",
                     "properties": {"product_id": {"type": "string"},
                                    "amount": {"type": "number"},
                                    "currency": {"type": "string"}}}},
]


class ManifestFromTools(unittest.TestCase):
    def test_maps_names_to_param_names(self):
        self.assertEqual(
            manifest_from_tools(_TOOLS),
            {"get_product": ["product_id"],
             "place_order": ["product_id", "amount", "currency"]},
        )

    def test_no_properties_is_empty_param_list(self):
        self.assertEqual(manifest_from_tools([{"name": "ping", "inputSchema": {}}]), {"ping": []})

    def test_missing_input_schema_is_empty_param_list(self):
        self.assertEqual(manifest_from_tools([{"name": "ping"}]), {"ping": []})

    def test_unnamed_tool_skipped(self):
        self.assertEqual(manifest_from_tools([{"inputSchema": {"properties": {"x": {}}}}]), {})

    def test_empty_input(self):
        self.assertEqual(manifest_from_tools([]), {})
        self.assertEqual(manifest_from_tools(None), {})


class ServerMapForTools(unittest.TestCase):
    def test_maps_all_named_tools_to_server_id(self):
        self.assertEqual(
            server_map_for_tools(_TOOLS, "my-server"),
            {"get_product": "my-server", "place_order": "my-server"},
        )

    def test_unnamed_tool_skipped(self):
        tools = [{"inputSchema": {}}, {"name": "ping"}]
        self.assertEqual(server_map_for_tools(tools, "s"), {"ping": "s"})

    def test_empty_input(self):
        self.assertEqual(server_map_for_tools([], "s"), {})
        self.assertEqual(server_map_for_tools(None, "s"), {})


if __name__ == "__main__":
    unittest.main()
