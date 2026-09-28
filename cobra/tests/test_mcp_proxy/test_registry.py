"""Unit tests for the MCP proxy tool-hash registry."""

import os
import tempfile
import unittest

from cobra.mcp_proxy.registry import (
    canonical_tool_bytes,
    load_registry,
    save_registry,
    tool_hash,
)

_TOOL = {"name": "place_order", "description": "Buy a product.",
         "inputSchema": {"type": "object", "properties": {"id": {"type": "string"}}}}


class Canonicalisation(unittest.TestCase):
    def test_hash_is_key_order_independent(self):
        a = {"name": "t", "description": "d", "inputSchema": {"x": 1, "y": 2}}
        b = {"inputSchema": {"y": 2, "x": 1}, "description": "d", "name": "t"}
        self.assertEqual(tool_hash(a), tool_hash(b))

    def test_description_change_changes_hash(self):
        self.assertNotEqual(tool_hash(_TOOL), tool_hash(dict(_TOOL, description="Buy a product. EVIL")))

    def test_input_schema_change_changes_hash(self):
        mutated = dict(_TOOL, inputSchema={"type": "object", "properties": {"id": {"type": "number"}}})
        self.assertNotEqual(tool_hash(_TOOL), tool_hash(mutated))

    def test_ignores_extraneous_fields(self):
        # transport metadata outside the trusted surface must not affect the hash
        self.assertEqual(canonical_tool_bytes(_TOOL), canonical_tool_bytes(dict(_TOOL, _meta="x")))


class Persistence(unittest.TestCase):
    def test_missing_file_is_empty(self):
        self.assertEqual(load_registry("/no/such/path/reg.json"), {})

    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "nested", "reg.json")
            reg = {"srv::place_order": {"hash": tool_hash(_TOOL), "ts": "t", "approved": True}}
            save_registry(reg, path)
            self.assertEqual(load_registry(path), reg)

    def test_unparsable_file_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "reg.json")
            with open(path, "w") as f:
                f.write("{not json")
            self.assertEqual(load_registry(path), {})


if __name__ == "__main__":
    unittest.main()
