"""Contract-vector tests for the BRH constraint semantics (CaMeL side).

Runs the shared golden vectors (`brh_contract_vectors.json` at the
repository root) against the reference evaluator `cobra.brh.contract`.
The HTTP proxy suite (`tests/test_http_proxy/test_brh_check.py`)
runs the *same file* against the enforcer's stdlib reimplementation —
that pairing, not a code import, is what keeps the two sides in parity.
"""

import json
import unittest
from pathlib import Path

from cobra.brh.contract import is_placeholder, satisfies, typed_equal

# tests/test_brh/test_contract.py -> test_brh -> tests -> cobra (package root)
VECTORS_PATH = Path(__file__).resolve().parents[2] / "brh_contract_vectors.json"


def load_vectors() -> list[dict]:
    return json.loads(VECTORS_PATH.read_text(encoding="utf-8"))["vectors"]


class TestVectorFile(unittest.TestCase):
    def test_vector_file_exists(self):
        self.assertTrue(VECTORS_PATH.is_file(), f"missing shared vector file: {VECTORS_PATH}")

    def test_vector_ids_unique(self):
        ids = [v["id"] for v in load_vectors()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_vectors_well_formed(self):
        for v in load_vectors():
            with self.subTest(id=v["id"]):
                self.assertIn(v["expect"], ("satisfied", "violation"))
                self.assertIn("op", v["constraint"])
                self.assertIn("value", v["constraint"])
                self.assertIn("observed", v)

    def test_both_outcomes_covered(self):
        outcomes = {v["expect"] for v in load_vectors()}
        self.assertEqual(outcomes, {"satisfied", "violation"})


class TestContractVectors(unittest.TestCase):
    def test_reference_evaluator_matches_vectors(self):
        for v in load_vectors():
            with self.subTest(id=v["id"], note=v.get("note", "")):
                got = satisfies(v["constraint"]["op"], v["constraint"]["value"], v["observed"])
                self.assertEqual(
                    got,
                    v["expect"] == "satisfied",
                    f"vector '{v['id']}': expected {v['expect']}, evaluator said {got}",
                )


class TestHelpers(unittest.TestCase):
    def test_is_placeholder(self):
        self.assertTrue(is_placeholder("trigger_value"))
        self.assertTrue(is_placeholder("from_plan"))
        self.assertFalse(is_placeholder("GBP"))
        self.assertFalse(is_placeholder(42))
        self.assertFalse(is_placeholder(None))

    def test_typed_equal_rejects_cross_type(self):
        self.assertFalse(typed_equal("42", 42))
        self.assertFalse(typed_equal(1, True))
        self.assertTrue(typed_equal(42, 42.0))
        self.assertTrue(typed_equal("a", "a"))

    def test_satisfies_is_total_on_junk(self):
        # Must return False, never raise.
        self.assertFalse(satisfies(None, None, None))
        self.assertFalse(satisfies("==", [1, 2], 1))
        self.assertFalse(satisfies("in", "not-a-list", "x"))
        self.assertFalse(satisfies("<=", {"a": 1}, 5))

    def test_exact_pin_on_nonscalar_is_structural(self):
        # An exact (==) pin on a list/dict means structural equality, not the
        # old "always False" guard: a from_plan list pin (op resolved to ==) must
        # accept an identical list and reject a steered/extended one.
        self.assertTrue(satisfies("==", ["AI", "Blockchain"], ["AI", "Blockchain"]))
        self.assertFalse(satisfies("==", ["AI", "Blockchain"], ["AI", "Blockchain", "Tracker"]))
        self.assertFalse(satisfies("==", ["src/utils.js"], "."))
        self.assertTrue(satisfies("==", {"k": 1}, {"k": 1}))
        self.assertFalse(satisfies("==", {"k": 1}, {"k": 2}))
        # ordering ops on a non-scalar stay fail-closed (the list-guard's real job)
        self.assertFalse(satisfies("<=", [1, 2], [1, 2]))
        # a placeholder inside the pinned structure is unsatisfiable (fail-closed)
        self.assertFalse(satisfies("==", ["from_plan"], ["from_plan"]))


if __name__ == "__main__":
    unittest.main()
