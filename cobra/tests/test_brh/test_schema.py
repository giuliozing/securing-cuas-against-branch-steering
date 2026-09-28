import unittest

import pydantic

from cobra.brh.schema import FieldConstraint


class FieldConstraintOpPairingTest(unittest.TestCase):
    def test_comparison_with_scalar_ok(self):
        fc = FieldConstraint(path="amount", op="<=", value="trigger_value")
        self.assertEqual(fc.value, "trigger_value")

    def test_in_with_list_ok(self):
        fc = FieldConstraint(path="currency", op="in", value=["GBP", "EUR"])
        self.assertEqual(fc.value, ["GBP", "EUR"])

    def test_in_with_scalar_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            FieldConstraint(path="currency", op="in", value="GBP")

    def test_in_with_empty_list_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            FieldConstraint(path="currency", op="in", value=[])

    def test_comparison_with_list_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            FieldConstraint(path="amount", op="<=", value=[10, 50])

    def test_range_is_two_anded_constraints(self):
        lo = FieldConstraint(path="amount", op=">=", value=10)
        hi = FieldConstraint(path="amount", op="<=", value=50)
        self.assertEqual((lo.path, hi.path), ("amount", "amount"))

    def test_bool_member_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            FieldConstraint(path="flag", op="in", value=[True, False])

    def test_subset_with_list_ok(self):
        fc = FieldConstraint(path="sans", op="subset", value=["a.acme.local", "b.acme.local"])
        self.assertEqual(fc.op, "subset")
        self.assertEqual(fc.value, ["a.acme.local", "b.acme.local"])

    def test_subset_with_scalar_rejected(self):
        with self.assertRaises(pydantic.ValidationError):
            FieldConstraint(path="sans", op="subset", value="a.acme.local")

    def test_eq_struct_with_dict_ok(self):
        rec = {"type": "A", "name": "pay.acme.local", "value": "203.0.113.10"}
        fc = FieldConstraint(path="record", op="eq_struct", value=rec)
        self.assertEqual(fc.op, "eq_struct")
        self.assertEqual(fc.value, rec)


if __name__ == "__main__":
    unittest.main()
