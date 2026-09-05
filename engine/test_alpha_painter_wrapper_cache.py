from __future__ import annotations

import ast
import unittest
from pathlib import Path


class PainterWrapperMeasurementCacheTests(unittest.TestCase):
    def test_supportless_wrapper_preserves_measurement_cache_contract(self) -> None:
        source = Path("app/alpha_candidate_identity.py").read_text(encoding="utf-8")
        tree = ast.parse(source)
        wrapper = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef)
            and node.name == "apply_with_supportless_simple_silhouette"
        )
        self.assertIn("measurement_cache", [arg.arg for arg in wrapper.args.args])
        original_calls = [
            node for node in ast.walk(wrapper)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "original_apply"
        ]
        self.assertEqual(len(original_calls), 2)
        for call in original_calls:
            self.assertTrue(any(
                keyword.arg == "measurement_cache"
                and isinstance(keyword.value, ast.Name)
                and keyword.value.id == "measurement_cache"
                for keyword in call.keywords
            ))


if __name__ == "__main__":
    unittest.main()
