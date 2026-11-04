"""The Cake planner port against FlashInfer's own planner, on the CPU.

Runs the pinned, unmodified FlashInfer `plan_attention` (reference/, stubbed
torch, no CUDA) over a dense grid of capture contexts at 148 and 152 SMs and
compares every batch's split count with scripts/cake_plan.py.
"""

import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "scripts"))
from cake_plan import cake_splits
from planner_oracle import check_policy, load_reference, reference_plans


class CakePlanTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plans = reference_plans()

    def test_dense_upstream_plans_and_recorded_golden(self):
        check_policy(cake_splits, self.plans)
        golden = json.loads((HERE / "planner-golden.json").read_text())
        for record in golden["cases"]:
            expected = tuple(map(tuple, record["bands"]))
            self.assertEqual(self.plans[record["sm"], record["local_context"]], expected)
            self.assertEqual(max(hi * n for _, hi, n in expected), record["max_rows_times_splits"])
        print(
            f"FlashInfer oracle: {len({ctx for _, ctx in self.plans})} contexts x "
            f"{len({sm for sm, _ in self.plans})} SM counts x 64 batches = "
            f"{len(self.plans) * 64} shapes"
        )

    def test_unsupported_inputs_are_refused(self):
        for local_context, sm in ((0, 148), (8191, 148), (131072, 80), (131072, 144)):
            with self.subTest(local_context=local_context, sm=sm), self.assertRaises(ValueError):
                cake_splits(local_context, sm)

    def test_oracle_rejects_changed_source(self):
        with tempfile.TemporaryDirectory() as directory:
            p = Path(directory) / "reference.py"
            p.write_text("raise RuntimeError('must never execute')")
            with self.assertRaisesRegex(ValueError, "SHA-256"):
                load_reference(p)

    def test_all_eight_planner_mutants_are_detected(self):
        source = (HERE.parents[1] / "scripts/cake_plan.py").read_text()
        mutants = {
            "wave15": ("+ 16)", "+ 15)"),
            "wave17": ("+ 16)", "+ 17)"),
            "merge0.04": ("+ 0.05 * items", "+ 0.04 * items"),
            "threshold1.10": ("< 1.15", "< 1.10"),
            "threshold1.20": ("< 1.15", "< 1.20"),
            "max64": ("min(256, tiles // 2)", "min(64, tiles // 2)"),
            "guard4096": ("local_context < 8192", "local_context < 4096"),
            "pairs_plus_one": ("pairs = sm_count // 2", "pairs = sm_count // 2 + 1"),
        }
        detected = 0
        for name, (old, new) in mutants.items():
            with self.subTest(mutant=name):
                self.assertEqual(source.count(old), 1)
                module = types.ModuleType("_cake_plan_mutant_" + name)
                sys.modules[module.__name__] = module
                try:
                    exec(  # noqa: S102 -- fixed local-source mutations
                        compile(source.replace(old, new), name, "exec"), module.__dict__
                    )
                    with self.assertRaises(ValueError):
                        check_policy(module.cake_splits, self.plans)
                    detected += 1
                finally:
                    del sys.modules[module.__name__]
        print(f"Planner mutation regression: {detected}/{len(mutants)} detected")


if __name__ == "__main__":
    unittest.main()
