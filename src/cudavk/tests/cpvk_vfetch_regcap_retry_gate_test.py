#!/usr/bin/env python3
"""CPU-only parser regressions using the original B200 presence logs."""
from pathlib import Path
import unittest
from cpvk_vfetch_regcap_retry_gate import validate_logs


class PresenceParserTest(unittest.TestCase):
    def setUp(self):
        root = Path(__file__).parent / "fixtures/vfetch_regcap_retry"
        self.logs = [(root / (n + ".log")).read_text()
                     for n in ("default", "0", "1")]

    def test_actual_valid_log(self):
        validate_logs(*self.logs)  # No teardown census: fixture omits destroy.

    def reject(self, on):
        with self.assertRaises(ValueError):
            validate_logs(self.logs[0], self.logs[1], on)

    def test_original_bad_refusal(self):
        self.reject(self.logs[1])

    def test_plan_without_actual_launch(self):
        self.reject("\n".join(line for line in self.logs[2].splitlines()
                              if not line.startswith("run-geometry-launch:")))

    def test_census_cannot_replace_launch(self):
        on = "\n".join(line for line in self.logs[2].splitlines()
                       if not line.startswith("run-geometry-launch:"))
        self.reject(on + "\ncudavk: vsfetch shader 0 launches 9 fused 9 N 4 admitted\n")

    def test_bad_resources_or_geometry(self):
        for old, new in [("cap 32 regs 32 spill 0", "cap 32 regs 32 spill 4"),
                         ("members=9 draws=9 verts=162000", "members=9 draws=9 verts=1"),
                         ("fused regs  32", "fused regs  35"),
                         ("blocks/sm 8 admitted", "blocks/sm 6 admitted")]:
            with self.subTest(new=new):
                self.reject(self.logs[2].replace(old, new))

    def test_missing_initial_resources(self):
        logs = ["\n".join(l for l in log.splitlines()
                            if not l.startswith("cudavk: shader ")) for log in self.logs]
        with self.assertRaises(ValueError):
            validate_logs(*logs)


if __name__ == "__main__":
    unittest.main()
