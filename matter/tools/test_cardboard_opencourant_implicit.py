#!/usr/bin/env python3
"""Deck-generation checks; these do not run or qualify OpenCourant."""
import json
from pathlib import Path
import tempfile
import unittest

from cardboard_opencourant_implicit import export_implicit_cases


class ImplicitOutputDecks(unittest.TestCase):
    def test_default_export_preserves_no_output_controls(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = export_implicit_cases(root)

            self.assertEqual(
                [case["id"] for case in cases],
                ["implicit-elastic-nonlinear-force-control",
                 "implicit-cohesive-nonlinear-force-control"],
            )
            for case in cases:
                case_dir = root / case["id"]
                starter = (case_dir / case["starter"]).read_text()
                engine = (case_dir / case["engines"][0]).read_text()
                # Starter history definitions pre-existed, but the default
                # engine does not enable time-history or animation output.
                self.assertEqual(starter.count("/TH/NODE/101"), 1)
                self.assertEqual(starter.count("/TH/NODE/102"), 1)
                self.assertEqual(starter.count("/TH/BRIC/103"), 1)
                self.assertNotIn("/TH/TITLE", engine)
                self.assertNotIn("/TFILE", engine)
                self.assertNotIn("/ANIM/", engine)
                self.assertNotIn("native_outputs", case)
                self.assertEqual(case["prediction"]["mean_upper_DZ_mm"], .01)

    def test_instrumented_variants_add_named_native_channels_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = export_implicit_cases(root, include_instrumented=True)

            self.assertEqual(len(cases), 4)
            controls = {case["id"]: case for case in cases if not case["id"].endswith("-instrumented")}
            instruments = {case["id"].removesuffix("-instrumented"): case
                           for case in cases if case["id"].endswith("-instrumented")}
            self.assertEqual(set(controls), set(instruments))

            for control_id, control in controls.items():
                instrument = instruments[control_id]
                base_dir = root / control_id
                instrument_dir = root / instrument["id"]
                base_starter = (base_dir / control["starter"]).read_text()
                instrument_starter = (instrument_dir / instrument["starter"]).read_text()
                base_engine = (base_dir / control["engines"][0]).read_text()
                engine = (instrument_dir / instrument["engines"][0]).read_text()
                manifest = json.loads((instrument_dir / "case.json").read_text())

                self.assertEqual(instrument_starter.count("/TH/NODE/101"), 1)
                self.assertEqual(instrument_starter.count("/TH/NODE/102"), 1)
                self.assertEqual(instrument_starter.count("/TH/BRIC/103"), 1)
                self.assertIn("REACZ", instrument_starter)
                self.assertIn("DZ", instrument_starter)
                self.assertIn("IE", instrument_starter)
                self.assertIn("/TFILE/4", engine)
                self.assertIn("/ANIM/VECT/DISP", engine)
                self.assertIn("/ANIM/ELEM/ENER", engine)
                self.assertIn("/TH/TITLE", engine)
                self.assertNotIn("/TFILE", base_engine)
                self.assertNotIn("/TH/TITLE", base_engine)
                self.assertNotIn("/ANIM/", base_engine)
                self.assertEqual(manifest["prediction"], control["prediction"])
                self.assertIn("stored as impulse in T01", manifest["native_outputs"]["time_history"]["force_semantics"])
                self.assertIn("official th_to_csv", manifest["native_outputs"]["time_history"]["force_semantics"])
                self.assertIn("/TH/NODE/101 REACZ", manifest["native_outputs"]["time_history"]["force_channel"])

                # Output cards do not change the applied load or the material model.
                for card in ("/CLOAD/1",):
                    self.assertEqual(base_starter.count(card), instrument_starter.count(card))
                for card in ("/IMPL/NONL/01", "/IMPL/SOLV/01"):
                    self.assertEqual(base_engine.count(card), engine.count(card))
                if "elastic" in control_id:
                    self.assertIn("/MAT/LAW1/1", instrument_starter)
                    self.assertIn("/PROP/TYPE14/1", instrument_starter)
                else:
                    self.assertIn("/MAT/LAW117/1", instrument_starter)
                    self.assertIn("/PROP/TYPE43/1", instrument_starter)


if __name__ == "__main__":
    unittest.main()
