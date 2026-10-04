import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "python/metalrobo"))
import biological_viewers as views


class ViewerHandoffs(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name).resolve()
        self.addCleanup(self.tmp.cleanup)

    def source(self, name, data=b"source bytes\n"):
        p = self.root / name; p.write_bytes(data); return p

    def test_owner_formats_and_related_data_are_distinct(self):
        for path, kind in [("protein.PDB.GZ", "structure"), ("ligand.sdf", "structure"),
                           ("rna.gbff", "sequence"), ("alignment.aln-fasta", "sequence"),
                           ("sample.h5ad", "slide"), ("section.ome.tiff", "slide")]:
            self.assertEqual(views.route(path), kind)
        for path in ("movie.xtc", "overlay.geojson", "data.csv", "report.json", "image.png"):
            with self.assertRaises(ValueError): views.route(path)

    def test_open_intent_and_companion_hashes_survive_verification(self):
        h = views.prepare(self.source("system.pdb"), "NumiVivo", placement="side-pane",
                          companions=[self.source("run.xtc")])
        verified = views.verify(h)
        self.assertEqual(verified["open"], h["open"])
        self.assertEqual(h["open"]["tool"], "structure.open_in_side_pane")
        self.assertFalse(verified["viewerReady"])
        self.source("run.xtc", b"changed trajectory")
        with self.assertRaises(ValueError): views.verify(h)

    def test_changed_source_and_untrusted_dispatch_fail(self):
        path = self.source("sequence.fasta")
        h = views.prepare(path, "NumiVivo")
        bad = copy.deepcopy(h); bad["open"]["tool"] = "sequence.edit"
        with self.assertRaises(ValueError): views.verify(bad)
        bad = copy.deepcopy(h); bad["open"]["arguments"]["arbitrary"] = "bad"
        with self.assertRaises(ValueError): views.verify(bad)
        path.write_bytes(b"source BYTES\n")
        with self.assertRaises(ValueError): views.verify(h)

    def test_limits_symlinks_and_wrong_companions_fail(self):
        path = self.source("protein.pdb")
        with self.assertRaises(ValueError): views.prepare(path, "NumiVivo", max_bytes=1)
        link = self.root / "linked.pdb"; link.symlink_to(path)
        with self.assertRaises(ValueError): views.prepare(link, "NumiVivo")
        with self.assertRaises(ValueError):
            views.prepare(path, "NumiVivo", companions=[self.source("assay.h5ad")])

    def context(self):
        state = {"revision": 4, "selection": {"assayID": "a", "specimenID": "s", "populationID": "p",
                 "gene": "G1", "conditionID": "c", "target": "t", "experimentID": None},
                 "drafts": [{"secret": "reserved observation"}], "token": "credential-not-for-viewer"}
        return state, views.public_context(state, self.root)

    def test_context_excludes_observations_and_credentials(self):
        state, context = self.context()
        self.assertNotIn("secret", json.dumps(context))
        self.assertNotIn("credential-not-for-viewer", json.dumps(context))
        self.assertEqual(state["selection"], context["selection"])
        self.assertEqual(context["observationAccess"], "unchanged")

    def test_return_keeps_axes_and_fails_after_human_edits(self):
        state, context = self.context()
        h = views.prepare(self.source("section.h5ad"), "NumiVivo", context=context)
        config = {"workspace": str(self.root)}
        edit = views.return_selection(h, config, state, "G2", 4)
        self.assertEqual(edit["selection"], {**state["selection"], "gene": "G2"})
        self.assertEqual(edit["action"], "selection")
        self.assertNotIn("authorizeReveal", edit)
        state["revision"] = 5
        with self.assertRaises(ValueError): views.return_selection(h, config, state, "G2", 5)
        state["revision"] = 4; state["selection"]["target"] = "human edit"
        with self.assertRaises(ValueError): views.return_selection(h, config, state, "G2", 4)
        with self.assertRaises(ValueError): views.return_selection(h, {"workspace": "/different"}, state, "G2", 4)

    def test_cli_does_not_overwrite_handoff(self):
        source = self.source("sequence.fasta")
        output = self.source("handoff.json", b"retained evidence")
        self.assertEqual(views.main(["prepare", str(source), "--output", str(output)]), 2)
        self.assertEqual(output.read_bytes(), b"retained evidence")


if __name__ == "__main__": unittest.main()
