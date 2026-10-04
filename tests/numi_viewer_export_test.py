import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'python/metalrobo'))
from biological_viewers import verify
from wet_lab_shared import SharedWorkspace, Conflict, write
from wet_lab_viewer_export import admitted_rows, export_selected


class SelectedExport(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.selected = {'assayID': 'a', 'specimenID': 's', 'populationID': 'p', 'conditionID': 'c', 'gene': 'G', 'target': 'T'}
        self.readout = {'access': {'status': 'development-open-only', 'reservedResponsesReturned': False},
            'datasetFingerprint': 'dataset', 'observationAccessFingerprint': 'access', 'modeled': False,
            'measured': True, 'gene': 'G', 'normalization': 'raw_counts', 'units': 'raw UMI counts', 'scope': 'accessible development page',
            'arms': [{'target': 'T', 'population': 'p', 'controlCellIDs': ['source:c1'], 'controlValues': [3.],
                      'observedCellIDs': ['source:t1'], 'observedValues': [7.], 'reservedOutcomeCells': 900}]}

    def test_owner_access_missing_feature_and_identity_mismatch_fail(self):
        for mutate in [lambda r: r['access'].update(reservedResponsesReturned=True),
                       lambda r: r.update(measured=False), lambda r: r.update(gene='OTHER'),
                       lambda r: r['arms'][0].update(observedCellIDs=['other', 'extra']),
                       lambda r: r['arms'][0].update(observedValues=[float('nan')]),
                       lambda r: r['arms'][0].update(observedCellIDs=['source:c1'])]:
            bad = copy.deepcopy(self.readout); mutate(bad)
            with self.assertRaises(ValueError): admitted_rows(bad, self.selected)

    def test_no_intervention_deduplicates_exact_control_identity(self):
        self.selected['target'] = 'no-intervention'
        self.readout['arms'][0].update(target='no-intervention', observedCellIDs=['source:c1'], observedValues=[3.])
        rows = admitted_rows(self.readout, self.selected)
        self.assertEqual(rows, [{'id': 'source:c1', 'value': 3., 'role': 'control-and-selected-intervention'}])

    @unittest.skipUnless(importlib.util.find_spec('h5py'), 'Run in the Wet Lab h5py environment for AnnData serialization')
    def test_export_preserves_counts_ids_and_revision_without_reserved_rows(self):
        import h5py
        shared = SharedWorkspace(self.root, {'assays': {'a': str(self.root/'assay.json')}})
        state = {'revision': 4, 'selection': self.selected, 'drafts': [], 'operations': [], 'history': []}
        write(shared.path, state)
        class Owner:
            @staticmethod
            def capabilities(_): return {'measuredExploration': True, 'prediction': False}
            @staticmethod
            def readout(*args): return self.readout
        exported = export_selected(shared, Owner, 4)
        self.assertEqual(json.loads(shared.path.read_text()), state)
        self.assertEqual(exported['coverage']['exportedCells'], 2)
        self.assertEqual(exported['coverage']['exportedGenes'], 1)
        verify(exported['handoff'])
        with h5py.File(exported['handoff']['source']['path']) as f:
            self.assertEqual(f['X'][:].tolist(), [[3.], [7.]])
            self.assertEqual(f['obs/_index'].asstr()[:].tolist(), ['source:c1', 'source:t1'])
            self.assertEqual(f['var/_index'].asstr()[:].tolist(), ['G'])
            self.assertEqual(list(f['obsm']), [])
            provenance = json.loads(f['uns/numi_provenance_json'].asstr()[()])
            self.assertFalse(provenance['coverage']['reservedResponsesReturned'])
            self.assertEqual(provenance['observationAccessFingerprint'], 'access')
        with self.assertRaises(Conflict): export_selected(shared, Owner, 3)
        with self.assertRaises(ValueError): export_selected(shared, Owner, 4, limit=129)
        with patch('wet_lab_viewer_export.write_h5ad', side_effect=OSError('disk failure')):
            with self.assertRaises(OSError): export_selected(shared, Owner, 4)
        self.assertEqual(len(list((self.root/'viewer-artifacts').iterdir())), 1)


if __name__ == '__main__': unittest.main()
