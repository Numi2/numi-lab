"""Materialize only an owner's accessible bounded readout for native inspection.

No raw assay file is opened here. The existing owner admits each cell and supplies
every value. This is a format conversion, not preprocessing or a new analysis.
"""
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
import uuid

from biological_viewers import prepare, public_context
from wet_lab_shared import Conflict, read


def admitted_rows(readout, selection):
    if readout.get('access') != {'status': 'development-open-only', 'reservedResponsesReturned': False}:
        raise ValueError('Owner has not admitted an accessible-only measured readout')
    if (readout.get('modeled') is not False or readout.get('measured') is not True
            or readout.get('normalization') != 'raw_counts'
            or readout.get('units') != 'raw UMI counts'
            or readout.get('gene') != selection.get('gene')):
        raise ValueError('A measured, unambiguous raw-count feature is required')
    arms = readout.get('arms', [])
    if len(arms) != 1 or arms[0].get('target') != selection.get('target') or arms[0].get('population') != selection['populationID']:
        raise ValueError('Owner readout does not match the selected intervention and population')
    arm = arms[0]
    rows, seen = [], {}
    for role, id_key, value_key in [('control', 'controlCellIDs', 'controlValues'),
                                     ('selected-intervention', 'observedCellIDs', 'observedValues')]:
        ids, values = arm.get(id_key, []), arm.get(value_key, [])
        if len(ids) != len(values) or len(ids) > 128:
            raise ValueError('Owner cell identities and bounded values must match')
        for identifier, value in zip(ids, values):
            if not isinstance(identifier, str) or not identifier or not isinstance(value, (float, int)) or isinstance(value, bool) or not math.isfinite(value) or value < 0 or value != int(value):
                raise ValueError('Invalid source cell identity or raw UMI count')
            if identifier in seen:
                prior = seen[identifier]
                if selection['target'] != 'no-intervention' or prior['value'] != value or prior['role'] != 'control':
                    raise ValueError('Ambiguous duplicate source cell identity')
                prior['role'] = 'control-and-selected-intervention'
                continue
            row = {'id': identifier, 'value': value, 'role': role}
            rows.append(row); seen[identifier] = row
    if not rows:
        raise ValueError('No accessible measured cells in this page; no zero-filled source was created')
    return rows


def write_h5ad(path, rows, selection, provenance):
    try:
        import h5py
        import numpy as np
    except ImportError as error:
        raise ValueError('The Wet Lab owner environment needs h5py and NumPy for AnnData export') from error
    with h5py.File(path, 'x') as f:
        f.attrs.update({'encoding-type': 'anndata', 'encoding-version': '0.1.0'})
        x = f.create_dataset('X', data=np.array([[r['value']] for r in rows], dtype=np.float64))
        x.attrs.update({'encoding-type': 'array', 'encoding-version': '0.2.0'})
        def strings(parent, key, values):
            d = parent.create_dataset(key, data=values, dtype=h5py.string_dtype('utf-8'))
            d.attrs.update({'encoding-type': 'string-array', 'encoding-version': '0.2.0'})
        def frame(name, ids, columns):
            g = f.create_group(name)
            g.attrs.update({'encoding-type': 'dataframe', 'encoding-version': '0.2.0', '_index': '_index'})
            g.attrs.create('column-order', list(columns), dtype=h5py.string_dtype('utf-8'))
            strings(g, '_index', ids)
            for key, values in columns.items(): strings(g, key, values)
        frame('obs', [r['id'] for r in rows], {'numi_role': [r['role'] for r in rows],
              'numi_population': [selection['populationID']] * len(rows)})
        frame('var', [selection['gene']], {'gene_symbol': [selection['gene']]})
        for name in ('obsm', 'obsp', 'varm', 'varp', 'layers', 'uns'):
            g = f.create_group(name); g.attrs.update({'encoding-type': 'dict', 'encoding-version': '0.1.0'})
        d = f['uns'].create_dataset('numi_provenance_json', data=json.dumps(provenance, allow_nan=False), dtype=h5py.string_dtype('utf-8'))
        d.attrs.update({'encoding-type': 'string', 'encoding-version': '0.2.0'})


def export_selected(shared, owner, expected_revision, *, limit=128, offset=0):
    if type(limit) is not int or not 1 <= limit <= 128 or type(offset) is not int or offset < 0:
        raise ValueError('Choose a bounded page of 1–128 cells per group and a nonnegative offset')
    with shared.lock():
        state = read(shared.path)
        if expected_revision != state['revision']:
            raise Conflict('Selection changed. Read context and review the new selection before exporting.')
        selection = state.get('selection')
        if not selection or not selection.get('gene') or not selection.get('target'):
            raise ValueError('Select a population, gene and intervention first')
        config = shared.config['assays'][selection['assayID']]
        capability = owner.capabilities(Path(config))
        if capability.get('measuredExploration') is not True or capability.get('prediction') is not False:
            raise ValueError('This owner has no admitted measured-only viewer export; keep its existing readout')
        query = {'kind': 'gene', 'mode': 'measured', 'gene': selection['gene'], 'target': selection['target'],
                 'normalization': 'raw_counts', 'limit': limit, 'offset': offset}
        result = owner.readout(config, None, query, shared.selection(selection))
        rows = admitted_rows(result, selection)
        context = public_context(state, shared.root)
        provenance = {'schema': 'numi.viewer-readout.v1', 'producer': 'NumiVivo owner readout → Numi AnnData format export',
                      'context': context, 'query': query, 'units': result['units'],
                      'datasetFingerprint': result['datasetFingerprint'],
                      'observationAccessFingerprint': result['observationAccessFingerprint'],
                      'coverage': {'exportedCells': len(rows), 'exportedGenes': 1, 'spatialCoordinates': False,
                                   'scope': result['scope'], 'pageOffset': offset, 'limitPerGroup': limit,
                                   'reservedResponsesReturned': False, 'fullAssay': False},
                      'statistics': 'No statistics recomputed; source counts preserved exactly'}
        parent = shared.root / 'viewer-artifacts'
        parent.mkdir(exist_ok=True, mode=0o700)
        if parent.is_symlink(): raise ValueError('Viewer artifact directory must not be a symbolic link')
        destination = parent / uuid.uuid4().hex
        staging = Path(tempfile.mkdtemp(prefix='.preparing-', dir=parent))
        try:
            write_h5ad(staging / 'selected-cells.h5ad', rows, selection, provenance)
            handoff = prepare(staging / 'selected-cells.h5ad', 'NumiVivo', context=context)
            handoff['source']['path'] = str(destination / 'selected-cells.h5ad')
            handoff['open']['arguments']['path'] = handoff['source']['path']
            handoff['sourceAssociation'] = {'status': 'owner-readout-bound', 'coverage': provenance['coverage'],
                                          'datasetFingerprint': result['datasetFingerprint'],
                                          'observationAccessFingerprint': result['observationAccessFingerprint']}
            (staging / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False) + '\n')
            (staging / 'handoff.json').write_text(json.dumps(handoff, indent=2, allow_nan=False) + '\n')
            os.rename(staging, destination)
        except BaseException:
            shutil.rmtree(staging)
            raise
        return {'status': 'prepared', 'handoffPath': str(destination / 'handoff.json'),
                'handoff': handoff, 'coverage': provenance['coverage'], 'revision': state['revision'],
                'observationAccess': 'unchanged', 'viewerReady': False}
