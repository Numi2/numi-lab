#!/usr/bin/env python3
"""Source-bound CPU diagnosis of native run 287's three-face activation rejection."""
from __future__ import annotations
import hashlib
import json
import re
from pathlib import Path

RUN = Path('/Users/home/cardboard-implementation-20261014/native/ordered-cohesive-pages-stage-diagnostics-287')
OUT = Path('/Users/home/cardboard-implementation-20261014/analysis/three-face-mortar-parent-ambiguity-287')
SRC = RUN / 'inputs/source'

def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()

def require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)

def node(i: int, j: int, k: int) -> int:
    return i + 3*j + 9*k

def reconstruct_split_pad():
    split = ((0,1,2,6),(0,2,3,6),(0,3,7,6),(0,7,4,6),(0,4,5,6),(0,5,1,6))
    tets = []
    for k in range(2):
        for j in range(2):
            for i in range(2):
                c = (node(i,j,k), node(i+1,j,k), node(i+1,j+1,k),
                     node(i,j+1,k), node(i,j,k+1), node(i+1,j,k+1),
                     node(i+1,j+1,k+1), node(i,j+1,k+1))
                tets.extend(tuple(c[x] for x in local) for local in split)
    target = tuple(sorted((node(0,0,1), node(1,0,1), node(1,1,1))))
    uses = []
    for tet_index, tet in enumerate(tets):
        for opposite in range(4):
            face = tuple(sorted(tet[k] for k in range(4) if k != opposite))
            if face == target:
                uses.append({'tet':tet_index, 'opposite':opposite,
                             'primitive_id':4*tet_index+opposite, 'tet_nodes':list(tet)})
    return target, uses

def main():
    receipt = json.loads((RUN/'receipt.json').read_text())
    build = json.loads((RUN/'inputs/external/source-build.json').read_text())
    tile_summary = json.loads((RUN/'diagnostic-state/tiled-parity-result.json').read_text())
    trace_path = RUN/'diagnostic-state/face-count-3/legacy/fixture-activation.log'
    records = [json.loads(line.split('=',1)[1]) for line in trace_path.read_text().splitlines()
               if line.startswith('fem_residual_armijo_iteration=')]
    require(receipt['exit_code'] == 1 and receipt['source_inputs_unchanged'] is True and
            receipt['timed_out'] is False, 'run receipt does not match retained native failure')
    require(build['source_snapshot'].endswith('source-captured-point-and-contact-scope-086'),
            'source snapshot mismatch')
    key_files = {
        'fixture': 'matter/tools/cardboard_full_contact_armijo_check.mm',
        'contact': 'matter/src/metal/contact.metalinc',
        'validation': 'matter/src/validation_impl.cpp',
        'shared_status': 'matter/include/numi/matter/shared.h',
    }
    hashes = {}
    for label, rel in key_files.items():
        path = SRC/rel
        require(path.is_file() and rel in build['source_sha256'] and
                sha(path) == build['source_sha256'][rel], f'sealed {label} source hash mismatch')
        hashes[label] = {'path': rel, 'sha256': sha(path)}
    require(len(records) == 32, f'expected 32 Armijo records, found {len(records)}')
    first = records[0]['environments'][0]
    status = first['solver_status']
    trial = first['trial_status']
    require(status['code'] == 6 and status['failing_index'] == 78 and
            status['fgmres_iterations'] == 0 and status['diagnostics'] == [16,99,0,1],
            'first activation status does not match code-6 pair diagnostic')
    require(trial['code'] == 6 and trial['diagnostics'][0:2] == [16,99] and
            trial['diagnostics'][3] == 5, 'independent trial failure does not identify pair 16/99 branch 5')
    require(tile_summary['arms'][2]['error'].find('failing_index=78') >= 0,
            'arm summary does not bind failing slot')
    face_nodes, uses = reconstruct_split_pad()
    require(face_nodes == (9,10,13) and [(u['tet'],u['opposite'],u['primitive_id']) for u in uses]
            == [(4,0,16),(24,3,99)], 'candidate primitive ids do not match reconstructed parent face')
    require((16 >> 2) == 4 and (99 >> 2) == 24, 'primitive parent decode failed')
    fixture_text = (SRC/key_files['fixture']).read_text()
    contact_text = (SRC/key_files['contact']).read_text()
    require('CohesiveMortarPatchSource part = interface;' in fixture_text and
            'part.stableIdentifier = kFaceId + tile;' in fixture_text and
            'part.firstTetrahedron' not in fixture_text, 'fixture copy pattern changed')
    require('if (found) return false;' in contact_text, 'duplicate parent-pair guard changed')
    require(re.search(r'float4\(float\(candidate\.identity\.x\),\s*float\(candidate\.identity\.y\),\s*0\.0f,\s*1\.0f\)', contact_text) is not None,
            'generation failure diagnostic branch changed')
    require('collisionThickness, 5.0f' in contact_text,
            'independent trial-validation diagnostic branch changed')
    require(len(tile_summary['arms']) == 4 and
            tile_summary['arms'][2]['face_count'] == 3 and
            tile_summary['arms'][3]['face_count'] == 3,
            'three-face arms missing')
    report = {
        'schema':'numi.cardboard.three_face_mortar_parent_ambiguity_audit.v1',
        'run_path':str(RUN),
        'run_receipt':receipt,
        'source_snapshot':build['source_snapshot'],
        'source_build_manifest_sha256':sha(RUN/'inputs/external/source-build.json'),
        'retained_source_bindings':hashes,
        'native_observation':{
            'arm':'face_count=3, legacy activation, first Newton record',
            'status_code':status['code'], 'failing_index':status['failing_index'],
            'fgmres_iterations':status['fgmres_iterations'],
            'status_diagnostics':status['diagnostics'],
            'trial_status_code':trial['code'], 'trial_diagnostics':trial['diagnostics'],
            'candidate_primitive_ids':[16,99],
            'candidate_pair_identity_is_4tet_plus_opposite':True
        },
        'independent_geometry_reconstruction':{
            'split_pad_node_index':'i + 3*j + 9*k',
            'selected_shared_interface_face_nodes':list(face_nodes),
            'parent_face_uses':uses,
            'same_parent_pair_for_three_authored_tiles':True,
            'basis':'all three patch objects copy `interface`; only stable ID and barycentric subtriangle differ'
        },
        'source_control_flow':{
            'duplicate_match_behavior':'activeMixedCohesiveFaceForPrimitivePair returns false when a second active mixed-mode face matches the same parent primitive pair',
            'generation_caller':'contact generation records CONTACT_FAILURE with diagnostics [leftPrimitive,rightPrimitive,0,1] when that helper returns false',
            'trial_validation_caller':'private contact trial validation records CONTACT_FAILURE with [leftPrimitive,rightPrimitive,collisionThickness,5] on the same helper failure',
            'pre_fg':True
        },
        'adjudication':{
            'confirmed':'three disjoint mortar subtriangles are authored on one exact parent tetrahedron pair; primitive IDs 16 and 99 identify that pair, and the runtime helper rejects the second active face reference before FGMRES',
            'not_supported_by_this_evidence':['geometric overlap between adjacent tiles','tiled-storage-specific defect','constitutive or FGMRES failure'],
            'smallest_safe_next_step':'For a layout parity fixture, author three non-overlapping mortar faces on three distinct parent primitive pairs (one active mortar face per parent pair). If the intended supported contract is multiple disjoint tiles on one parent pair, implement and test union coverage/complement contact across all matching active pages; do not merely remove the duplicate guard or select one face.'
        },
        'diagnostic_trace_sha256':sha(trace_path),
        'prior_parity_report_sha256':sha(Path('/Users/home/cardboard-implementation-20261014/analysis/tiled-cohesive-stage-diagnostics-287/report.json')),
    }
    out = OUT/'analysis.json'
    out.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':'confirmed_parent_pair_multiplicity_rejection',
                      'candidate_pair':[16,99], 'parent_tets':[4,24],
                      'analysis':str(out)},indent=2))

if __name__=='__main__':
    main()
