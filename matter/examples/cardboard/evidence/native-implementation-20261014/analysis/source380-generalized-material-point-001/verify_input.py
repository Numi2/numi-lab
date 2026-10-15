#!/usr/bin/env python3
"""CPU preflight for the exact source380 tet674 diagnostic input; never launches Metal."""
from __future__ import annotations
import hashlib, json, pathlib, struct, sys


def sha(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def flat(value):
    if isinstance(value, list):
        for child in value:
            yield from flat(child)
    else:
        yield value


def f32_word(value) -> str:
    return f"0x{struct.unpack('<I', struct.pack('<f', float(value)))[0]:08x}"


def verify(input_path: pathlib.Path, medium_path: pathlib.Path) -> dict:
    doc = json.loads(input_path.read_text())
    assert doc['schema'] == 'numi.cardboard.source380-native-material-point-input.v2'
    assert doc['case_count'] == 1 and len(doc['cases']) == 1
    bind = doc['run_binding']; case = doc['cases'][0]
    assert bind['run_name'] == 'explicit32-certified-coupon-380'
    run = pathlib.Path(bind['run_path'])
    relative = {
        'output/rejected-conditioning-000002/candidate_nodes.bin':
            bind['source_candidate_identity']['candidate_nodes_sha256'],
        'output/material_state_000002_bent.json':
            bind['source_candidate_identity']['accepted_history_sha256'],
        'output/mesh.json': bind['source_candidate_identity']['mesh_sha256'],
        'output/compiled.nmatterpack': bind['compiled_package_sha256'],
    }
    observed = {}
    for rel, expected in relative.items():
        path = run / rel
        actual = sha(path)
        assert actual == expected, f'{rel} hash differs from source-bound input'
        assert bind['artifact_sha256'][rel] == actual, f'{rel} artifact hash differs'
        observed[rel] = actual
    assert sha(medium_path) == bind['source_medium_asset_sha256']
    history_path = run / 'output/material_state_000002_bent.json'
    history = json.loads(history_path.read_text())
    assert history['environment'] == case['environment'] == 0
    tet = history['tetrahedra'][case['tet_id']]
    assert tet['index'] == case['tet_id'] == 674
    assert tet['material_index'] == case['material_index'] == 1
    history_words = [f32_word(x) for x in tet['state']]
    assert history_words == case['accepted_state_f32_bits']
    words = []
    for key in ('F_high','F_low','Fdot','deformation_direction','rate_direction'):
        words.extend(f32_word(x) for x in flat(case[key]))
    words.extend((f32_word(case['dt_seconds']), f32_word(case['temperature_K'])))
    assert len(words) == 47 and words == case['packed_input_f32_bits']
    high_words = [f32_word(x) for x in flat(case['F_high'])]
    low_words = [f32_word(x) for x in flat(case['F_low'])]
    assert high_words == list(flat(case['F_high_bits']))
    assert low_words == list(flat(case['F_low_bits']))
    return {
        'status':'cpu_binding_preflight_pass',
        'input_sha256':sha(input_path),
        'medium_source_sha256':sha(medium_path),
        'run_artifacts_sha256':observed,
        'environment':case['environment'], 'tet_id':case['tet_id'],
        'material_index':case['material_index'],
        'accepted_history_words_match':True,
        'F_high_low_provenance':'CPU transcription bound to exact candidate/package hashes; not native readback',
        'physical_steps':0, 'physical_validation':False,
    }

if __name__ == '__main__':
    if len(sys.argv) != 3:
        raise SystemExit('usage: verify_input.py INPUT.json MEDIUM_SOURCE.nmatter')
    print(json.dumps(verify(pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])), indent=2))
