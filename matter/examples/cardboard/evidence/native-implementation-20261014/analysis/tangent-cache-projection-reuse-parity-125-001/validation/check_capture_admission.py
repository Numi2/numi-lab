#!/usr/bin/env python3
"""CPU-only immutable input admission for the source125 cache parity replay."""
import hashlib, json, pathlib, struct, sys
RUN=pathlib.Path('/Users/home/cardboard-implementation-20261014/solver/reset-augmentation-first-failure-capture-360')
EXPECTED={
 'operator-capture':'548dce37fbe5a28a609dd6bc45318926072c878774ea0caffa3237d925cebb33',
 'output/compiled.nmatterpack':'3c25f41c6c26654ccf3f8b4f4b510bbb64ab0ea0f52b94d05d8b1fa4060fafe5',
 'output/rejected-conditioning-000003/candidate_nodes.bin':'dcd6b8ea62c05c3ef9a04dc4d46ece7e7fbad58101d8d55d2bee6dfd9d7e8614',
 'output/rejected-conditioning-000003/manifest.json':'f2775bdc8f1747dcda5136b2bbe890e1d0645804df304c66bfe3143e90d955ad',
 'output/material_state_000003_bent.json':'e352376a5e63ae33802d6eedb225839e4e71d0245076679fe51b5fe938a45a2b',
 'output/manifest.json':'2a34ad4f3328616fb5660fe22022941ca9ace9383ca9f1f1b7262ecfd4350ec8',
}
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def fbits(x): return struct.pack('<f',float(x))
def main():
    for rel,h in EXPECTED.items():
        got=sha(RUN/rel)
        if got!=h: raise SystemExit(f'FAIL hash {rel}: {got}')
    capture=[json.loads(s) for s in (RUN/'operator-capture').read_text().splitlines() if s]
    header=next(r for r in capture if r['record']=='header')
    assert header['schema']=='numi.cardboard.fem-newton-capture.v1'
    assert header['root']==2 and header['newton_iteration']==23
    assert header['world_fingerprint_hex']=='6de3475316b7e8b9'
    assert header['environment_count']==2 and header['node_count_per_environment']==882 and header['tetrahedron_count_per_environment']==1920
    assert header['cached_algorithmic_tangent_capture']['available'] is True
    assert header['cached_algorithmic_tangent_capture']['policy_enabled'] is True
    nodes=[r for r in capture if r['record']=='node']; tets=[r for r in capture if r['record']=='tetrahedron']
    assert len(nodes)==1764 and len(tets)==3840
    raw=(RUN/'output/rejected-conditioning-000003/candidate_nodes.bin').read_bytes()
    assert len(raw)==2*882*80
    nbit=0
    for r in nodes:
        i=r['environment']*882+r['node_id']
        expected=b''.join(fbits(x) for v4 in r['center']['raw_state_float4s'] for x in v4)
        actual=raw[i*80:(i+1)*80]
        assert len(expected)==80 and actual==expected, f'candidate center mismatch env={r["environment"]} node={r["node_id"]}'
        nbit+=20
    # Verify accepted-history exports round-trip to the captured FP32 words for
    # the bent environment; environment 1 remains available only in the capture.
    side=json.loads((RUN/'output/material_state_000003_bent.json').read_text())
    assert side['environment']==0 and len(side['tetrahedra'])==1920
    for t,row in enumerate(side['tetrahedra']):
        cap=next(r for r in tets if r['environment']==0 and r['tet_id']==t)
        assert row['material_index']==cap['material_index']
        assert [fbits(x) for x in row['state']]==[fbits(x) for x in cap['center']['accepted_state']]
    mats=header['materials']
    assert [m['name'] for m in mats]==['hajali2009_liner_hill_bilinear_fit','hajali2009_medium_hill_bilinear_fit','starch2007_finite_glue_elastic']
    report={'schema':'numi.cardboard.tangent-cache-projection-reuse-cpu-admission.v1','status':'pass','physical_steps':0,'run_id':'360','root':2,'newton_iteration':23,'candidate_node_float_words_bitmatched':nbit,'capture_node_rows':len(nodes),'capture_tet_rows':len(tets),'accepted_state_rows_env0_bitmatched':1920,'state_layout':'per-environment/per-tet accepted_state from capture; sidecar confirms bent env0','source125_metallib_dispatch_required':True,'expected_sha256':EXPECTED}
    print(json.dumps(report,sort_keys=True,indent=2))
if __name__=='__main__': main()
