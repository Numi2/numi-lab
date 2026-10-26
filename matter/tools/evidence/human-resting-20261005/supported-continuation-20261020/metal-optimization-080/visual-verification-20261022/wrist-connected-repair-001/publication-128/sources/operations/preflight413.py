from pathlib import Path
import json,hashlib,numpy as np
D=Path('/Users/numi/Work/numi-human-connected-surface-069');P=D/'D/surface-witness-basis-413/run-001-launch'
a=json.load(open(P/'draft-argv.json'))
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arg(k):return a[a.index(k)+1]
base=Path(arg('--base-candidate'))
assert sha(base)=='141ca8ff15a4377092e198c9d7d52d08034541cbf295979e7b539c34d508a090'
j=json.load(open(base.parent/'audit-130/summary.json'));s=json.load(open(base.parent/'audit-159/report.json'))
assert j['candidate_sha256']==sha(base) and j['complete_surface_audit'] and j['exact_PLC_surface_pairs']==0 and j['inside_node_bank_pairs']==406
assert s['candidate_sha256']==sha(base) and s['complete'] and len(s['banks'])==185 and sum((b.get('result')or{}).get('count',0)for b in s['banks'])==3268
prep=D/'D/connected-continuation-prep-412/run-001/preparation-report.json';r=json.load(open(prep))
assert r['candidate_sha256']==sha(base) and r['status']=='PASS_CONSTRUCTION_INPUTS_ONLY' and r['native_admission']is False and r['accepted_physical_steps']==0
assert r['335']['passing_targets']==3268 and r['390']['passing_targets']==406
for k,n in [('--face-target-npz','335'),('--point-target-npz','390')]:assert sha(arg(k))==r[n]['npz_sha256']
assert sha(arg('--point-depth-rows-npz'))=='0a58ac13a63afb427f577f66c14cc5f05a1476fb1ec9a0c301bee6f45f92e7fe'
assert sha(arg('--point-depth-rows-report'))=='990f37f5f2e8c526e227528642e135cf0752f8dceba590ee334e6cda4b51e8ec'
# Additional strengthened seed/guard artifacts are bound in draft argv before execution.
seedp=Path(arg('--self-pairs'));seedj=json.load(open(seedp.parent/'report.json'))
assert seedj['candidate_sha256']==sha(base) and seedj['output_pairs_sha256']==sha(seedp)
assert seedj['zero_step_feasible'] is True and seedj['zero_step_max_rhs_mm']<=2e-8
assert sha(seedp)=='3ed3d20a59c462e0d0d2c1476b20ba69090be1e58b3d9d27379701eb66ba4079'
assert len(np.load(seedp))==18092
assert sha(arg('--fixed-boundary-npz'))=='38461a7c4164c9d89529bcbb41ec5f2e0fd89245fce2261248320d81f6b6ba93'
assert sha(arg('--fixed-boundary-report'))=='a554d30df694d153203b982d4acb084059134a253cff18e1d28963141423cdbd'
g=json.load(open(arg('--fixed-boundary-report')));assert g['base_candidate_sha256']==sha(base)
for k,v in {'--trust-mm':'1','--source-cap-mm':'12','--area-floor':'0.1','--max-edge-gradient':'0.5','--fixed-prefix-nodes':'0','--time-limit-s':'90','--retained-gain-fraction':'0.95'}.items():assert arg(k)==v
for f in ['--immediate-parent-basis','--fine-point-controls','--fine-surface-controls']:assert f in a
z=np.load(base);assert np.array_equal(z['source_O_faces'],z['area_reference_037_source_O_faces'])
paths={a[0],a[1],str(base.parent/'audit-130/summary.json'),str(base.parent/'audit-159/report.json'),str(prep),str(seedp.parent/'report.json')}
for v in a[2:]:
 if v.startswith('/Users/') and v!=arg('--output'):paths.add(v)
for n in ['fine_point_basis_405.py','surface_witness_basis_408.py','precision_phase_one_integration_396.py']:paths.add(str(D/'D/surface-witness-basis-408'/n))
paths.add(str(D/'D/connected-fixed-boundary-314/emit_introduced_skin_bone_barriers_314.py'))
manifest={p:dict(sha256=sha(p),bytes=Path(p).stat().st_size)for p in sorted(paths)}
(P/'argv.json').write_text(json.dumps(a,indent=2)+'\n')
out=dict(status='PASS_FROZEN_413_INPUT_AND_OWNER_PREFLIGHT',argv_sha256=sha(P/'argv.json'),argv=a,input_manifest=manifest,candidate_sha256=sha(base),self_seed_count=len(np.load(seedp)),surface_targets=3268,inside_targets=406,typed_anchor_count=len(z['typed_coarse_anchor_output_rows_018']),matched037_faces=True,native_admission=False,accepted_physical_steps=0)
(P/'preflight.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps({k:v for k,v in out.items()if k not in ['argv','input_manifest']}))

