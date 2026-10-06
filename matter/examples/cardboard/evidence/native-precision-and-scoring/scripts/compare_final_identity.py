from pathlib import Path
import json,hashlib,csv
r=Path(__file__).parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(n,f):return json.loads((r/n/f).read_text())
a=read('barrier-exact1-compile-001','manifest.json');b=read('barrier-rounded1-compile-001','manifest.json')
assert a['compiled_world_fingerprint']==b['compiled_world_fingerprint']
assert a['tooling']['prescribed_tool_barrier_stiffness_scale']==b['tooling']['prescribed_tool_barrier_stiffness_scale']==1
assert b['tooling']['prescribed_tool_barrier_stiffness_scale_requested']==1.00000001
rows=list(csv.DictReader((r/'box-feature-rest-002/observations.csv').open()))
assert len(rows)==4
for row in rows:
 assert row['step_accepted']=='1' and all(float(row[k])==0 for k in ['max_node_displacement_m','max_free_speed_m_s','free_force_imbalance_l2_N','kinetic_energy_J'])
same={f:sha(r/'box-feature-rest-002'/f)==sha(r/'box-rest-001'/f) for f in ['mesh.json','initial.obj','observations.csv']}
assert all(same.values())
names=['barrier-exact1-compile-001','barrier-rounded1-compile-001','box-feature-rest-002']
receipts=[json.loads((r/'receipts'/n/'receipt.json').read_text()) for n in names]
assert all(x['inputs_unchanged'] and x['return_code']==0 for x in receipts)
report={'schema':'numi.cardboard.final-identity-comparison.v1','physical_validation':False,'same_native_bindings':all(x['bindings']==receipts[0]['bindings'] for x in receipts),'barrier_effective_scale':1,'requested_rounded_scale':b['tooling']['prescribed_tool_barrier_stiffness_scale_requested'],'same_world_fingerprint':a['compiled_world_fingerprint'],'box_rest_steps_per_environment':2,'box_exact_zero_motion_speed_force_and_energy':True,'box_byte_identical_to_prior':same,'scope':'Identity and zero-load rest checks, not assembly or contact calibration.','file_hashes':{n:{f:sha(r/n/f) for f in ['manifest.json','result.json']} for n in names}}
assert report['same_native_bindings']
with (r/'final-identity-comparison.json').open('x') as out:out.write(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
