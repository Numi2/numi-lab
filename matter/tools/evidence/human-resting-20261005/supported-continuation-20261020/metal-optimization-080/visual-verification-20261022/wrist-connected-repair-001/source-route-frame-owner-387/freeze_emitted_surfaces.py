from pathlib import Path
import contextlib,hashlib,io,json,runpy,os
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
OUT=E/'source-route-frame-owner-387'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with contextlib.redirect_stdout(io.StringIO()): e=runpy.run_path(str(OUT/'current_pose_preflight.py'),run_name='freeze_emissions')
assert e['report']['baseline_match']['per_bank_submicron']
arrays={};rows=[]
for sid in sorted(e['WANTED']):
 bV,bF,_=e['base'][sid];cV,cF,_=e['cand'][sid]
 assert np.array_equal(bF,cF)
 arrays[f'base_{sid}_vertices']=np.asarray(bV,np.float32)
 arrays[f'candidate_{sid}_vertices']=np.asarray(cV,np.float32)
 arrays[f'{sid}_faces']=np.asarray(bF,np.uint32)
 rows.append({'stable_id':sid,'vertices':len(bV),'faces':len(bF),'baseline_xyz_sha256':hashlib.sha256(np.asarray(bV,np.float32).tobytes()).hexdigest(),'candidate_xyz_sha256':hashlib.sha256(np.asarray(cV,np.float32).tobytes()).hexdigest(),'faces_sha256':hashlib.sha256(np.asarray(bF,np.uint32).tobytes()).hexdigest()})
p=OUT/'emitted-surfaces-identity003-current-191.npz';np.savez_compressed(p,**arrays)
receipt={'schema':'numi.human.nhtiss.route-weight-counterfactual.emitted-surfaces.v1','candidate_receipt_sha256':sha(E/'source-route-frame-owner-386/candidate-receipt.json'),'candidate_payload_sha256':sha(E/'source-route-frame-owner-386/corrected-route-weights-candidate.nhtissue'),'baseline_payload_sha256':sha(Path(e['BASE'])),'preflight_builder_sha256':sha(OUT/'current_pose_preflight.py'),'identity003_sha256':sha(e['D']/'registered-skin-poses.json'),'registered191_cache_sha256':sha(E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz'),'baseline_match':e['report']['baseline_match'],'array_path':str(p),'array_sha256':sha(p),'rows':rows,'scope':'Exact 191-order Float32 current-pose emitted XYZ and unchanged face indices for source and 386 candidate. The exact test stopped after stable 131 self-intersection failure; no uncompleted bank is represented as gated.'}
rp=OUT/'emitted-surfaces-identity003-current-191.receipt.json';rp.write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'array_path':str(p),'array_sha256':sha(p),'receipt_path':str(rp),'receipt_sha256':sha(rp),'banks':len(rows),'baseline_match':receipt['baseline_match']},indent=2))
