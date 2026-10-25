"""Static native source query of proposed rest geometry; NEVER physical admission."""
from pathlib import Path
import importlib.util,csv,json,hashlib,subprocess,sys,time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
ORIG=E/'bone-clearance-fit-040-shared-columns/native-source-route-requery-001/run_candidate_query_040.py'
spec=importlib.util.spec_from_file_location('frozen040',ORIG);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
mod.validate_pins()
C=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-volume-continuation-041/iteration-004/candidate.npz');REPORT=C.with_name('report.json')
assert mod.sha(C)=='a75ec9a5b566a8568e8f64dc9fe7b2af3cab15125b3243a58c6719f014055c56'
z=np.load(C,allow_pickle=False);X=z['points_world_f32'];T=z['tetrahedra']
ap=np.load(E/'bone-clearance-feasibility-033/world-bone-clearance-constraints.npz',allow_pickle=False)
assert X.dtype==np.float32 and X.shape==(7233,3) and np.isfinite(X).all()
assert np.array_equal(T,ap['tetrahedra'])
V=X.astype(float)[T];D=np.stack([V[:,i]-V[:,0] for i in(1,2,3)],axis=2);det=np.linalg.det(D)
e2=np.max(np.stack([np.sum((V[:,i]-V[:,j])**2,axis=1) for i,j in[(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]],axis=1),axis=1);quality=det/e2**1.5
assert np.all(det>0) and quality.min()>=1e-5
base=list(csv.DictReader(mod.BASE_NODES.open()))
assert np.array_equal(ap['global_nodes'],np.array([int(r['global_node']) for r in base]))
OUT=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001/connected-volume-continuation-041/iteration-004/native-query');assert not OUT.exists();OUT.mkdir()
override=OUT/'candidate-node-override.csv'
with override.open('x',newline='') as f:
 w=csv.writer(f,lineterminator='\n');w.writerow(('global_node','layer','row','x_m','y_m','z_m','mass_kg'))
 for r,x in zip(base,X):w.writerow((r['global_node'],r['layer'],r['row'],*(format(float(v),'.9g') for v in x),r['mass_kg']))
rows=list(csv.DictReader(override.open()))
assert np.array_equal(np.array([[float(r[c+'_m']) for c in 'xyz'] for r in rows],np.float32).view(np.uint32),X.view(np.uint32))
manifest=json.loads(mod.BASE.read_text());env=dict(manifest['environment']);env.pop('NUMI_HUMAN_INITIAL_SKIN_BONE_WITNESS_CSV',None)
env.update({'NUMI_HUMAN_D37_CANDIDATE_NODE_OVERRIDE_CSV':str(override),'NUMI_HUMAN_D37_CANDIDATE_NODE_OVERRIDE_SHA256':mod.sha(override),'NUMI_HUMAN_INITIAL_SKIN_BONE_ALL_BANK_CSV':str(OUT/'source-preflight/all-bank-containment.csv'),'NUMI_HUMAN_D37_EXACT_EMITTED_BANK_DUMP_PREFIX':str(OUT/'source-preflight/exact-emitted')})
manifest['environment']=env;manifest['scope']='connected continuation static geometry STATIC source query ONLY: no reference/material/mass admission, no physical steps'
mf=OUT/'base-launch-manifest.json';mf.write_text(json.dumps(manifest,indent=2)+'\n')
command=[sys.executable,str(mod.CHECKED),'--base-manifest',str(mf),'--app',str(mod.APP),'--runtime',str(mod.RUNTIME),'--metallib',str(mod.METALLIB),'--run-dir',str(OUT/'source-preflight'),'--source-root',str(mod.APPS),'--mode','source-preflight','--static-bvh','--timeout','180']
(OUT/'command.json').write_text(json.dumps(command,indent=2)+'\n')
start=time.monotonic();proc=subprocess.run(command,text=True,capture_output=True);elapsed=time.monotonic()-start
(OUT/'checked-runner.stdout.txt').write_text(proc.stdout);(OUT/'checked-runner.stderr.txt').write_text(proc.stderr)
receipt=OUT/'source-preflight/execution-receipt.json';rr=json.loads(receipt.read_text()) if receipt.exists() else {}
result={'schema':'numi.human.040-candidate-source-query.v1','scope':manifest['scope'],'source_only':True,'physical_qualification':False,'runner_returncode':proc.returncode,'elapsed_s':elapsed,'candidate_sha256':mod.sha(C),'candidate_report_sha256':mod.sha(REPORT),'wrapper_sha256':mod.sha(Path(__file__)),'original_runner_sha256':mod.sha(ORIG),'native_app_sha256':mod.sha(mod.APP),'runtime_sha256':mod.sha(mod.RUNTIME),'metallib_sha256':mod.sha(mod.METALLIB),'override_csv_sha256':mod.sha(override),'input_global_id_owner':'035 baseline + AP exact row identity; mass CSV retained ONLY for static query metadata','minimum_q_f32':float(quality.min()),'nonpositive_tets':0,'accepted_physical_steps':0,'execution_receipt':rr,'status':'SOURCE_QUERY_FAILED_OUTPUTS_RETAINED'}
if proc.returncode==0:
 assert rr.get('status')=='SOURCE_ONLY_INTENTIONAL_STOP' and rr.get('physical_qualification') is False
 for name in('exact-emitted.triangles.csv','exact-emitted.bodies.csv'):
  f=OUT/'source-preflight'/name;assert mod.sha(f)==mod.PIN[mod.FROZEN/name]
 import os
 fresh=OUT/'source-preflight/exact-emitted.triangles.csv';retained=mod.FROZEN/'exact-emitted.triangles.csv'
 assert mod.sha(fresh)==mod.sha(retained)
 fresh.unlink();os.link(retained,fresh)
 result['status']='SOURCE_QUERY_PASS_BANKS_IDENTICAL_TO_039_NOT_PHYSICAL_ADMISSION'
 result['outputs']={p.name:{'sha256':mod.sha(p),'bytes':p.stat().st_size} for p in (OUT/'source-preflight').glob('*.csv')}
(OUT/'source-query-report.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in['execution_receipt','outputs']},indent=2))
if proc.returncode:print(proc.stderr[-2500:]);raise SystemExit(2)
