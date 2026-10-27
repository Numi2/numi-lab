from pathlib import Path
import hashlib,json,subprocess,time
OUT=Path(__file__).resolve().parent
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
CAND=OUT/'candidate.npz'
D071=ROOT/'D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/root-topology-a69-safe-subset-001/candidate.npz'
A69=ROOT/'D/publication-200-stage-001/evidence/retained/candidate.npz'
PY=Path('/Users/numi/numi-human-prep-venv-20261005/bin/python')
SLOT=Path('/tmp/numi-human-geometry-slot.py')
A130=ROOT/'audit_connected_outer_geometry_130.py'
A159=ROOT/'audit_skin_bone_surfaces_159.py'
LAUNCH=OUT/'full-audits-launch.json';EXEC=OUT/'full-audits-execution.json';LOG130=OUT/'audit-130.log';LOG159=OUT/'audit-159.log'
C130=OUT/'audit-130';C159=OUT/'audit-159'
EXPECTED_CAND='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def jread(p):return json.loads(Path(p).read_text())
if not CAND.is_file() or sha(CAND)!=EXPECTED_CAND:raise RuntimeError('candidate pin mismatch')
if LAUNCH.exists() or EXEC.exists() or C130.exists() or C159.exists():raise FileExistsError('unique audit outputs already exist')
inputs=[CAND,D071,A69,PY,SLOT,A130,A159]
before={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs}
launch={'schema':'numi.human.combined.d071.b109_114.b61_97.full-audit-launch.v1','candidate_sha256':sha(CAND),'d071_sha256':sha(D071),'a69_sha256':sha(A69),'source_pins':before,'commands':[[str(PY),str(SLOT),str(PY),str(A130),str(CAND),str(C130)],[str(PY),str(SLOT),str(PY),str(A159),str(CAND),str(C159)]],'geometry_admitted':False,'physical_steps':0}
LAUNCH.write_text(json.dumps(launch,indent=2,sort_keys=True)+'\n')
start=time.monotonic();runs=[]
for label,script,dest,log in [('audit-130',A130,C130,LOG130),('audit-159',A159,C159,LOG159)]:
 cmd=[str(PY),str(SLOT),str(PY),str(script),str(CAND),str(dest)]
 t=time.monotonic()
 with log.open('wb') as f:cp=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT)
 rec={'name':label,'argv':cmd,'returncode':cp.returncode,'elapsed_seconds':time.monotonic()-t,'log_sha256':sha(log),'log_path':str(log)}
 if label=='audit-130' and (dest/'summary.json').is_file():
  d=jread(dest/'summary.json');rec['summary_path']=str(dest/'summary.json');rec['summary_sha256']=sha(dest/'summary.json');rec['summary']={k:d.get(k) for k in ('status','candidate_sha256','points','faces','exact_PLC_surface_pairs','complete_surface_audit','inside_nodes','inside_node_bank_pairs','query_pair_count','geometry_admitted','edge_and_solid_clearance_certified')};
 if label=='audit-159' and (dest/'report.json').is_file():
  d=jread(dest/'report.json');rec['report_path']=str(dest/'report.json');rec['report_sha256']=sha(dest/'report.json');rec['report']={k:d.get(k) for k in ('status','candidate_sha256','surface_point_count','surface_face_count','bank_count','expected_bank_count','complete','geometry_or_native_admission','elapsed_seconds')}
 runs.append(rec)
after={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in inputs}
result={'schema':'numi.human.combined.d071.b109_114.b61_97.full-audit-execution.v1','status':'AUDITS_RUN','candidate_sha256':sha(CAND),'runs':runs,'source_pins_unchanged':before==after,'source_pins_before':before,'source_pins_after':after,'elapsed_seconds':time.monotonic()-start,'geometry_admitted':False,'physical_steps':0}
EXEC.write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':result['status'],'candidate_sha256':result['candidate_sha256'],'runs':runs,'source_pins_unchanged':result['source_pins_unchanged'],'execution_receipt':str(EXEC),'execution_sha256':sha(EXEC)},indent=2))
if any(r['returncode'] for r in runs):raise SystemExit(1)
