from pathlib import Path
from collections import Counter
import importlib.util,hashlib,json,os,sys
import numpy as np
N=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546'); ROOT=N.parent
base=N/'connected-topology-580/run-009-from772/lineage/candidate.npz';audit=N/'connected-topology-580/run-009-from772/full-check/audit-159/report.json';check=N/'connected-topology-580/run-009-from772/full-check/report.json'
runner=ROOT/'D/fixed-bone-outward-plane-458/run_grouped_fit_458.py';sat=N/'pair-sat-routing-551/pair_sat_routing_551.py';owner=ROOT/'surface-witness-basis-408/propose_joint_clearance_witnesses_408.py';source=ROOT.parent/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
os.environ['NUMI_FIXED_BONE_OUTWARD_CANDIDATE_SHA256_458']=sha(base);os.environ['NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458']=sha(audit)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
r=load('sat_pair_inputs',runner);o=load('sat_pair_axes',sat);own=load('sat_pair_bounds',owner); c=r.prepare_crossing_inputs(base,audit)
ref,prov=own.compose_source_reference(np.load(base,allow_pickle=False),source);an=json.loads(check.read_text())['current_anchor_lineage'];assert an['pass'] and an['anchors']==237
anchors=np.asarray(an['current_output_rows'],dtype=np.int64);support=np.unique(np.concatenate([c['faces'][int(p['face_row'])] for p in c['pairs']])); lo,hi,binfo=own.source_safe_trust_bounds(c['points_mm'],ref*1000,support,anchors,4,12);local={int(v):i for i,v in enumerate(support)}
res=[]; hist=Counter()
for pair in c['pairs']:
 nodes=c['faces'][int(pair['face_row'])];skin=c['points_mm'][nodes];bone=c['bone_triangles_mm'][(pair['body'],pair['bone_id'],pair['triangle_id'])];axes=o._signed_sat_axes(skin,bone); plausible=[]
 for axis in axes:
  n=np.asarray(axis['normal_world_unit']);bmax=np.max(bone@n);margin=o._float32_world_m_emission_margin_mm(skin,bone,n);rhs=bmax+margin-skin@n
  upper=np.array([np.maximum(n*lo[3*local[int(i)]:3*local[int(i)]+3],n*hi[3*local[int(i)]:3*local[int(i)]+3]).sum() for i in nodes]);gap=rhs-upper
  if float(np.max(gap))<=2e-7:plausible.append((float(np.max(np.maximum(rhs,0.0))),str(axis['axis_source']),int(axis['axis_index']),int(axis['axis_sign'])))
 hist[len(plausible)]+=1
 if not plausible:res.append({'face_row':int(pair['face_row']),'body':int(pair['body']),'bone_id':int(pair['bone_id']),'bone_triangle_id':int(pair['triangle_id']),'skin_nodes':list(map(int,nodes)),'tested_axes':len(axes)})
out=N/'hard-face-feasibility-634/prepared-from766-001/sat-axis-availability.json'
x={'candidate':str(base),'candidate_sha256':sha(base),'audit_sha256':sha(audit),'check_sha256':sha(check),'runner_sha256':sha(runner),'sat551_sha256':sha(sat),'sat466_sha256':o.SAT_SHA256,'408_owner_sha256':sha(owner),'source_base_sha256':sha(source),'source_reference_provenance':prov,'bounding_policy':binfo,'pair_count':len(c['pairs']),'unique_nodes':len(support),'pairs_without_box_feasible_candidate_axis':len(res),'axes_feasible_count_histogram':dict(sorted(hist.items())),'unavailable_pairs':res,'classification':'per-pair necessary and sufficient feasibility of three independent corner inequalities for one fixed SAT axis under 408 component boxes only; shared-node, area, edge, self, other-triangle constraints and actual triangle validity are not assessed','tolerance_mm':2e-7,'native_admission':False,'accepted_physical_steps':0}
out.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in x.items() if k not in ('unavailable_pairs','source_reference_provenance','bounding_policy')}))
