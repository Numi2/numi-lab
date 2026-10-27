from pathlib import Path
from collections import defaultdict
import importlib.util,hashlib,json,os,sys
import numpy as np
from scipy.optimize import linprog
N=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546');ROOT=N.parent
base=N/'connected-topology-580/run-009-from772/lineage/candidate.npz'
audit=N/'connected-topology-580/run-009-from772/full-check/audit-159/report.json'
check=N/'connected-topology-580/run-009-from772/full-check/report.json'
runner=ROOT/'D/fixed-bone-outward-plane-458/run_grouped_fit_458.py'; objective=N/'pair-sat-routing-551/pair_sat_routing_551.py'
owner=ROOT/'surface-witness-basis-408/propose_joint_clearance_witnesses_408.py'
source=ROOT.parent/'full-connected-clearance-150/run-009-certified-source-normal-union-rays/endpoint-candidate.npz'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
os.environ['NUMI_FIXED_BONE_OUTWARD_CANDIDATE_SHA256_458']=sha(base);os.environ['NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458']=sha(audit)
def load(name,path):
 s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);sys.modules[name]=m;s.loader.exec_module(m);return m
r=load('probe458_node',runner);o=load('objective458_node',objective);own=load('owner408_node',owner)
c=r.prepare_crossing_inputs(base,audit); data=np.load(base,allow_pickle=False); ref,provenance=own.compose_source_reference(data,source)
a=json.loads(check.read_text())['current_anchor_lineage'];assert a['pass'] and a['anchors']==237
anchors=np.asarray(a['current_output_rows'],dtype=np.int64)
rows=defaultdict(list)
for pair in c['pairs']:
 face=int(pair['face_row']); nodes=c['faces'][face]
 p=o.choose_pair_sat_plane(c['points_mm'][nodes],c['bone_triangles_mm'][(pair['body'],pair['bone_id'],pair['triangle_id'])])
 for j,(node,h) in enumerate(zip(nodes,p['corner_lower_mm'])):
  rows[int(node)].append((p['normal_world_unit'],float(h),int(pair['body']),int(pair['bone_id']),int(pair['triangle_id']),face,j))
support=np.asarray(sorted(rows),dtype=np.int64)
lo,hi,bounds=own.source_safe_trust_bounds(c['points_mm'],ref*1000,support,anchors,4,12)
results=[]; single=[]
for i,node in enumerate(support):
 records=rows[int(node)]; normals=np.asarray([v[0] for v in records]); rhs=np.asarray([v[1] for v in records])
 maxproj=np.maximum(normals*lo[3*i:3*i+3],normals*hi[3*i:3*i+3]).sum(axis=1)
 for k in np.flatnonzero(rhs-maxproj>2e-7):
  single.append({'node':int(node),'body':records[k][2],'bone_id':records[k][3],'bone_triangle':records[k][4],'skin_face':records[k][5],'corner':records[k][6],'rhs_mm':float(rhs[k]),'normal':normals[k].tolist(),'bound_min_mm':lo[3*i:3*i+3].tolist(),'bound_max_mm':hi[3*i:3*i+3].tolist(),'max_possible_projection_mm':float(maxproj[k]),'minimum_deficit_mm':float(rhs[k]-maxproj[k])})
 A=np.column_stack((-normals,-np.ones(len(normals)))); b=-rhs
 res=linprog([0,0,0,1],A_ub=A,b_ub=b,bounds=[(lo[3*i+j],hi[3*i+j]) for j in range(3)]+[(0,None)],method='highs')
 if not res.success:raise RuntimeError('node feasibility LP failed: '+str(res.message))
 delta=res.x[:3];slack=float(res.x[3]); residual=normals@delta-rhs
 if slack>2e-7:
  violated=np.argsort(residual)[:min(8,len(records))]
  results.append({'node':int(node),'row_count':len(records),'minimax_deficit_mm':slack,'component_bound_mm':float(hi[3*i]),'min_residual_mm':float(residual.min()),'witnesses':[{'body':records[k][2],'bone_id':records[k][3],'bone_triangle':records[k][4],'skin_face':records[k][5],'corner':records[k][6],'rhs_mm':records[k][1],'residual_mm':float(residual[k])} for k in violated]})
results.sort(key=lambda x:(-x['minimax_deficit_mm'],x['node']))
out=N/'hard-face-feasibility-634/prepared-from766-001/node-sat-selected-feasibility.json'
x={'candidate':str(base),'candidate_sha256':sha(base),'audit_sha256':sha(audit),'check_sha256':sha(check),'458_runner_sha256':sha(runner),'551_sat_objective_sha256':sha(objective),'408_owner_sha256':sha(owner),'source_base_sha256':sha(source),'source_reference_provenance':provenance,'bounding_policy':bounds,'pairs':len(c['pairs']),'face_corner_rows':sum(map(len,rows.values())),'unique_nodes':len(rows),'single_row_infeasible_count':len(single),'single_row_infeasible_node_count':len({z['node'] for z in single}),'single_row_infeasible':single,'per_node_infeasible_count':len(results),'per_node_infeasible':results,'classification':'independent node LP lower bound for the finite-pair 551 minimum-deficit SAT-plane row formulation under the existing conservative trust/source cubes; omits coupled area, edge, self and other hard rows and does not prove physical impossibility','threshold_mm':2e-7,'native_admission':False,'accepted_physical_steps':0}
out.write_text(json.dumps(x,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({k:v for k,v in x.items() if k not in ('per_node_infeasible','single_row_infeasible','source_reference_provenance','bounding_policy')}))
