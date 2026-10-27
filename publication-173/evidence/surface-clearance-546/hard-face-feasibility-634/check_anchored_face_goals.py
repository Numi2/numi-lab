from pathlib import Path
import importlib.util,hashlib,json,os,sys
N=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546'); ROOT=N.parent
base=N/'connected-topology-580/run-009-from772/lineage/candidate.npz'
audit=N/'connected-topology-580/run-009-from772/full-check/audit-159/report.json'
check=N/'connected-topology-580/run-009-from772/full-check/report.json'
runner=ROOT/'D/fixed-bone-outward-plane-458/run_grouped_fit_458.py'
objective=runner.with_name('fixed_bone_outward_plane_458.py')
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
os.environ['NUMI_FIXED_BONE_OUTWARD_CANDIDATE_SHA256_458']=sha(base)
os.environ['NUMI_FIXED_BONE_OUTWARD_AUDIT159_SHA256_458']=sha(audit)
def load(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
r=load('fixed_bone458_anchor_probe',runner);o=load('fixed_bone458_anchor_objective',objective)
c=r.prepare_crossing_inputs(base,audit)
a=json.loads(check.read_text())['current_anchor_lineage'];assert a['pass'] and a['anchors']==237
anchors=set(map(int,a['current_output_rows']));bad=[];covered=0
for x in c['pairs']:
 face=int(x['face_row']);nodes=c['faces'][face];tri=c['points_mm'][nodes]
 goal=o.outward_plane_for_pair(tri,c['bone_triangles_mm'][(x['body'],x['bone_id'],x['triangle_id'])])
 for j,(node,h) in enumerate(zip(nodes,goal['corner_lower_mm'])):
  if int(node) in anchors:
   covered+=1
   if h>1e-12:bad.append({'body':x['body'],'bone_id':x['bone_id'],'triangle_id':x['triangle_id'],'skin_face_row':face,'corner_index':j,'anchor_node':int(node),'positive_lower_mm':float(h)})
out=N/'hard-face-feasibility-634/prepared-from766-001/anchored-positive-corners.json'
result={'candidate':str(base),'candidate_sha256':sha(base),'audit_sha256':sha(audit),'construction_check_sha256':sha(check),'runner_sha256':sha(runner),'objective_sha256':sha(objective),'pair_count':len(c['pairs']),'anchor_corner_occurrences':covered,'positive_anchored_corner_count':len(bad),'positive_anchored_corners':bad,'native_admission':False,'accepted_physical_steps':0}
out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ('positive_anchored_corners',)}))
