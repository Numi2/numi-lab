import sys,json,time,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
E=Path(__file__).resolve().parent
ns={};exec((E/'audit_bilateral_skin_reweighted_candidate_016.py').read_text().split('def main():')[0],ns)
OUT=Path(sys.argv[1]);audit_name=sys.argv[sys.argv.index("--audit-name")+1] if "--audit-name" in sys.argv else "connected-hands-full-audit.json";asset=OUT/"replacement-bilateral-hands038.nhskin";payload=asset.read_bytes();d=ns["decode_payload"](payload);F=np.array(d["indices"],int).reshape(-1,3);build=json.loads((OUT/"build-report.json").read_text());actual=hashlib.sha256(np.asarray(d["indices"],dtype="<u4").tobytes()).hexdigest();assert hashlib.sha256(payload).hexdigest()==build["candidate_payload_sha256"];assert actual==build["global_index_sha256"];baseD=ns["decode_payload"]((E/"finger-profile-060"/"replacement-bilateral-hands038.nhskin").read_bytes());oldF=np.array(baseD["indices"],int).reshape(-1,3);assert np.array_equal(F,oldF);g=np.load(OUT/"replacement-geometry.npz");patches={s:np.load(OUT/(s+"-patch.npz")) for s in ("left","right")};bones_only="--bones-only" in sys.argv
h={};exec((E/'wrist-loft-rebuild-019/audit_wrist_bones035.py').read_text().split("report={'schema'")[0],h)
edges,cnt=np.unique(np.sort(np.vstack((F[:,[0,1]],F[:,[1,2]],F[:,[2,0]])),axis=1),axis=0,return_counts=True);oe,oc=np.unique(np.sort(np.vstack((oldF[:,[0,1]],oldF[:,[1,2]],oldF[:,[2,0]])),axis=1),axis=0,return_counts=True)
report={'status':'in_progress','skin_sha256':hashlib.sha256(asset.read_bytes()).hexdigest(),'scope':'Source registration and retained wrist-pose checks; not physical dynamics or breathing qualification.','preparation':{'finite_vertices':bool(np.isfinite(d['vertices_f'][:,:6]).all()),'valid_indices':bool(F.min()>=0 and F.max()<d['vertex_count']),'edge_incidence_max':int(cnt.max()),'boundary_edges':int(sum(cnt==1)),'baseline_boundary_edges':int(sum(oc==1)),'unchanged_baseline_boundary_edges':bool(np.array_equal(edges[cnt==1],oe[oc==1])),'maximum_weight_sum_error':float(np.max(abs(d['full_weights'].sum(1)-1))),'outside_patch_source_positions_unchanged':bool(np.array_equal(d['vertices_f'][np.setdiff1d(np.arange(d['vertex_count']),g['patch_vertices']),:3],baseD['vertices_f'][np.setdiff1d(np.arange(d['vertex_count']),g['patch_vertices']),:3]))},'poses':{}}
report["preparation"]["decoded_index_sha256"]=actual;report["preparation"]["exact_baseline060_indices"]=True;report["preparation"]["exact_build_report_payload_sha256"]=True
report['audit_inputs']={'entrypoint_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'geometry_sha256':hashlib.sha256((OUT/'replacement-geometry.npz').read_bytes()).hexdigest(),'build_report_sha256':hashlib.sha256((OUT/'build-report.json').read_bytes()).hexdigest(),'shared_surface_audit_sha256':hashlib.sha256((E/'audit_bilateral_skin_reweighted_candidate_016.py').read_bytes()).hexdigest(),'bone_audit_sha256':hashlib.sha256((E/'wrist-loft-rebuild-019/audit_wrist_bones035.py').read_bytes()).hexdigest(),'bone_payload_sha256':h['sha'](h['bonepath'])}
report['limits']=['Unsigned intersection audits are not solid-containment, physical layer or contact qualification.','The unchanged 115 head-height boundary edges remain open; this is not globally closed skin.']
def save(): (OUT/audit_name).write_text(json.dumps(report,indent=2)+'\n')
save();print('preparation',report['preparation'],flush=True)
pre=report['preparation']
if not (pre['finite_vertices'] and pre['valid_indices'] and pre['edge_incidence_max']<=2 and pre['unchanged_baseline_boundary_edges'] and pre['maximum_weight_sum_error']<=1e-5 and pre['outside_patch_source_positions_unchanged']):
 report['status']='failed_preparation';save();sys.exit(1)
posefiles=sorted((E/'native-pose-checks/repair-005').glob('*/*.skin-poses.json'),key=lambda p:('neutral' not in str(p),str(p)))
if '--pose-json' in sys.argv:
 posefiles=[Path(sys.argv[i+1]) for i,arg in enumerate(sys.argv) if arg=='--pose-json']
 report['scope']='Exact emitted Float32 surface at native accepted muscle-actuated endpoint poses; not continuous-path contact or full tissue mechanics qualification.'
 report_path=OUT/audit_name
 def save(): report_path.write_text(json.dumps(report,indent=2)+'\n')
 save()
for pf in posefiles:
 pp=json.loads(pf.read_text());Q=ns['posed'](d,pp).astype(np.float32);pts,qf=ns['prep'](Q,F);pr={'source_pose_sha256':hashlib.sha256(pf.read_bytes()).hexdigest(),'sides':{}};report['poses'][pf.parent.name]=pr
 for side in ('left','right'):
  ids=patches[side]['old_faces'];pr['sides'][side]={}
  if not bones_only:
   r=ns['pair_audit'](pts,qf,ids);r['edge_metrics']=ns['edge_metrics'](Q,F,ids);pr['sides'][side]={'self':r};print(pf.parent.name,side,'self',r['intersection_pairs'],flush=True);save()
   if r['intersection_pairs']:report['status']='failed_self';save();sys.exit(2)
   other=np.setdiff1d(np.arange(len(F)),ids);z=ns['pair_audit'](pts,qf,ids,other);pr['sides'][side]['surrounding_skin']=z;print(pf.parent.name,side,'outside',z['intersection_pairs'],flush=True);save()
   if z['intersection_pairs']:report['status']='failed_surrounding_skin';save();sys.exit(3)
  sf=F[ids];bv,bf,bn=h['bone_mesh'](pp,side);vv=np.vstack((Q,bv));ff=np.vstack((sf,bf+len(Q)));bp,bq=ns['prep'](vv,ff);br=h['pair_audit'](bp,bq,np.arange(len(sf)),np.arange(len(sf),len(ff)));br['crossing_bones']=dict(Counter(bn[b-len(sf)] for a,b in br['all_intersection_pairs']));pr['sides'][side]['bones']=br;print(pf.parent.name,side,'bones',br['intersection_pairs'],br['crossing_bones'],flush=True);save()
  if br['intersection_pairs']:
   report['status']='failed_bones';save()
   if not bones_only:sys.exit(4)
if report['status']!='failed_bones':report['status']='passed_retained_pose_bones_only_not_full_geometry' if bones_only else ('passed_native_actuated_endpoint_geometry_not_continuous_physical_qualification' if '--pose-json' in sys.argv else 'passed_source_and_retained_wrist_pose_checks_not_physical_qualification')
save()
