from pathlib import Path
import sys,json,ast,csv,hashlib,time,argparse
import numpy as np
from scipy.spatial import cKDTree
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
ap=argparse.ArgumentParser();ap.add_argument('--geometry-npz',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);args=ap.parse_args(); OUT=args.output;OUT.mkdir(exist_ok=False)
H={};exec((E/'wrist-loft-rebuild-019/audit_wrist_bones035.py').read_text().split("report={'schema'")[0],H)
sys.path.insert(0,str(Path(__file__).resolve().parent)); from bone_point_queries import closest; ns={'closest':closest}
P=next((E/'native-pose-checks/repair-005/neutral').glob('*.skin-poses.json'));pose=json.loads(P.read_text());L=E/'replacement-mechanical-070-jac-quality-allcomponents/joint-component-layers.npz';X=np.load(L)['outer_world'].copy();C=np.load(E/'replacement-mechanical-061-normal-cone-001/coarse-to-source.npz');S=np.load(E/'replacement-mechanical-061-normal-cone-001/coarse-surface.npz');G=np.load(args.geometry_npz);regions={int(r['coarse_row']):r for r in csv.DictReader((E/'replacement-mechanical-067-ancestry-regions/coarse-row-anatomy.csv').open())};used=np.unique(S['faces']);rows=used[S['vertex_kind'][used]!=0];sf=C['source_face'][rows];b=C['source_barycentric'][rows];X[rows]=np.einsum('ni,nij->nj',b,G['world'][G['faces'][sf]])+np.array([-.05,.20,2.0]);X=X.astype(np.float32).astype(float)
reference=np.load(E/'finger-profile-068-whole-hand-minimax/replacement-geometry.npz')
if G['world'].shape!=reference['world'].shape or not np.array_equal(G['faces'],reference['faces']) or not np.isfinite(G['world']).all():raise ValueError('candidate geometry topology/finite-coordinate contract failed')
distance=np.full(len(X),np.inf);closest=np.full_like(X,np.nan);triangle=np.full(len(X),-1,np.int64);owner=np.full(len(X),'',dtype='U64');counts=np.zeros(len(X),np.int64);checks=[];start=time.monotonic();sideReports={}
for side in ['left','right']:
 ids=np.array([int(i) for i in rows if regions[int(i)]['side']==side]);v,f,owners=H['bone_mesh'](pose,side);T=v.astype(float)[f]+np.array([-.05,.20,2.0]);owners=np.array(owners);cent=T.mean(1);radii=np.max(np.linalg.norm(T-cent[:,None,:],axis=2),axis=1);maxrad=float(radii.max());kt=cKDTree(cent)
 for row in ids:
  p=X[row];_,j=kt.query(p);d0,_=ns['closest'](p,T[int(j):int(j)+1]);radius=float(d0[0]+maxrad+1e-10);cand=np.array(sorted(kt.query_ball_point(p,radius)),int)
  # Any triangle closer than d0 has centroid distance <=d0+its radius <=radius.
  if not len(cand):raise RuntimeError('empty conservative triangle range')
  ds,qs=ns['closest'](p,T[cand]);k=int(np.argmin(ds));j=int(cand[k]);distance[row]=ds[k];closest[row]=qs[k];triangle[row]=j;owner[row]=owners[j];counts[row]=len(cand)
 verify=np.unique(np.r_[ids[np.linspace(0,len(ids)-1,32,dtype=int)],[i for i in [2749,3197,3780,3786,3796,3810,4863,5077,5351,5403] if i in set(ids)]])
 for row in verify:
  ds,_=ns['closest'](X[row],T);full=float(ds.min());err=abs(full-distance[row]);checks.append({'row':int(row),'absolute_error_m':err});assert err<1e-14
 sideReports[side]={'rows':len(ids),'bone_triangles':len(T),'maximum_triangle_centroid_radius_m':maxrad,'candidate_faces_p50':float(np.median(counts[ids])),'candidate_faces_p99':float(np.percentile(counts[ids],99)),'min_clearance_mm':float(distance[ids].min()*1000),'under_1mm_rows':ids[distance[ids]<.001].tolist()}
 print(json.dumps({'side':side,**sideReports[side]}),flush=True)
np.savez_compressed(OUT/'source-exact-point-clearance.npz',outer_world=X,active_new_rows=rows,point_bone_clearance_m=distance,closest_bone_point_world_m=closest,closest_bone_triangle=triangle,bone_owner=owner,candidate_face_count=counts,available_after_0p1mm_gap_m=distance-.0001)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
report={'status':'PASS_POINT_QUERY_NOT_FULL_LAYER_ADMISSION','method':'Distance to all same-side bone triangles using a conservative centroid candidate sphere: one triangle supplies upper bound d0; every possible closer triangle centroid lies within d0+maximum triangle centroid radius. Exact point-to-triangle projection within that set. Deterministic sorted triangle witnesses. Emitted Float32 current candidate source-face+bary outer, zero stale residual on new rows; kind0 baseline unchanged.','units':'metres','input_sha256':{'coarse':sha(E/'replacement-mechanical-061-normal-cone-001/coarse-surface.npz'),'mapping':sha(E/'replacement-mechanical-061-normal-cone-001/coarse-to-source.npz'),'geometry':sha(args.geometry_npz),'bone_payload':H['sha'](H['bonepath']),'pose':sha(P)},'rows':len(rows),'sides':sideReports,'all_triangle_brute_force_checks':checks,'elapsed_wall_s':time.monotonic()-start,'output_sha256':sha(OUT/'source-exact-point-clearance.npz'),'limits':['Unsigned point distances do not establish closed-solid containment, triangle or interior-layer clearance.','Only retained neutral pose screened; posed full layer/bone checks remain required.','No source geometry, physical material, force owner or native headers changed.']}
(OUT/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps({'output':str(OUT),'elapsed_s':report['elapsed_wall_s'],'rows':len(rows)}),flush=True)
