import sys,json,hashlib,importlib.util
from pathlib import Path
from fractions import Fraction
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069');S=ROOT/'D/surface-clearance-546/component-sat-selector-638';E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
helper_path=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py');exact_path=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py';sys.path.insert(0,str(E/'incremental-exact-audit-owner-253/src'))
def module(name,path):
 spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
ex=module('numilab_human.cardiac_cavity_intersections',exact_path);h=module('bankhelper',helper_path)
base=S/'edge-refinement-661-threebank-001/generation-002-all-banks-001/candidate.npz';C=ROOT/'D/publication-182-stage-001/combined-653/candidate.npz';W=json.load(open(S/'per-bank-workers-20261027/b74-104/baseline-witnesses.json'))
Z=np.load(C,allow_pickle=False);B0=np.load(base,allow_pickle=False);X=Z['points_world_m'].astype(np.float32);F=Z['source_O_faces'].astype(np.int64)
assert X.shape==B0['points_world_m'].shape and F.shape==B0['source_O_faces'].shape
assert np.array_equal(F,B0['source_O_faces'])
assert np.array_equal(X[np.array([19094,21136,19092,20914,12694,20915])],B0['points_world_m'][np.array([19094,21136,19092,20914,12694,20915])])
assert np.array_equal(Z['refinement_face_parent_face_row'][22587:22591],B0['refinement_face_parent_face_row'][22587:22591])
assert hashlib.sha256(C.read_bytes()).hexdigest()=='7f2cb323550b273ae772ffc92d1dd714ab047d7f973af7f60793367fe43222af'
EVID=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight';T,ids,gids,bodies,M,off,pos,keys,pins,_=h.read_bank_inputs(EVID,E);mask=(ids[:,0]==74)&(ids[:,1]==104);Bt=T[mask];bid=ids[mask,3]
local=(X.astype(np.float64)-pos[74])@M[74].T-off[74]
bank_records=[]
for i,t in enumerate(Bt):
 tri=tuple(tuple(Fraction.from_float(float(x)) for x in q) for q in t)
 bank_records.append((tri,tuple(min(q[a] for q in tri) for a in range(3)),tuple(max(q[a] for q in tri) for a in range(3)),int(bid[i]),(3*i,3*i+1,3*i+2)))
prep=ex.prepare_point_location(bank_records)
loop=[19094,21136,19092,20914,12694,20915]
locations=[]
for n in loop:
 loc=ex.point_location_prepared(tuple(Fraction.from_float(float(x)) for x in local[n]),prep)['location']
 locations.append({'node':int(n),'location':loc,'world_f32_m':X[n].tolist()})

def orient2(a,b,c):return ex._orient2(a,b,c)
def point_on_seg2(p,a,b):return orient2(a,b,p)==0 and all(min(a[k],b[k])<=p[k]<=max(a[k],b[k]) for k in (0,1))
def seg_intersect2(a,b,c,d):
 o1,o2,o3,o4=orient2(a,b,c),orient2(a,b,d),orient2(c,d,a),orient2(c,d,b)
 if o1==0 and point_on_seg2(c,a,b):return True
 if o2==0 and point_on_seg2(d,a,b):return True
 if o3==0 and point_on_seg2(a,c,d):return True
 if o4==0 and point_on_seg2(b,c,d):return True
 return (o1>0) != (o2>0) and (o3>0) != (o4>0)
def seg_hits_tri(p,q,t):
 n=ex._cross(ex._sub(t[1],t[0]),ex._sub(t[2],t[0]))
 dp=ex._dot(n,ex._sub(p,t[0]));dq=ex._dot(n,ex._sub(q,t[0]))
 if dp==0 and ex._inside(ex._signs(p,1,t,n)):return [tuple(p)]
 if dq==0 and ex._inside(ex._signs(q,1,t,n)):return [tuple(q)]
 if dp*dq<0:return ex._segment_triangle(p,q,t,n)
 if dp!=0 or dq!=0:return []
 axes=ex._axes(n);a=tuple(t[0][k] for k in axes);b=tuple(t[1][k] for k in axes);c=tuple(t[2][k] for k in axes);u=tuple(p[k] for k in axes);v=tuple(q[k] for k in axes)
 if ex._inside([orient2(a,b,u),orient2(b,c,u),orient2(c,a,u)]):return [tuple(p)]
 if ex._inside([orient2(a,b,v),orient2(b,c,v),orient2(c,a,v)]):return [tuple(q)]
 tri2=[a,b,c]
 for i in range(3):
  c0,d0=tri2[i],tri2[(i+1)%3]
  if seg_intersect2(u,v,c0,d0):return [tuple(p),tuple(q)]
 return []
edge_results=[]
for i,a0 in enumerate(loop):
 b0=loop[(i+1)%len(loop)];p=tuple(Fraction.from_float(float(x)) for x in local[a0]);q=tuple(Fraction.from_float(float(x)) for x in local[b0]);hits=[]
 for j,t in enumerate(bank_records):
  if any(max(p[k],q[k])<t[1][k] or min(p[k],q[k])>t[2][k] for k in range(3)):continue
  pts=seg_hits_tri(p,q,t[0])
  if pts:hits.append({'bank_triangle':int(bid[j]),'intersection_vertices_body74_local_m':[[float(x) for x in v] for v in pts]})
 edge_results.append({'edge_nodes':[int(a0),int(b0)],'intersection_count':len(hits),'witnesses':hits})
# Previous report had one Steiner vertex sample followed by six fan-face centroid samples.
old=json.load(open(S/'per-bank-workers-20261027/b74-104/steiner-remesh-report.json'))
clarification={'previous_trial_inside_samples_are':'4 of 6 fan-face centroids (the first sample was the Steiner vertex; it was outside); they were not boundary vertices','previous_trial_sample_order':'[Steiner vertex, centroid of fan face rows 22587, 22588, 22589, 22590, 42379, 42380]','previous_trial_sample_locations':old['fan_test_point_locations'],'fixed_boundary_vertices':locations,'fixed_boundary_vertices_all_outside':all(x['location']=='outside' for x in locations),'fixed_boundary_edges':edge_results,'fixed_boundary_edges_all_disjoint_from_bone':all(e['intersection_count']==0 for e in edge_results),'candidate_653_path':str(C),'candidate_653_sha256':hashlib.sha256(C.read_bytes()).hexdigest(),'candidate_653_compatible_with_refined_baseline':True,'compatibility_basis':'same points and face shapes; exact source_O_faces equality; identical coordinates for six boundary nodes; identical immediate parent-face rows for source face children; candidate combination report lists baseline refined candidate d367 and no target-bank changed nodes'}
# Verify source-boundary exact witness points from original crossing data are strictly inside the parent face.
ancestry=np.load(S/'edge-refinement-661-threebank-001/generation-002-all-banks-001/candidate.npz',allow_pickle=False)['refinement_face_parent_barycentric'];cross=[]
for r in W['crossing_pairs']:
 fr=int(r['skin_child_face_row']);tri=np.array(r['skin_body74_local_m']);A=np.column_stack((tri[1]-tri[0],tri[2]-tri[0]));parent_map=ancestry[fr]
 for q in r['intersection_vertices_body74_local_m']:
  uv=np.linalg.lstsq(A,np.asarray(q)-tri[0],rcond=None)[0];lam=np.asarray([1-uv.sum(),*uv]);pb=lam@parent_map;cross.append(pb)
mins=[float(np.min(x)) for x in cross]
clarification['baseline_intersection_witnesses_all_strictly_inside_source_face']=all(x>0 for x in mins)
clarification['minimum_source_face_barycentric_coordinate_over_witnesses']=min(mins)
clarification['bank_input_pins']=pins
O=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b74-104/steiner-multinode-001');p=O/'boundary-feasibility.json';p.write_text(json.dumps(clarification,indent=2)+'\n');print(json.dumps(clarification,indent=2))
