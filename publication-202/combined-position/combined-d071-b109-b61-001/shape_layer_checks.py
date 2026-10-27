from pathlib import Path
import hashlib,importlib.util,json,sys
import numpy as np
ROOT=Path('/Users/numi/Work/numi-human-connected-surface-069')
S=ROOT/'D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027'
OUT=Path(__file__).resolve().parent
D071=S/'root-topology-a69-safe-subset-001/candidate.npz'
CAND=OUT/'candidate.npz'
D071_LAYER=S/'root-topology-a69-safe-subset-001/layer-diagnostic/skin-fat-layers-541.npz'
A69_LAYER=ROOT/'D/publication-200-stage-001/evidence/layer/guarded-65-quarter/skin-fat-layers-541.npz'
FLIP_REPORT=S/'root-topology-a69-safe-subset-001/layer-identity-comparison.json'
HELPER=S/'b109-114/attempt-022-a69-rebase-checks-001/rebase_and_screen.py'
LAYER_OUT=OUT/'skin-fat-layers-541.npz'

def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def load(p):
 with np.load(p,allow_pickle=False) as z:return {k:z[k].copy() for k in z.files}
def metrics(T):
 T=np.asarray(T,dtype=np.float64);u=T[:,1]-T[:,0];v=T[:,2]-T[:,0];w=T[:,2]-T[:,1]
 N=np.cross(u,v);tw=np.linalg.norm(N,axis=1);L=np.stack([np.linalg.norm(u,axis=1),np.linalg.norm(v,axis=1),np.linalg.norm(w,axis=1)],axis=1)
 angles=[]
 for i in range(3):
  a=T[:,(i+1)%3]-T[:,i];b=T[:,(i+2)%3]-T[:,i]
  q=np.einsum('ij,ij->i',a,b)/np.maximum(np.linalg.norm(a,axis=1)*np.linalg.norm(b,axis=1),1e-300)
  angles.append(np.degrees(np.arccos(np.clip(q,-1,1))))
 return {'area':tw*.5,'angle':np.min(np.stack(angles,axis=1),axis=1),'aspect':L.max(axis=1)**2/(2*np.sqrt(3)*np.maximum(tw,1e-300)),'normal':N}
def shape_rows(P,Y,F,rows):
 a=metrics(P[F[rows]].astype(np.float64));b=metrics(Y[F[rows]].astype(np.float64))
 ar=b['area']/np.maximum(a['area'],1e-300);ang=b['angle']/np.maximum(a['angle'],1e-300);asp=b['aspect']/np.maximum(a['aspect'],1e-300)
 dot=np.einsum('ij,ij->i',a['normal'],b['normal'])/np.maximum(np.linalg.norm(a['normal'],axis=1)*np.linalg.norm(b['normal'],axis=1),1e-300)
 rot=np.degrees(np.arccos(np.clip(dot,-1,1)));bad=(ar<.95)|(ang<.95)|(asp>1.05)|(dot<=0)
 return {'faces_checked':int(len(rows)),'violating_faces':int(bad.sum()),'minimum_area_ratio':float(ar.min()),'minimum_angle_ratio':float(ang.min()),'maximum_aspect_ratio':float(asp.max()),'minimum_normal_dot':float(dot.min()),'maximum_normal_rotation_degrees':float(rot.max()),'thresholds':{'area_ratio_floor':.95,'minimum_angle_ratio_floor':.95,'aspect_ratio_ceiling':1.05,'positive_orientation':True}}

d=load(D071);c=load(CAND);A=load(ROOT/'D/publication-200-stage-001/evidence/retained/candidate.npz')
if not np.array_equal(d['source_O_faces'],c['source_O_faces']):raise RuntimeError('merged topology differs from d071')
for k in d:
 if k!='points_world_m' and (k not in c or c[k].dtype!=d[k].dtype or c[k].shape!=d[k].shape or c[k].tobytes()!=d[k].tobytes()):raise RuntimeError('merged nonposition metadata differs from d071: '+k)
P=np.asarray(d['points_world_m'],dtype=np.float32);Y=np.asarray(c['points_world_m'],dtype=np.float32);F=np.asarray(d['source_O_faces'],dtype=np.int64)
changed_nodes=np.flatnonzero(np.any(P.view(np.uint32)!=Y.view(np.uint32),axis=1))
changed_faces=np.flatnonzero(np.any(d['source_O_faces']!=A['source_O_faces'],axis=1))
affected=np.flatnonzero(np.any(np.isin(F,changed_nodes),axis=1))
affected=np.union1d(affected,changed_faces).astype(np.int64)
anchors=np.asarray(d['typed_coarse_anchor_output_rows_018'],dtype=np.int64)
maxmove=float(np.linalg.norm(Y.astype(np.float64)-P.astype(np.float64),axis=1).max()*1000)
shape={'status':'SHAPE_CHECK_COMPLETE','baseline_candidate_sha256':sha(D071),'candidate_sha256':sha(CAND),'topology_matches_d071':True,'changed_node_count':int(len(changed_nodes)),'changed_d071_connectivity_face_rows':changed_faces.astype(int).tolist(),'changed_connectivity_faces_included_in_shape_rows':True,'affected_face_count':int(len(affected)),'affected_faces_vs_d071':shape_rows(P,Y,F,affected),'12_flip_faces_vs_d071':shape_rows(P,Y,F,changed_faces),'max_displacement_from_d071_mm':maxmove,'anchors_bitwise_unchanged':bool(np.array_equal(P[anchors].view(np.uint32),Y[anchors].view(np.uint32))),'nonposition_metadata_preserved_bytewise':True,'pass':False,'geometry_admitted':False,'physical_steps':0}
s=shape['affected_faces_vs_d071'];shape['pass']=bool(s['violating_faces']==0 and shape['anchors_bitwise_unchanged'] and shape['nonposition_metadata_preserved_bytewise'] and maxmove<=3.000001)
# Propagate the exact Float32 outer displacement through d071's 541 layers.
layer0=load(D071_LAYER)
if not np.array_equal(layer0['outer_points_f64'],P.astype(np.float64)):raise RuntimeError('d071 layer outer points mismatch d071 candidate')
if not np.array_equal(layer0['source_outer_faces_raw'],F):raise RuntimeError('d071 layer face topology mismatch')
if not np.array_equal(layer0['outer_faces'],F[:,[0,2,1]]):raise RuntimeError('d071 layer winding/connectivity mismatch')
dy=Y.astype(np.float64)-P.astype(np.float64);layer1={k:v.copy() for k,v in layer0.items()}
for key in ('outer_points_f64','inner_points_f64','deep_points_f64'):layer1[key]=np.asarray(layer0[key],dtype=np.float64)+dy
for key in ('outer_points_f32','inner_points_f32','deep_points_f32'):layer1[key]=layer1[key.replace('_f32','_f64')].astype(np.float32)
layer1['skin_thickness_actual_norm_m']=np.linalg.norm(layer1['inner_points_f64']-layer1['outer_points_f64'],axis=1)
layer1['fat_thickness_actual_norm_m']=np.linalg.norm(layer1['deep_points_f64']-layer1['inner_points_f64'],axis=1)
with LAYER_OUT.open('xb') as f:np.savez_compressed(f,**layer1)
spec=importlib.util.spec_from_file_location('b109_layer_failure_helper',HELPER);mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod;spec.loader.exec_module(mod)
layer_delta={}
for label,ok,ik,fk in [('skin','outer_points_f64','inner_points_f64','skin_thickness_transferred_m'),('fat','inner_points_f64','deep_points_f64','fat_thickness_transferred_m')]:
 old=mod.layer_failures(layer0,ok,ik,fk);new=mod.layer_failures(layer1,ok,ik,fk);layer_delta[label]={}
 for name in old:
  added=new[name]-old[name];removed=old[name]-new[name]
  face_ids=lambda ids:sorted(set(int(x) if name=='authored_orientation' else int(x)//3 for x in ids))
  flipset=set(map(int,changed_faces))
  layer_delta[label][name]={'baseline_failure_count':len(old[name]),'candidate_failure_count':len(new[name]),'added_count':len(added),'removed_count':len(removed),'added_ids':sorted(map(int,added)),'removed_ids':sorted(map(int,removed)),'added_on_12_connectivity_faces':sorted(set(face_ids(added))&flipset),'removed_on_12_connectivity_faces':sorted(set(face_ids(removed))&flipset),'baseline_failures_on_12_connectivity_faces':sorted(set(face_ids(old[name]))&flipset),'candidate_failures_on_12_connectivity_faces':sorted(set(face_ids(new[name]))&flipset)}
added_total=sum(v['added_count'] for layer in layer_delta.values() for v in layer.values())
layer={'status':'LAYER_IDENTITY_CHECK_COMPLETE','baseline_d071_layer_path':str(D071_LAYER),'baseline_d071_layer_sha256':sha(D071_LAYER),'a69_layer_path':str(A69_LAYER),'a69_layer_sha256':sha(A69_LAYER),'candidate_layer_path':str(LAYER_OUT),'candidate_layer_sha256':sha(LAYER_OUT),'topology_preserved_from_d071':True,'d071_changed_connectivity_face_rows':changed_faces.astype(int).tolist(),'checked_all_layer_faces':int(len(F)),'candidate_outer_matches_combined_candidate':bool(np.array_equal(layer1['outer_points_f32'].view(np.uint32),Y.view(np.uint32))),'per_layer_failure_identity_delta_vs_d071':layer_delta,'added_failure_count':int(added_total),'no_new_failure_identities':added_total==0,'geometry_admitted':False,'physical_steps':0}
report={'schema':'numi.human.combined.d071.b109_114.b61_97.shape-layer.v1','status':'SHAPE_AND_LAYER_CHECKED','shape':shape,'layer':layer,'inherited_d071_12_face_layer_comparison_path':str(FLIP_REPORT),'inherited_d071_12_face_layer_comparison_sha256':sha(FLIP_REPORT),'inherited_d071_12_face_layer_comparison':json.loads(FLIP_REPORT.read_text()),'geometry_admitted':False,'physical_steps':0}
(OUT/'shape-layer-report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'shape_pass':shape['pass'],'changed_nodes':len(changed_nodes),'affected_faces':len(affected),'flip_faces':len(changed_faces),'min_area':s['minimum_area_ratio'],'min_angle':s['minimum_angle_ratio'],'max_aspect':s['maximum_aspect_ratio'],'min_normal_dot':s['minimum_normal_dot'],'max_move_mm':maxmove,'layer_added':added_total,'layer_no_new':layer['no_new_failure_identities'],'report_sha256':sha(OUT/'shape-layer-report.json'),'layer_candidate_sha256':sha(LAYER_OUT)},indent=2))
