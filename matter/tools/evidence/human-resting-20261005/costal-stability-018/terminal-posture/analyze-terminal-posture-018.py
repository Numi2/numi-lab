from pathlib import Path
import csv,hashlib,importlib.util,json,sys,time
import numpy as np
E=Path('/Users/n/numi-human-resting-evidence-20261005');O=E/'terminal-posture-inspection-018';R=E/'costal-fit-stability-018'
sys.path.insert(0,'/Users/n/numi-human-resting-vascular-structure-source-013/matter/tools')
import accepted_mrvpack_surface_audit as a
D=E/'heart-swept-source-inverse-native017-001/actual-rib-pair-audit/optimized_driver-002.py'
spec=importlib.util.spec_from_file_location('exact_driver',D);d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
pred=a.predicate_module(d.PRED)
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pack0=R/'resting-human.mrvpack';pack1=O/'myosim-fullbody-articulated-bodyparts-bones-source-skinned-shell.mrvpack'
opened=[];meshes=[]
for p in [pack0,pack1]:
 raw,st,off,s=a.read_pack(p);opened.append((raw,st));f=np.asarray(s[(51007,1)]['faces']);lo=int(f.min());hi=int(f.max());v=np.ndarray((hi-lo+1,20),dtype='<f4',buffer=raw,offset=off+lo*80)[:,:3].copy();meshes.append((v,f-lo,raw,off,s[(51007,1)]['faces']))
assert np.array_equal(meshes[0][1],meshes[1][1])
v0,faces=meshes[0][:2];v1=meshes[1][0]
# Explicit source-space separation avoids classifying adjacent groin triangles as inter-limb contacts.
valid=v0[faces]
right=np.flatnonzero((valid[:,:,1]<-.92).all(1)&(valid[:,:,0]<-.025).all(1))
left=np.flatnonzero((valid[:,:,1]<-.92).all(1)&(valid[:,:,0]>-.025).all(1))
report={'qualification':'Read-only native Metal forward-kinematic reconstruction of exact terminal q; no physiology replay and no physical steps. Inter-limb subset is explicit and excludes the adjacent groin. A zero subset result would not imply whole-skin self clearance.', 'script_sha256':sha(__file__),'pack_sha256':{str(p):sha(p) for p in [pack0,pack1]},'source_log_sha256':sha(R/'native.log'),'predicate_sha256':sha(d.PRED),'pair_driver_sha256':sha(D),'partition':{'initial_supine_world_y_max_m':-.92,'initial_midline_x_m':-.025,'right_face_count':len(right),'left_face_count':len(left)},'frames':[]}
for name,m in zip(['initial','terminal320s'],meshes):
 v,f,raw,off,gf=m
 sets=[[gf[i] for i in ids] for ids in [right,left]]
 pts=[a.surface_points(raw,off,x) for x in sets];scale=max(x[2] for x in pts)
 records=[a.make_records(p,scale,f,pred) for p,f in zip(pts,sets)]
 t=time.monotonic();nc,hits=d.exact_all(records[0],records[1],pred)
 row={'frame':name,'aabb_candidates':nc,'exact_intersecting_triangle_pairs':len(hits),'complete':True,'elapsed_s':time.monotonic()-t,'minimum_skin_z_m':float(v[:,2].min()),'vertices_below_1mm':int((v[:,2]<-.001).sum()),'coordinate_scale_per_m':scale,'first_witnesses':hits[:16]};report['frames'].append(row);print(json.dumps({k:v for k,v in row.items() if k!='first_witnesses'}),flush=True)
state=json.loads(next(x.split('=',1)[1] for x in (R/'native.log').read_text().splitlines() if x.startswith('stand_terminal_state=')))
manifest=Path('/Users/n/numi-human-resting-build-20261005/resting-scene-20261005/Build/skin-source-fit-recovery-20261004/myosim-fullbody-reference.manifest.json')
def find(x):
 if isinstance(x,dict):
  if 'source_joint_map' in x:return x['source_joint_map']
  for v in x.values():
   y=find(v)
   if y:return y
 return None
report['joint_manifest_sha256']=sha(manifest);report['lower_limb_joint_coordinates']=[]
for j in find(json.loads(manifest.read_text())):
 if any(s in j['source_name'] for s in ['hip','knee','ankle','subtalar','mtp']):
  i=j['core_q_index'];report['lower_limb_joint_coordinates'].append({'name':j['source_name'],'q_index':i,'initial':state['initial_q'][i],'terminal':state['q'][i],'source_range':j['source_range'],'core_limit_status':j['core_limit_status']})
rows=list(csv.DictReader((R/'resting-coupled.csv').open()));report['bed_normal_reaction_windows']=[]
for lo,hi in [(0,10),(10,40),(140,170),(290,320)]:
 x=[float(r['normal_impulse_ns'])/.002 for r in rows if lo<=float(r['time_s'])<=hi];report['bed_normal_reaction_windows'].append({'time_window_s':[lo,hi],'sample_count':len(x),'min_n':min(x),'mean_n':sum(x)/len(x),'max_n':max(x),'reference_weight_n':72*9.81,'meaning':'Sampled final-step total normal impulse divided by timestep, not a full-horizon impulse integral or tangential force balance.'})
(O/'posture-audit.json').write_text(json.dumps(report,indent=2)+'\n')
for raw,st in opened:raw.close();st.close()
