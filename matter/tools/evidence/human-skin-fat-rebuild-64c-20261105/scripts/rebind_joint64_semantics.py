"""Retain valid registrations; re-evaluate only changed incident semantic witnesses using owner 217."""
from pathlib import Path
import csv,json,importlib.util,hashlib,time,sys
import numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');C=W/'joint-boundary-rebuild-64c/compiler'
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
S=H/'current-native-1024-root/current-217-1036'
start=time.time()
def loadmod(p,name):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
with (C/'registration217/candidate-node-map.csv').open() as f:r=csv.DictReader(f);fields=r.fieldnames;rows=list(r)
n=int(max(int(x['joint_node_id']) for x in rows))
regrows=[None]*n
for x in rows:regrows[int(x['joint_node_id'])-1]=x
don=np.array([int(x['geometric_donor_body']) for x in regrows]);region=np.array([int(x['anatomical_region_id']) for x in regrows])
t=np.loadtxt(C/'handoff938/skin-fat-combined.ele',skiprows=1,dtype=np.int64)[:,1:5]-1
labs=np.loadtxt(C/'p1-export/child.labels-850.csv',skiprows=1,delimiter=',',dtype=np.int64)[:,1:4]
valid=np.zeros(n,dtype=bool)
for k in range(4):
 v=t[:,k];ok=(don[v]==labs[:,1])&(region[v]==labs[:,2]);valid[v[ok]]=True
bad=np.flatnonzero(~valid);print('SEMANTIC_WITNESSES',len(bad),flush=True)
m=loadmod(S/'register_current_combined_volume_nodes_217_1036.py','current217')
capture=H.parent/'native-physical-parent-mass-330/native-source-identity-control-003'
parent,owner=m.read_parent_current(capture/'compiled-parent-cell-mass.csv',S);lookup=m.make_lookup(parent,owner)
incident={int(i):set() for i in bad};badmask=~valid
for k in range(4):
 ii=np.flatnonzero(badmask[t[:,k]])
 for c in ii:incident[int(t[c,k])].add(tuple(map(int,labs[c])))
updates={};witnesses=[]
for j in bad:
 x=regrows[j];point=np.array([float(x[f'current_{a}_m']) for a in 'xyz']);role=x['role']
 allowed=sorted(k for k in incident[int(j)] if role=='interior' or k[0]==(16 if role=='D' else 15))
 candidates=[m.nearest_for_tuple(point,k,lookup,owner,.02,1e-10) for k in allowed if k in lookup]
 candidates=[v for v in candidates if v['distance']<=.02]
 if not candidates:raise ValueError(('no bounded parent',int(j),role,allowed))
 best=min(candidates,key=lambda v:(v['distance'],v['tuple'],v['tet']))
 y=dict(x);y['parent_tet_row']=str(best['tet'])
 for k in range(4):y[f'parent_bary{k}']=format(float(best['beta'][k]),'.17g')
 for k,a in enumerate('xyz'):y[f'parent_current_residual_{a}_m']=format(float(best['residual'][k]),'.17g')
 y['parent_registration_distance_m']=format(best['distance'],'.17g')
 y['geometric_donor_body']=str(best['tuple'][1]);y['anatomical_region_id']=str(best['tuple'][2]);updates[int(j)]={k:y[k] for k in ['parent_tet_row',*[f'parent_bary{k}' for k in range(4)],*[f'parent_current_residual_{a}_m' for a in 'xyz'],'parent_registration_distance_m','geometric_donor_body','anatomical_region_id']}
 witnesses.append({'node':int(j),'role':role,'old_parent':int(x['parent_tet_row']),'new_parent':best['tet'],'new_tuple':best['tuple'],'distance_m':best['distance']})
out=C/'registration217/candidate-node-map-semantic.csv'
with out.open('w') as f:
 w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
 for x in rows:
  j=int(x['joint_node_id'])-1
  if j in updates:x.update(updates[j])
  w.writerow(x)
report={'status':'PASS_BOUNDED_CURRENT_INCIDENT_PARENT_REBIND_REQUIRES_433_VALIDATION','original_map_sha256':sha(C/'registration217/candidate-node-map.csv'),'output_sha256':sha(out),'owner_path':str(S/'register_current_combined_volume_nodes_217_1036.py'),'owner_sha256':sha(S/'register_current_combined_volume_nodes_217_1036.py'),'changed_nodes':len(bad),'witnesses':witnesses,'wall_s':time.time()-start,'physical_admission':False,'native_steps':0}
(C/'registration217/semantic-rebinding.json').write_text(json.dumps(report,indent=2)+'\n');print('PASS',len(bad),time.time()-start,flush=True)
