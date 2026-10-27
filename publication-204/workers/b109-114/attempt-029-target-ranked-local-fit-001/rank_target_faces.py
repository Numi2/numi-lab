#!/usr/bin/env python3
from pathlib import Path
import json,hashlib,importlib.util,sys
from collections import Counter,defaultdict,deque
import numpy as np
S=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638')
W=S/'per-bank-workers-20261027';D=W/'b109-114/attempt-029-target-ranked-local-fit-001'
C=W/'combined-d071-b109-b61-001/candidate.npz';R159=W/'combined-d071-b109-b61-001/audit-159/report.json';R130=W/'combined-d071-b109-b61-001/audit-130/summary.json'
EXPECTED='806abdf650524b249951c81a95cd467e1a3864d24c2cb185b5a8c3297cf1c756'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
if sha(C)!=EXPECTED:raise RuntimeError('candidate pin mismatch')
doc=json.loads(R159.read_text());bank=next(x for x in doc['banks'] if (int(x['body']),int(x['stable_bone_id']))==(109,114));pairs=np.asarray(bank['result']['triangle_pairs'],np.int64)
with np.load(C,allow_pickle=False) as z:
 P=np.asarray(z['points_world_m'],np.float32);F=np.asarray(z['source_O_faces'],np.int64);parents=np.asarray(z['refinement_face_parent_face_row'],np.int64);keys=set(z.files)
counts=Counter(map(int,pairs[:,0]));top=counts.most_common(24)
hp=Path('/Users/numi/Work/generic-static-query-042-freeze/queryrow-tet-edge-043/matter/tools/human_skin_fat_bone_audit.py')
sp=importlib.util.spec_from_file_location('bankhelper',hp);h=importlib.util.module_from_spec(sp);sys.modules[sp.name]=h;sp.loader.exec_module(h)
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-2022/wrist-connected-repair-001')
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
BANK=E/'bone-clearance-fit-038-exact-bank-export-039/run-002/source-preflight'
tri,ids,gids,bodies,M,off,pos,bkeys,pins,exact=h.read_bank_inputs(BANK,E)
mask=(ids[:,0]==109)&(ids[:,1]==114);BT=np.asarray(tri[mask],np.float64)*1000.;BID=np.asarray(ids[mask,3],np.int64)
loc=((P.astype(np.float64)-pos[109])@M[109].T-off[109])*1000.
def token(v):return tuple(np.asarray(v,np.float32).view(np.uint32).tolist())
vmap={};ti=np.empty((len(BT),3),np.int64)
for i,T in enumerate(BT):
 for j,v in enumerate(T):ti[i,j]=vmap.setdefault(token(v),len(vmap))
uses=defaultdict(list)
for i,t in enumerate(ti):
 for j in range(3):
  a,b=int(t[j]),int(t[(j+1)%3]);k=tuple(sorted((a,b)));sgn=1 if (a,b)==k else -1;uses[k].append((i,sgn))
ec=Counter(map(len,uses.values()));adj=[[] for _ in BT]
for uv in uses.values():
 if len(uv)==2:
  (i,si),(j,sj)=uv;rel=-si*sj;adj[i].append((j,rel));adj[j].append((i,rel))
orient=np.zeros(len(BT),np.int8);components=[];conflicts=0
for root in range(len(BT)):
 if orient[root]:continue
 orient[root]=1;q=deque([root]);comp=[]
 while q:
  i=q.popleft();comp.append(i)
  for j,rel in adj[i]:
   want=orient[i]*rel
   if orient[j]==0:orient[j]=want;q.append(j)
   elif orient[j]!=want:conflicts+=1
 components.append(comp)
vols=[]
for comp in components:
 vol=0.
 for i in comp:
  t=BT[i]*int(orient[i]);vol+=float(np.dot(t[0],np.cross(t[1],t[2]))/6.)
 if vol<0:orient[np.asarray(comp,np.int64)]*=-1;vol=-vol
 vols.append(vol)
if dict(ec)!={2:len(uses)} or conflicts or len(components)!=1:raise RuntimeError(f'not one closed orientable shell: {dict(ec)} {conflicts} {len(components)}')
ranked=[]
for fi,n in top:
 bone_rows=[int(x) for x in pairs[pairs[:,0]==fi,1]];normals=[];dists=[]
 for bone_row in bone_rows:
  k=bone_row;tb=BT[k];nn=np.cross(tb[1]-tb[0],tb[2]-tb[0])*int(orient[k]);nn/=np.linalg.norm(nn);normals.append(nn);dists.extend(((loc[F[fi]]-tb[0])@nn).tolist())
 mean=np.mean(normals,axis=0);mean/=np.linalg.norm(mean);align=[float(np.dot(mean,nm)) for nm in normals]
 ranked.append({'skin_face':int(fi),'parent_face':int(parents[fi]),'pair_count':int(n),'nodes':F[fi].astype(int).tolist(),'bone_triangle_rows':bone_rows,'bone_stable_ids':[int(BID[k]) for k in bone_rows],'normal_min_alignment':min(align),'mean_outward_normal':mean.tolist(),'most_inward_plane_distance_mm':float(min(dists))})
edgefaces=defaultdict(list)
for fi,face in enumerate(F):
 for a,b in ((face[0],face[1]),(face[1],face[2]),(face[2],face[0])):edgefaces[tuple(sorted((int(a),int(b))))].append(fi)
fadj=[set() for _ in F]
for fs in edgefaces.values():
 if len(fs)==2:a,b=fs;fadj[a].add(b);fadj[b].add(a)
seed=ranked[0]['skin_face'];sfaces={seed};front={seed}
for _ in range(4):
 nxt=set().union(*(fadj[i] for i in front))-sfaces if front else set();sfaces.update(nxt);front=nxt
snodes=sorted(set(map(int,F[sorted(sfaces)].ravel())))
insdoc=json.loads(R130.read_text());inside=[r for r in insdoc.get('inside_witnesses',[]) if (int(r['bank_body']),int(r['bank_stable_id']))==(109,114)]
result={'schema':'numi.human.b109-114.target-ranking.v1','candidate_sha256':sha(C),'audit159_sha256':sha(R159),'audit130_sha256':sha(R130),'target_bank':[109,114],'raw_pairs':int(len(pairs)),'unique_skin_faces':len(counts),'unique_bone_triangle_rows':len(set(map(int,pairs[:,1]))),'top_skin_faces':ranked,'bone_mesh_topology':{'vertices':len(vmap),'edges':len(uses),'faces':len(BT),'edge_use_counts':dict(ec),'components':len(components),'orientation_conflicts':conflicts,'positive_volume_mm3':vols},'selected_seed_face':int(seed),'selected_4ring_faces':len(sfaces),'selected_nodes':len(snodes),'target_inside_identities':len(inside),'operation':'ranked local outward-normal patch trial with fixed outer ring; exact Float32 crossing, shape, layer checks.'}
out=D/'target-ranking-corrected.json';out.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n')
print(json.dumps({'status':'RANKED','pairs':len(pairs),'unique_skin_faces':len(counts),'top_faces':[{'face':r['skin_face'],'parent':r['parent_face'],'pairs':r['pair_count'],'bone_ids':r['bone_triangle_rows'],'normal_min_alignment':r['normal_min_alignment'],'inward_mm':r['most_inward_plane_distance_mm']} for r in ranked[:12]],'selected_seed':seed,'selected_4ring_faces':len(sfaces),'selected_nodes':len(snodes),'inside_identities':len(inside),'bone_edge_use_counts':dict(ec),'report':str(out)},indent=2))
