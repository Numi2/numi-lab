from pathlib import Path
import json,numpy as np
from collections import defaultdict
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001');ns={};exec((E/'audit_bilateral_skin_reweighted_candidate_016.py').read_text().split('def main():')[0],ns)
D=ns['decode_payload']((E/'candidate-bilateral-arm-hand-weights-016.nhskin').read_bytes());pose=json.loads(next((E/'native-pose-checks/repair-005/neutral').glob('*.skin-poses.json')).read_text());P=ns['posed'](D,pose).astype(float);F=np.asarray(D['indices'],int).reshape(-1,3);axes=json.loads((E/'wrist-distal-radius-contour-fit-003.json').read_text())['sides'];B=D['bindings_u'][:,0];W=D['full_weights'];names={int(x['core_body_index']):x['myosim_body'] for x in json.loads(ns['MANIFEST'].read_text())['source']['anchors']};owners=B[np.argmax(W,axis=1)]
def sections(P,F,c,a,stations):
 e1=np.cross(a,[0,1,0]);e1/=np.linalg.norm(e1);e2=np.cross(a,e1);out=[]
 for st in stations:
  t=(P-c)@a-st;sel=np.flatnonzero((t[F].min(1)<=0)&(t[F].max(1)>0));pts=[];adj=defaultdict(set);positions={}
  for fi in sel:
   f=F[fi];p=[]
   for k in range(3):
    u,v=f[k],f[(k+1)%3]
    if t[u]*t[v]<0:
     q=P[u]+t[u]/(t[u]-t[v])*(P[v]-P[u]);p.append(q)
   if len(p)!=2:continue
   keys=[tuple(np.rint(q*1e7).astype(int)) for q in p]
   for k,q in zip(keys,p):positions[k]=q
   adj[keys[0]].add(keys[1]);adj[keys[1]].add(keys[0])
  un=set(adj);comps=[]
  while un:
   v=un.pop();comp={v};stack=[v]
   while stack:
    for n in adj[stack.pop()]:
     if n in un:un.remove(n);comp.add(n);stack.append(n)
   if len(comp)>8:
    q=np.array([positions[k] for k in comp]);dist=np.linalg.norm(q.mean(0)-(c+st*a));comps.append((dist,q,all(len(adj[k])==2 for k in comp)))
  if not comps:out.append({'s':st});continue
  _,q,closed=min(comps,key=lambda z:z[0]);uv=np.column_stack(((q-c)@e1,(q-c)@e2));lo=uv.min(0);hi=uv.max(0);center=(lo+hi)/2;span=hi-lo
  out.append({'s':st,'closed':closed,'n':len(q),'center':center.tolist(),'span':span.tolist(),'perimeter':sum(np.linalg.norm(positions[u]-positions[v]) for u in adj for v in adj[u] if u<v and u in set(map(lambda q:tuple(np.rint(q*1e7).astype(int)),q)))})
 return out,e1,e2
if __name__=='__main__':
 for side in ['left','right']:
  d=axes[side];c=np.array(d['distal_radius_center_world_m']);a=np.array(d['radius_axis_toward_capitate_world']);m=np.array([names.get(int(b),'').endswith('_l' if side=='left' else '_r') and ns['region'](names.get(int(b),''))!='other' for b in owners]);f=F[np.all(m[F],axis=1)];out,_,_=sections(P,f,c,a,np.arange(-.06,.061,.01));print(side,json.dumps(out))
