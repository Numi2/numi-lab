"""Construct connected face-patch proposals around exact crossing components."""
from collections import defaultdict,Counter
import numpy as np
def component_patch_centers(faces,pairs,max_centers=2000,rings=2,max_faces=512):
 f=np.asarray(faces)
 if f.ndim!=2 or f.shape[1]!=3 or f.dtype.kind not in"iu":raise ValueError("integer triangle domain required")
 hits=Counter(int(row)for ps in pairs.values()for row,_ in ps)
 if any(row<0 or row>=len(f)for row in hits):raise ValueError("crossing face outside domain")
 em=defaultdict(list)
 for i,t in enumerate(f):
  for a,b in((t[0],t[1]),(t[1],t[2]),(t[2],t[0])):em[tuple(sorted((int(a),int(b))))].append(i)
 adj=[set()for _ in f]
 for incident in em.values():
  for i in incident:adj[i].update(j for j in incident if j!=i)
 unseen=set(hits);components=[]
 while unseen:
  seed=min(unseen);unseen.remove(seed);group={seed};stack=[seed]
  while stack:
   i=stack.pop();take=adj[i]&unseen;unseen-=take;group|=take;stack.extend(sorted(take,reverse=True))
  components.append(group)
 candidates={}
 for group in components:
  patch=set(group)
  for radius in range(rings+1):
   if 2<=len(patch)<=max_faces:
    key=tuple(sorted(patch));candidates.setdefault(key,{"crossed_component":sorted(group),"ring":radius})
   patch|={j for i in tuple(patch)for j in adj[i]}
 ordered=sorted(candidates,key=lambda rr:(-sum(hits.get(i,0)for i in rr),len(rr),rr))
 return ordered[:max_centers],candidates

