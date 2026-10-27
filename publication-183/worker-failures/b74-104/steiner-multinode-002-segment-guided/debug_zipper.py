import json,numpy as np
from pathlib import Path
R=Path('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b74-104/steiner-multinode-002-segment-guided')
g=json.load(open(R/'segment-graph-report.json'));Z=np.load('/Users/numi/Work/numi-human-connected-surface-069/D/publication-182-stage-001/combined-653/candidate.npz',allow_pickle=False)
W=json.load(open('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b74-104/baseline-witnesses.json'))
b=json.load(open('/Users/numi/Work/numi-human-connected-surface-069/D/surface-clearance-546/component-sat-selector-638/per-bank-workers-20261027/b74-104/steiner-multinode-001/boundary-feasibility.json'))
F=Z['source_O_faces'];old=list(map(int,W['skin_child_faces']));loop=[int(x['node']) for x in b['fixed_boundary_vertices']]
nb={}
for fr in old:
 for n,bc in zip(F[fr],Z['refinement_face_parent_barycentric'][fr]):nb[int(n)]=np.asarray(bc)
ob=np.asarray([nb[n] for n in loop]); iball=np.asarray(g['intersection_points_barycentric'],float);edges=g['segments_point_indices']
ad=[[] for _ in iball]
for a,c in edges:ad[int(a)].append(int(c));ad[int(c)].append(int(a))
cy=[0];prev=-1;cur=0
while True:
 nx=[x for x in ad[cur] if x!=prev][0]
 if nx==cy[0]:break
 cy.append(nx);prev,cur=cur,nx
ip=iball[cy];ox=ob[:,1:];ix=ip[:,1:]
def area(p):return .5*sum(p[i,0]*p[(i+1)%len(p),1]-p[(i+1)%len(p),0]*p[i,1] for i in range(len(p)))
if area(ox)<0:ox=ox[::-1]
if area(ix)<0:ix=ix[::-1]
def centroid(p):
 c=[]; 
 for i in range(len(p)):c.append(p[i,0]*p[(i+1)%len(p),1]-p[(i+1)%len(p),0]*p[i,1])
 a=sum(c)*.5;v=np.zeros(2)
 for i in range(len(p)):v+=(p[i]+p[(i+1)%len(p)])*c[i]
 return v/(6*a)
cen=centroid(ix)
def order(p):
 ang=np.mod(np.arctan2(p[:,1]-cen[1],p[:,0]-cen[0]),2*np.pi);return list(np.argsort(ang)),ang
oo,oa=order(ox);ii,ia=order(ix);oi=len(oo)-1;ij=len(ii)-1;ao=ai=0;ann=[]
while ao<len(oo) or ai<len(ii):
 to=oa[oo[ao]] if ao<len(oo) else np.inf;ti=ia[ii[ai]] if ai<len(ii) else np.inf
 if to<ti:
  j=oo[ao];ann.append([('o',oo[oi]),('o',j),('i',ii[ij])]);oi=j;ao+=1
 else:
  j=ii[ai];ann.append([('o',oo[oi]),('i',j),('i',ii[ij])]);ij=j;ai+=1
def xy(t):return ox[t[1]] if t[0]=='o' else ix[t[1]]
ar=[]
for tri in ann:
 p=np.array([xy(t) for t in tri]);ar.append(area(p))
print('outer area',area(ox),'inner area',area(ix),'center',cen,'annum',len(ann),'ann sum signed',sum(ar),'abs',sum(abs(x) for x in ar),'minmax',min(ar),max(ar))
print('outer xy',ox.tolist());print('inner first',ix[:8].tolist());print('angles outer',np.round([oa[x] for x in oo],4).tolist());print('angles inner',np.round([ia[x] for x in ii],4).tolist())
print('first ann',ann[:5])

