import json, pathlib, struct, collections
from fractions import Fraction as F
p=pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/mortar-union-four-patch-overlay-006/candidates.log')
def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def exact(s): return F.from_float(f32(float.fromhex(s)))
def orient(a,b,c): return (b[0]-a[0])*(c[1]-a[1])-(b[1]-a[1])*(c[0]-a[0])
count=collections.Counter(); ex=[]
for line in p.read_text().splitlines():
 if not line.startswith('UNPROVEN,'): continue
 f=line.split(','); side=f[1]; ids=tuple(map(int,f[2:4])); vals=[exact(x) for x in f[5:]]
 A=[(vals[i*3],vals[i*3+1],vals[i*3+2]) for i in range(3)]
 B=[(vals[9+i*3],vals[10+i*3],vals[11+i*3]) for i in range(3)]
 common=[]
 for i,a in enumerate(A):
  for j,b in enumerate(B):
   if a==b: common.append((i,j,a))
 if len(common)!=2:
  key=f'exact_common_{len(common)}'
 else:
  (ia,ib,x),(ja,jb,y)=common
  if x[:2]==y[:2]: key='common_edge_projected_degenerate'
  else:
   ka=next(k for k in range(3) if k not in (ia,ja)); kb=next(k for k in range(3) if k not in (ib,jb))
   sa=orient(x,y,A[ka]); sb=orient(x,y,B[kb])
   key=('exact_opposite' if sa*sb<0 else 'exact_same_or_zero')
   if key=='exact_opposite':
    ex.append((ids,side, float(sa),float(sb),str(sa),str(sb)))
 count[key]+=1
print(dict(count)); print('exact opposite examples',ex[:4])
