import pathlib,struct,math,json
from fractions import Fraction as F
src=pathlib.Path('/Users/home/cardboard-implementation-20261014/analysis/mortar-union-four-patch-overlay-006/candidates.log')
def f32(x): return struct.unpack('<f',struct.pack('<f',x))[0]
def exact32(s): return F.from_float(f32(float.fromhex(s)))
def cross(a,b): return a[0]*b[1]-a[1]*b[0]
def sub(a,b): return (a[0]-b[0],a[1]-b[1])
def orient(a,b,c): return cross(sub(b,a),sub(c,a))
def intersection(a,b):
 o=orient(b[0],b[1],b[2])
 if o==0: return None
 sign=1. if o>0 else -1.
 poly=list(a)
 for i in range(3):
  e0,e1=b[i],b[(i+1)%3]; edge=sub(e1,e0)
  def side(p): return sign*cross(edge,sub(p,e0))
  nxt=[]
  for j,p in enumerate(poly):
   q=poly[(j+1)%len(poly)]; sp,sq=side(p),side(q)
   pin,qin=sp>=0.,sq>=0.
   if pin:nxt.append(p)
   if pin != qin:
    den=sp-sq
    if den==0:return math.inf
    t=sp/den
    nxt.append((p[0]+t*(q[0]-p[0]),p[1]+t*(q[1]-p[1])))
  poly=nxt
 if len(poly)<3:return 0.
 twice=math.fsum(cross(poly[i],poly[(i+1)%len(poly)]) for i in range(len(poly)))
 return abs(twice)*.5
rows=[]
for line in src.read_text().splitlines():
 if not line.startswith('UNPROVEN,'):continue
 f=line.split(',');vals=[exact32(x) for x in f[5:]]
 a=[(float(vals[i*3]),float(vals[i*3+1])) for i in range(3)]
 b=[(float(vals[9+i*3]),float(vals[10+i*3])) for i in range(3)]
 d=intersection(a,b)
 rows.append(d)
th=f32(1e-10)
finite=[x for x in rows if math.isfinite(x)]
print(json.dumps({'candidate_records':len(rows),'finite':len(finite),'nonfinite':len(rows)-len(finite),'positive':sum(x>0 for x in finite),'above_fp32_threshold':sum(x>th for x in finite),'max_double_area':max(finite),'max_positive':max([x for x in finite if x>0],default=0),'threshold_float32':th,'distinct_positive_count':len(set(x for x in finite if x>0))},indent=2))
