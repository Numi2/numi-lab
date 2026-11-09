from pathlib import Path
import ctypes,itertools,json
import numpy as np
lib=ctypes.CDLL('/tmp/crow-box-distance.dylib')
ptr=ctypes.POINTER(ctypes.c_double);lib.crow_distance.argtypes=[ptr,ptr];lib.crow_distance.restype=ctypes.c_double
rng=np.random.default_rng(82736)
def rot(q):
 x,y,z,w=q
 return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],[2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],[2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])
def fixture():
 q=rng.normal(size=4);q/=np.linalg.norm(q)
 return np.r_[rng.uniform(-2,2,3),q,rng.uniform(.1,1,3)]
def reference(a,b):
 # Independent box-constrained convex least-squares reference: enumerate all
 # fixed lower/free/fixed upper coordinate sets and solve each free subspace.
 M=np.c_[rot(a[3:7]),-rot(b[3:7])];d=a[:3]-b[:3];h=np.r_[a[7:],b[7:]]
 best=np.inf
 for pattern in itertools.product((-1,0,1),repeat=6):
  p=np.array(pattern);fixed=p!=0;free=~fixed
  x=p*h
  target=-(d+M[:,fixed]@x[fixed])
  if np.any(free):
   v=np.linalg.lstsq(M[:,free],target,rcond=None)[0]
   if np.any(np.abs(v)>h[free]+1e-9):continue
   x[free]=v
  best=min(best,np.linalg.norm(d+M@x))
 return best
max_error=0
for i in range(80):
 a=np.ascontiguousarray(fixture(),dtype=np.float64);b=np.ascontiguousarray(fixture(),dtype=np.float64)
 actual=lib.crow_distance(a.ctypes.data_as(ptr),b.ctypes.data_as(ptr));expected=reference(a,b)
 err=abs(actual-expected);max_error=max(max_error,err)
 if err>2e-8:raise AssertionError((i,actual,expected,a,b))
report={'random_seed':82736,'fixtures':80,'independent_reference':'exhaustive bound/free active-set convex least-squares','max_absolute_error':max_error}
Path('artifacts/crow-clearance-observer-20261109/reference-results.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
