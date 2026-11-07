# Independent ray/triangle pinhole projection of the authored tetrahedron.
# Does not model occlusion, clipping, scene bindings or Metal rasterization.
import math
def sub(a,b): return tuple(x-y for x,y in zip(a,b))
def cross(a,b): return (a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0])
def dot(a,b): return sum(x*y for x,y in zip(a,b))
def hit(o,d,a,b,c):
 e1=sub(b,a);e2=sub(c,a);h=cross(d,e2);det=dot(e1,h)
 if abs(det)<1e-12: return False
 inv=1/det;s=sub(o,a);u=inv*dot(s,h)
 if u<0 or u>1:return False
 q=cross(s,e1);v=inv*dot(d,q)
 return v>=0 and u+v<=1 and inv*dot(e2,q)>0
f=4.5/math.tan(math.radians(27))
triangles=((0,2,1),(0,1,3),(1,2,3),(2,0,3))
vertices=((-0.07,-0.07,-0.07),(0.07,-0.07,-0.07),(0,0.08,-0.07),(0,0,0.09))
for z in (3.5,2.0):
 points=[(x+.08,y,z+dz) for x,y,dz in vertices]
 for env,camera in enumerate(((.078068,.000340,1.478597),(.060628,-.002845,1.471298))):
  rays=[]
  for py in range(9):
   for px in range(16):
    direction=((px+.5-8)/f,(py+.5-4.5)/f,1)
    if any(hit(camera,direction,*(points[i] for i in t)) for t in triangles): rays.append((px,py))
  print("target_z",z,"env",env,"intersecting_pixel_rays",rays)
