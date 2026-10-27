"""Angular projection of a direction onto two unit-vector cosine caps."""
import math,numpy as np
def nearest_joint_direction597(source,current,reference,minimum_cosine=.05):
 s,c,v=(np.asarray(a,dtype=float)for a in(source,current,reference));floor=float(minimum_cosine)
 if not 0<floor<1 or any(a.shape!=(3,)or not np.isfinite(a).all()or abs(float(np.linalg.norm(a))-1)>2e-10 for a in(s,c,v)):raise ValueError("finite unit directions and valid cosine floor required")
 bound=floor+128*np.finfo(float).eps
 candidates=[]
 def add(d,case):
  d=np.asarray(d,dtype=float);length=float(np.linalg.norm(d))
  if length<=0 or not np.isfinite(length):return
  d=d/length
  if min(float(s@d),float(c@d))>=floor:candidates.append((float(v@d),tuple(d),case))
 add(v,"unconstrained")
 for n in(s,c):
  tangent=v-float(v@n)*n;t=float(np.linalg.norm(tangent))
  if t<=1e-14:
   axis=np.eye(3)[int(np.argmin(np.abs(n)))];tangent=np.cross(n,axis);t=float(np.linalg.norm(tangent))
   signs=(1.,-1.)
  else:signs=(1.,)
  for sign in signs:add(bound*n+sign*math.sqrt(1-bound*bound)*tangent/t,"one_cap")
 a=float(s@c);den=1+a
 if den>0:
  center=bound*(s+c)/den;h2=1-float(center@center);axis=np.cross(s,c);an=float(np.linalg.norm(axis))
  if h2>=-1e-13 and an>1e-14:
   h=math.sqrt(max(0.,h2))
   add(center+h*axis/an,"two_caps");add(center-h*axis/an,"two_caps")
 if not candidates:raise ValueError("joint source/current cosine caps infeasible or numerically unresolved")
 candidates.sort(key=lambda q:(-q[0],q[1],q[2]));cosine,coords,case=candidates[0];d=np.asarray(coords)
 return d,{"joint_source_current_feasible":True,"method":"maximum reference-direction cosine over two unit-sphere caps; active-set enumeration, not minimum ray-exit distance","reference_direction_world_unit":v.tolist(),"selected_reference_cosine":cosine,"selected_source_hemisphere_cosine":float(s@d),"selected_depth_cosine":float(c@d),"minimum_source_outward_cosine":floor,"minimum_depth_cosine":floor,"active_case":case,"direction_world_unit":d.tolist()}

