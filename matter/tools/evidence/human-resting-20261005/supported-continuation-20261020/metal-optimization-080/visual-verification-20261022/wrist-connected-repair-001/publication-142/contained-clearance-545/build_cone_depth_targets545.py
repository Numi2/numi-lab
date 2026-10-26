"""Deterministic cone directions through the existing exact first-union-exit builder.
Only construction target directions change; source normals/identities and exact
185-bank event/Float32 endpoint checks remain in the pinned builder.
"""
from pathlib import Path
import sys,importlib.util,hashlib,json,numpy as np,math
W=Path(__file__).parent
def main():
 def take(k):
  i=sys.argv.index(k);v=float(sys.argv[i+1]);del sys.argv[i:i+2];return v
 angle=take('--cone-angle-545');az=take('--cone-azimuth-545')
 if not(0<=angle<90 and np.isfinite(az)):raise ValueError('invalid cone')
 p=W/'build_direct_depth_targets545.py';sp=importlib.util.spec_from_file_location('cone_builder545',p);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m);old=m.direct_depth_unit_direction
 def direction(w,n,**kw):
  n,diag=old(w,n);axis=np.eye(3)[np.argmin(np.abs(n))];u=np.cross(n,axis);u/=np.linalg.norm(u);v=np.cross(n,u);a=math.radians(angle);b=math.radians(az);d=math.cos(a)*n+math.sin(a)*(math.cos(b)*u+math.sin(b)*v);d/=np.linalg.norm(d)
  return d,dict(method='deterministic local depth cone search',cone_angle_deg=angle,cone_azimuth_deg=az,selected_depth_cosine=float(d@n),selected_source_hemisphere_cosine=float(d@w),source_ray_hemisphere_required=False,source_normal_retained_for_provenance=True,direction_world_unit=d.tolist())
 m.direct_depth_unit_direction=direction;out=Path(sys.argv[sys.argv.index('--output-dir')+1]);m.main();r=out/'nearest-union-exits.json';j=json.loads(r.read_text());j['cone_direction545']=dict(angle_deg=angle,azimuth_deg=az,wrapper_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),builder_source_sha256=hashlib.sha256(p.read_bytes()).hexdigest());r.write_text(json.dumps(j,indent=2)+'\n')
if __name__=='__main__':main()
