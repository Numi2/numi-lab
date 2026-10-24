"""Geometry-only outward envelope proposal using finite ray exits, not infinite planes."""
from pathlib import Path
import importlib.util,argparse,numpy as np,json,time,hashlib
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('fit014scalar',E/'connected-skin-envelope-fit-014/fit_connected_envelope_014.py')
base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ray_outward_exits(p,d,tri,limit):
 # Float64 construction query only. Exact315 remains the final intersection owner.
 edge1=tri[:,1]-tri[:,0];edge2=tri[:,2]-tri[:,0];h=np.cross(np.broadcast_to(d,edge2.shape),edge2)
 det=np.einsum('ij,ij->i',edge1,h);normal=np.cross(edge1,edge2)
 scale=np.linalg.norm(edge1,axis=1)*np.linalg.norm(edge2,axis=1)
 good=np.abs(det)>np.maximum(scale*1e-13,1e-300)
 inv=np.zeros(len(tri));inv[good]=1./det[good]
 s=p-tri[:,0];u=np.einsum('ij,ij->i',s,h)*inv;q=np.cross(s,edge1)
 v=q@d*inv;t=np.einsum('ij,ij->i',edge2,q)*inv
 good &= (u>=-1e-10)&(v>=-1e-10)&(u+v<=1+1e-10)&(t>=-1e-9)&(t<=limit)&((normal@d)>0)
 return np.maximum(t[good],0)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mesh',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
 out=Path(args.out);out.mkdir(exist_ok=False);begun=time.monotonic();mp=Path(args.mesh);m=dict(np.load(mp));x=m['points_world_m'].astype(float);f=m['physical_outer_faces']
 zp=E/'global-inner-p1-full-domain-clipping-009/validation-001/zero-support-physical-O-face-pairs-010.npz';z=dict(np.load(zp))
 bp=E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz';banks=np.load(bp);ip=bp.with_name('inventory.json');inv=json.loads(ip.read_text())['banks']
 # Each refined node can query only tissues implicated on an incident original crossed face.
 face_banks={}
 for bi,of in zip(z['bank_ids'],z['physical_O_face_ids']):face_banks.setdefault(int(of),set()).add(int(bi))
 node_banks={}
 for fi,old in enumerate(m['physical_O_parent_face_rows']):
  bset=face_banks.get(int(old))
  if bset:
   for node in f[fi]:node_banks.setdefault(int(node),set()).update(bset)
 normals=base.vertex_directions(x,f);cache={}
 for bi in set().union(*node_banks.values()):
  key=inv[bi]['key'];v=banks[key+'_vertices'].astype(float);bf=banks[key+'_faces'];tri=v[bf]
  # Source winding is checked through signed volume for construction; not a closed-solid certificate.
  volume=np.einsum('ij,ij->i',tri[:,0],np.cross(tri[:,1],tri[:,2])).sum()
  if volume<=0:raise ValueError('nonpositive bank winding; repair query needs explicit orientation owner')
  cache[bi]=(tri,tri.min(1),tri.max(1))
 lo=np.zeros(len(x));records=[];cap=.025
 for node,bset in sorted(node_banks.items()):
  p=x[node];d=normals[node]
  if np.linalg.norm(d)<.9:raise ValueError('no outward source vertex direction')
  endpoint=p+cap*d;lower=np.minimum(p,endpoint)-1e-10;upper=np.maximum(p,endpoint)+1e-10
  best=0.;owner=-1;hitcount=0
  for bi in sorted(bset):
   tri,blo,bhi=cache[bi];sel=np.all(bhi>=lower,axis=1)&np.all(blo<=upper,axis=1)
   hits=ray_outward_exits(p,d,tri[sel],cap);hitcount+=len(hits)
   if len(hits) and hits.max()>best:best=float(hits.max());owner=bi
  if owner>=0:lo[node]=best+.00002
  records.append((node,owner,best,hitcount))
 d,ids,opt=base.fit_connected_scalar_field(x,f,lo,rings=4,smooth=2.)
 new=(x+normals*d[:,None]).astype(np.float32);actual=new.astype(float)-x
 t0=x[f];t1=new[f].astype(float);n0=np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0]);n1=np.cross(t1[:,1]-t1[:,0],t1[:,2]-t1[:,0])
 area0=np.linalg.norm(n0,axis=1);area1=np.linalg.norm(n1,axis=1);cos=np.einsum('ij,ij->i',n0,n1)/np.maximum(area0*area1,1e-300)
 dest=out/'outer-envelope-proposal.npz';np.savez_compressed(dest,points_world_m=new,physical_outer_faces=f,changed_node_ids=np.flatnonzero(np.any(actual,axis=1)),
  node_displacements_m=actual,construction_ray_records=np.asarray(records),ray_lower_bounds_m=lo,active_node_ids=ids)
 report={'status':'OUTWARD_RAY_CONSTRUCTION_ONLY','optimizer_success':bool(opt.success),'iterations':opt.nit,
  'queried_nodes':len(records),'nodes_with_outward_source_exit':int(sum(lo>0)),'active_nodes':len(ids),'moved_nodes':int(sum(np.any(actual,axis=1))),
  'maximum_displacement_mm':float(np.linalg.norm(actual,axis=1).max()*1000),'surface_normal_flips':int(sum(cos<=0)),
  'min_normal_cosine':float(cos.min()),'area_ratio_quantiles':np.quantile(area1/area0,[0,.01,.5,.99,1]).tolist(),
  'ray_limit_mm':cap*1000,'physical_admission':False,'native_admission':False,'needs_exact_geometry_gate':True,
  'input_sha256':{str(p):sha(p) for p in (mp,zp,bp,ip,Path(base.__file__),Path(__file__))},'output_sha256':sha(dest),
  'elapsed_s':time.monotonic()-begun}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

