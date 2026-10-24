"""Bounded feasibility-first fitting for a conformingly refined physical boundary."""
from pathlib import Path
import argparse,json,hashlib,time,importlib.util
import numpy as np
D=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('finite_fit014',D/'finite_witness_fit_014.py');fitmod=importlib.util.module_from_spec(spec);spec.loader.exec_module(fitmod)
E=fitmod.E
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mesh',required=True);ap.add_argument('--witnesses',required=True);ap.add_argument('--out',required=True);args=ap.parse_args()
 out=Path(args.out);out.mkdir(exist_ok=False);begun=time.monotonic();mesh=Path(args.mesh);wp=Path(args.witnesses)
 m=dict(np.load(mesh));z=dict(np.load(wp));bp=E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz';ip=bp.with_name('inventory.json')
 banks=np.load(bp);inv=json.loads(ip.read_text())['banks'];x=m['points_world_m'].astype(float);f=m['physical_outer_faces']
 A,b,w,sign=fitmod.contact_rows(x,f,z,banks,inv)
 delta,ids,info,owner=fitmod.fit(x,f,A,b,feasibility_only=True)
 print(json.dumps({'stage':'feasibility','constraints':len(b),'lp':info,'seconds':time.monotonic()-begun}),flush=True)
 if info['minimum_contact_slack_sum']>1e-6:
  report={'status':'REJECTED_AT_LINEARIZED_FEASIBILITY','lp':info,'constraints':len(b),
   'physical_admission':False,'native_admission':False,'fitted_surface_emitted':False,
   'elapsed_s':time.monotonic()-begun,'input_sha256':{str(p):sha(p) for p in (mesh,wp,bp,ip,owner,Path(fitmod.__file__),Path(__file__))}}
  (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');return
 delta,ids,info,owner=fitmod.fit(x,f,A,b,feasibility_only=False)
 new=(x+delta).astype(np.float32);actual=new.astype(float)-x
 t0=x[f];t1=new[f].astype(float)
 n0=np.cross(t0[:,1]-t0[:,0],t0[:,2]-t0[:,0]);n1=np.cross(t1[:,1]-t1[:,0],t1[:,2]-t1[:,0])
 cos=np.einsum('ij,ij->i',n0,n1)/(np.linalg.norm(n0,axis=1)*np.linalg.norm(n1,axis=1))
 proposed=out/'outer-envelope-proposal.npz';np.savez_compressed(proposed,points_world_m=new,physical_outer_faces=f,
  node_displacements_m=actual,changed_node_ids=np.flatnonzero(np.any(actual,axis=1)),active_node_ids=ids)
 report={'status':'CONSTRUCTION_REQUIRES_GEOMETRY_AND_TRANSFER_GATES','lp':info,'constraints':len(b),
  'max_displacement_mm':float(np.linalg.norm(actual,axis=1).max()*1000),'surface_normal_flips':int(sum(cos<=0)),
  'min_surface_normal_cosine':float(cos.min()),'min_emitted_constraint_slack_mm':float(np.min(A@(actual.ravel()*1000)-b)),
  'physical_admission':False,'native_admission':False,'elapsed_s':time.monotonic()-begun,
  'input_sha256':{str(p):sha(p) for p in (mesh,wp,bp,ip,owner,Path(fitmod.__file__),Path(__file__))},'output_sha256':sha(proposed)}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

