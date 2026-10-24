"""Reclip original implicated bank facets onto refined physical O facets with owner315."""
from pathlib import Path
import numpy as np,json,hashlib,importlib.util,sys,argparse,time
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001");D=Path(__file__).resolve().parent
sys.path.insert(0,str(E/'global-inner-p1-full-domain-clipping-009/exact-O-pair-011'))
from audit_exact_pairs_011 import ci
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--mesh',required=True);ap.add_argument('--out',required=True);a=ap.parse_args()
 out=Path(a.out);out.mkdir(exist_ok=False);begun=time.monotonic()
 mesh=Path(a.mesh);m=dict(np.load(mesh));x=m['points_world_m'];f=m['physical_outer_faces'];parent=m['physical_O_parent_face_rows']
 zp=E/'global-inner-p1-full-domain-clipping-009/validation-001/zero-support-physical-O-face-pairs-010.npz';z=dict(np.load(zp))
 bp=E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz';banks=np.load(bp);ip=bp.with_name('inventory.json');inv=json.loads(ip.read_text())['banks']
 children={}
 for i,p in enumerate(parent):children.setdefault(int(p),[]).append(i)
 ofids=np.unique(np.concatenate([np.asarray(children[int(p)]) for p in np.unique(z['physical_O_face_ids'])]))
 orecs,_=ci.float32_surface_records(x,f[ofids]);orecs=dict(zip(map(int,ofids),orecs));otri=x[f].astype(float)
 rows=[];hits=[];offsets=[0];pointgroups=[];tested=0;bankcache={}
 for bi in np.unique(z['bank_ids']):
  sel=np.flatnonzero(z['bank_ids']==bi);key=inv[int(bi)]['key'];v=banks[key+'_vertices'];bf=banks[key+'_faces']
  b_ids=np.unique(z['bank_local_face_ids'][sel]);br,_=ci.float32_surface_records(v,bf[b_ids]);br=dict(zip(map(int,b_ids),br));btri=v[bf].astype(float)
  for oldrow in sel:
   fi=int(z['bank_local_face_ids'][oldrow]);oldof=int(z['physical_O_face_ids'][oldrow]);bt=btri[fi];blo=bt.min(0);bhi=bt.max(0)
   for child in children[oldof]:
    st=otri[child]
    if np.any(st.max(0)<blo) or np.any(bhi<st.min(0)):continue
    tested+=1;points=tuple(sorted(set(ci.triangle_intersection_points(br[fi][0],orecs[child][0]))))
    if not points:continue
    rows.append((int(bi),fi,int(child),int(oldrow)));pointgroups.extend(tuple(float(t/(1<<149)) for t in p) for p in points);offsets.append(len(pointgroups))
  print(json.dumps({'bank':int(bi),'tested':tested,'hit_pairs':len(rows),'seconds':time.monotonic()-begun}),flush=True)
 rows=np.asarray(rows,np.int64).reshape(-1,4)
 dest=out/'refined-exact-witnesses.npz'
 np.savez_compressed(dest,bank_ids=rows[:,0],bank_local_face_ids=rows[:,1],physical_O_face_ids=rows[:,2],
  baseline_010_pair_rows=rows[:,3],exact_hit_offsets=np.asarray(offsets,np.int64),exact_hit_points_m=np.asarray(pointgroups).reshape(-1,3))
 report={'status':'EXACT_RECLIP_OF_ORIGINAL_WITNESS_NEIGHBORHOOD','baseline_pairs':len(z['bank_ids']),
  'tested_child_pairs':tested,'child_hit_pairs':len(rows),'uncovered_original_pairs':len(set(range(len(z['bank_ids'])))-set(rows[:,3])),
  'new_domain_complete':False,'physical_admission':False,'native_admission':False,'elapsed_s':time.monotonic()-begun,
  'input_sha256':{str(p):sha(p) for p in (mesh,zp,bp,ip,Path(__file__),Path(ci.__file__))},'output_sha256':sha(dest)}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

