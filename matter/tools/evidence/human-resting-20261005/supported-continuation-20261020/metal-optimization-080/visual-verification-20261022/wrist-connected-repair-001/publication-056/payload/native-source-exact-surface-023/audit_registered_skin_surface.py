"""Audit exact emitted registered skin, tying the CSV to its launch payload."""
from pathlib import Path
import argparse,hashlib,json,sys,time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
sys.path.insert(0,str(E/'current-detailed-skin-crossing-audit-018'))
from audit_current_detailed_skin_018 import load_native_positions,ci,rb
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def validate_launch_payload(skin,launch):
 argv=launch['argv'];ix=argv.index('--skin-payload')
 if Path(argv[ix+1]).resolve()!=Path(skin).resolve():raise ValueError('CSV launch payload path differs from supplied asset')
 pins={str(Path(k).resolve()):v for k,v in launch.get('input_hashes',{}).items()}
 if pins.get(str(Path(skin).resolve()))!=sha(skin):raise ValueError('launch payload hash pin absent or mismatched')
def main():
 ap=argparse.ArgumentParser()
 for key in ('skin','csv','pose','launch','out'):ap.add_argument('--'+key,type=Path,required=True)
 args=ap.parse_args();args.out.mkdir(exist_ok=False)
 start=time.monotonic();asset=rb.decode(args.skin.read_bytes())
 launch=json.loads(args.launch.read_text());validate_launch_payload(args.skin,launch)
 pose=json.loads(args.pose.read_text())
 # Existing replay owner verifies ABI, binding count and registration identity.
 helper=rb.registered_positions_float32(asset,pose,frame='current')
 vertices=load_native_positions(args.csv,asset['nv'])
 records,_=ci.float32_surface_records(vertices,asset['inds'])
 _,result=ci._prepare_complete_self_pair_baseline(records)
 faces=asset['inds'];areas=np.linalg.norm(np.cross(vertices[faces[:,1]].astype(float)-vertices[faces[:,0]].astype(float),
    vertices[faces[:,2]].astype(float)-vertices[faces[:,0]].astype(float)),axis=1)*.5
 errors=np.linalg.norm(vertices.astype(float)-helper.astype(float),axis=1)
 pairs=[list(map(int,z)) for z in result['triangle_pairs']]
 report=dict(status='PASS_EXACT_NATIVE_SOURCE_SURFACE' if not pairs and np.all(areas>0) else 'REJECT_NATIVE_SOURCE_SURFACE',
  scope='Actual native source-emission positions only, before accepted FEM deformation. Launch may stop at a later admission guard; this is not a physical/native simulation qualification.',
  input_sha256={str(p):sha(p) for p in (args.skin,args.csv,args.pose,args.launch,Path(__file__),Path(ci.__file__),Path(rb.__file__))},
  vertex_count=asset['nv'],face_count=len(faces),exact_self_pair_count=len(pairs),exact_self_pairs=pairs,
  degenerate_faces=np.flatnonzero(areas<=0).tolist(),helper_position_error_max_m=float(errors.max()),
  helper_position_error_rms_m=float(np.sqrt(np.mean(errors**2))),aabb_candidates=result['aabb_candidate_pairs'],
  complete_surface_audit=True,elapsed_s=time.monotonic()-start,physical_admission=False,native_simulation_admission=False)
 (args.out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report),flush=True)
if __name__=='__main__':main()

