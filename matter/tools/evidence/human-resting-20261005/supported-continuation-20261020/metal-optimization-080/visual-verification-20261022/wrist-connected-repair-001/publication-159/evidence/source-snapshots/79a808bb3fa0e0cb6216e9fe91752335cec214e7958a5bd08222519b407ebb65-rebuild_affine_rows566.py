"""Regenerate existing singleton affine ancestry from exact pinned prolongation.
No coordinates, connectivity, material coordinates, anchors or physics are changed.
"""
from pathlib import Path
import argparse,hashlib,importlib.util,json,sys
import numpy as np
N=Path(__file__).resolve().parent.parent
OWNER=N/'connected-reference-degeneracy-repair-557/frozen-v4/diagnose_reference_degeneracy_557.py'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(candidate,out):
 if out.exists():raise FileExistsError(out)
 spec=importlib.util.spec_from_file_location('lineage566owner',OWNER)
 m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
 for p,h in [(m.OWNER402,m.EXPECTED_402),(m.OWNER405,m.EXPECTED_405),
             (m.PARENT_CANDIDATE,m.PARENT_SHA),(m.CANONICAL_037,m.CANONICAL_037_SHA)]:
  if sha(p)!=h:raise ValueError('owner or ancestral source changed: '+str(p))
 c=m.load_candidate(candidate);original={k:np.array(v,copy=True)for k,v in c.items()}
 parent=m.load_candidate(m.PARENT_CANDIDATE);canonical=m.load_candidate(m.CANONICAL_037)
 owner405=m.load_module('lineage566405',m.OWNER405)
 expected,audit=m.derive_affine_face_barycentrics(c,parent,canonical,owner405)
 repair=m.repair_singleton_face_barycentrics(c,expected,audit)
 changed=m.changed_array_fields(original,c)
 if changed not in ([],['refinement_face_parent_barycentric']):raise ValueError('unrelated payload changed')
 owner402=m.load_module('lineage566402',m.OWNER402);owner402.validate_candidate(c)
 out.mkdir(parents=True);target=out/'candidate.npz'
 np.savez_compressed(target,**c)
 report={'status':'AFFINE_ANCESTRY_REGENERATED_GEOMETRY_IDENTICAL','input_candidate':str(candidate),'input_sha256':sha(candidate),
 'candidate':str(target),'candidate_sha256':sha(target),'changed_fields':changed,'audit':audit,'repair':repair,
 'coordinates_and_connectivity_bitwise_unchanged':all(m.arrays_bitwise_equal(original[k],c[k])for k in ['points_world_m','source_O_faces','area_reference_037_source_O_faces']),
 'source_pins':{str(p):sha(p)for p in [Path(__file__),OWNER,m.OWNER402,m.OWNER405,m.PARENT_CANDIDATE,m.CANONICAL_037]},
 'native_admission':False,'accepted_physical_steps':0,'requires_full_geometry_recheck_for_emitted_candidate':True}
 (out/'report.json').write_text(json.dumps(report,indent=2,sort_keys=True,allow_nan=False)+'\n')
 return report
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--candidate',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 r=run(a.candidate,a.out)
 print(json.dumps({k:r[k]for k in ['status','candidate','candidate_sha256','changed_fields','repair']}),flush=True)
if __name__=='__main__':main()

