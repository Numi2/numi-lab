"""Existing owner315 exact screen of zero-support O/bank candidate pairs."""
from pathlib import Path
from collections import Counter
import importlib.util,sys,json,hashlib,time
import numpy as np
E=Path("/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001")
D=E/'global-inner-p1-full-domain-clipping-009'
HERE=Path(__file__).resolve().parent
EXACT=E/'collision-free-seam-records-owner-315/src/numilab_human/cardiac_cavity_intersections.py'
sys.path.insert(0,'/Users/n/numi-human-anatomy-finish-1378/src')
spec=importlib.util.spec_from_file_location('numilab_human.exact_root011',EXACT)
ci=importlib.util.module_from_spec(spec);sys.modules[spec.name]=ci;spec.loader.exec_module(ci)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def classify(a,b):
 points=tuple(ci.triangle_intersection_points(a,b))
 if not points:return 'clear',points
 na=ci._cross(ci._sub(a[1],a[0]),ci._sub(a[2],a[0]))
 nb=ci._cross(ci._sub(b[1],b[0]),ci._sub(b[2],b[0]))
 common=set(a)&set(b)
 if len(common)==3:return 'coincident_triangle',points
 if len(points)==1:return 'point_contact',points
 if len(common)==2 and all(ci._allowed_shared_point(p,common) for p in points):
  return 'coincident_edge_contact',points
 coplanar=not any(ci._cross(na,nb)) and ci._dot(na,ci._sub(b[0],a[0]))==0
 if coplanar:
  area=ci._cross(ci._sub(points[1],points[0]),ci._sub(points[2],points[0])) if len(points)>2 else (0,0,0)
  return ('coplanar_area_overlap' if any(area) else 'coplanar_segment_contact'),points
 sa=[ci._dot(na,ci._sub(p,a[0])) for p in b]
 sb=[ci._dot(nb,ci._sub(p,b[0])) for p in a]
 if min(sa)<0<max(sa) and min(sb)<0<max(sb):return 'transverse_both_interiors',points
 return 'noncoplanar_edge_face_segment',points
def records(v,f,ids):
 ids=np.asarray(ids,np.int64);r,_=ci.float32_surface_records(v,np.asarray(f,np.int64)[ids])
 return {int(i):(x[0],x[1],x[2],int(i),x[4]) for i,x in zip(ids,r)}
def main():
 out=HERE/'run-001';assert not out.exists();out.mkdir()
 begun=time.monotonic()
 inp=D/'validation-001/zero-support-physical-O-face-pairs-010.npz'
 mesh=E/'global-inner-p1-active-layer-refinement-008/candidate-with-transfer-partial-field.npz'
 bank=E/'registered-surrounding-tissue-native013-191/registered-tissue-banks.npz'
 invp=bank.with_name('inventory.json')
 assert sha(inp)=='4ac5c272ec0b006871d2a81d4b0133a53f2e5517d0d80a9a42458360500f89d2'
 z=np.load(inp);m=np.load(mesh);banks=np.load(bank);inv=json.loads(invp.read_text())['banks']
 X=m['points_world_m'];O=m['physical_outer_faces'];orows=records(X,O,np.unique(z['physical_O_face_ids']))
 codes=[];classes=[];perbank=[];points_match=[];canonical_pair_rows=[];canonical_bank_ids=[];rowids=[]
 for bi in np.unique(z['bank_ids']):
  selected=np.flatnonzero(z['bank_ids']==bi);meta=inv[int(bi)]
  br=records(banks[meta['key']+'_vertices'],banks[meta['key']+'_faces'],np.unique(z['bank_local_face_ids'][selected]))
  pairs=[(br[int(z['bank_local_face_ids'][i])],orows[int(z['physical_O_face_ids'][i])]) for i in selected]
  canonical=ci._audit_record_pairs(iter(pairs),same_surface=False)
  got=set(map(tuple,canonical['triangle_pairs']))
  typed=Counter()
  for i,(a,b) in zip(selected,pairs):
   cls,points=classify(a[0],b[0]);hit=bool(points)
   assert hit==((a[3],b[3]) in got),'canonical pair gate disagrees with classification'
   # Record rounded metre points only as diagnostics; the predicate used exact integer/rational coordinates.
   world=sorted(tuple(float(v/(1<<149)) for v in p) for p in points)
   lo,hi=z['exact_hit_offsets'][i:i+2];old=sorted(map(tuple,z['exact_hit_points_m'][lo:hi]))
   points_match.append(world==old)
   if cls not in classes:classes.append(cls)
   codes.append(classes.index(cls));rowids.append(int(i));typed[cls]+=1
   if hit:canonical_pair_rows.append([a[3],b[3]]);canonical_bank_ids.append(int(bi))
  perbank.append({'bank_id':int(bi),'key':meta['key'],'kind':meta['kind'],'stable_id':meta['stable_id'],'layer':meta['layer'],'pairs':len(selected),'canonical_hits':len(got),'classes':dict(typed)})
 order=np.argsort(rowids);codes=np.asarray(codes,np.uint8)[order]
 np.savez_compressed(out/'canonical-pair-classification.npz',bank_ids=z['bank_ids'],bank_local_face_ids=z['bank_local_face_ids'],
  physical_O_face_ids=z['physical_O_face_ids'],class_codes=codes,class_names=np.asarray(classes),
  canonical_hit_bank_ids=np.asarray(canonical_bank_ids,np.uint16),canonical_hit_face_pairs=np.asarray(canonical_pair_rows,np.int64))
 dep={str(EXACT):sha(EXACT)}
 for name in ('cardiac_cavity_geometry','model','physiology'):
  module=sys.modules['numilab_human.'+name];dep[str(module.__file__)]=sha(module.__file__)
 report={'scope':'exact final predicate on construction-suggested pairs only; no global solid containment/self certification',
  'status':'PASS_CANONICAL_PAIR_CLASSIFICATION','pair_count':len(codes),'canonical_hit_count':len(canonical_pair_rows),
  'classes':dict(Counter(classes[int(c)] for c in codes)),'all_prior_rounded_hit_points_identical':all(points_match),
  'prior_rounded_hit_point_mismatches':int(np.sum(~np.asarray(points_match,bool))),
  'per_bank':perbank,'input_sha256':{str(p):sha(p) for p in (inp,mesh,bank,invp,Path(__file__))},
  'predicate_dependencies':dep,'elapsed_s':time.monotonic()-begun,'physical_admission':False,'native_admission':False,
  'anatomical_interface_exemptions_applied':False}
 (out/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k not in ('per_bank','input_sha256','predicate_dependencies')}),flush=True)
if __name__=='__main__':main()
