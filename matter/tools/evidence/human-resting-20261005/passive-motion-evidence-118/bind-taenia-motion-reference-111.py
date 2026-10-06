"""Bind retained exact audit arrays to the actual current NHANAT source."""
from pathlib import Path
import collections,hashlib,json,sys,numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005');out=E/'taenia-motion-reference-binding-111';out.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/n/numi-human-resting-launcher-source-001/src')
from numilab_human.resting_pleura_proxy import _parse_payload,_record_content_bytes
from numilab_human.resting_taenia_reference import build_candidate
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=E/'passive-organ-native-input-074';payload=base/'resting-thorax.nhanatomy';receipt_path=base/'resting-anatomy-receipt.json'
receipt=json.loads(receipt_path.read_text());_,_,rows=_parse_payload(payload.read_bytes())
passive=receipt['functional_bindings']['passive_viscera_geometry_binding']['stable_ids']
source_npz=E/'passive-neighbor-depth-062/source-surfaces.npz';original=np.load(source_npz);checks=[]
for sid in passive:
    if sid==457:continue
    if sid in (3,13):
        d=np.load(E/f'passive-organ-interface-conditioned-071/surface-{sid}.npz');v,f=d['vertices'],d['faces']
    else:v,f=original[f'v{sid}'],original[f'f{sid}']
    assert np.array_equal(rows[sid]['vertices6'][:,:3],v) and np.array_equal(rows[sid]['faces'],f),('audit source differs',sid)
    checks.append(sid)
assert np.array_equal(rows[457]['vertices6'][:,:3],original['v457']) and np.array_equal(rows[457]['faces'],original['f457'])
source_weld=E/'taenia-mesocolica-exact-weld-001/source-component.npz';d=np.load(source_weld)
def oriented_nonzero_faces(v,f):
    points=v[f].astype(float);cross=np.cross(points[:,1]-points[:,0],points[:,2]-points[:,0]);keep=np.any(cross!=0,axis=1)
    def key(p):
        p=tuple(tuple(map(float,x)) for x in p);return min(p,p[1:]+p[:1],p[2:]+p[:2])
    return collections.Counter(key(p) for p in points[keep]),int(np.sum(~keep))
before,removed=oriented_nonzero_faces(rows[457]['vertices6'][:,:3],rows[457]['faces']);after,zero=oriented_nonzero_faces(d['vertices'],d['faces'])
assert before==after and removed==2 and zero==0
audit_path=E/'taenia-motion-margin-audit-103/report.json';audit=json.loads(audit_path.read_text())
audit.update(source_payload_sha256=sha(payload),source_record_sha256={str(s):hashlib.sha256(_record_content_bytes(rows[s])).hexdigest() for s in passive},
             exact_audit_sha256=sha(audit_path),source_weld_sha256=sha(source_weld),
             source_weld_record_sha256=hashlib.sha256(_record_content_bytes(rows[457])).hexdigest(),
             source_binding_proof={'driver_sha256':sha(__file__),'audited_neighbor_positions_and_indices_exactly_match_current_payload':checks,
                                   'audited_reference_source_npz_sha256':sha(source_npz),'source_weld_preserves_all_oriented_nondegenerate_coordinate_triangles':True,
                                   'removed_exactly_degenerate_source_faces':removed})
bound=out/'source-bound-exact-audit.json';bound.write_text(json.dumps(audit,indent=2)+'\n')
args={'base_payload':payload,'base_receipt':receipt_path,'candidate_path':E/'taenia-motion-margin-audit-103/surface-457.npz',
      'audit_path':bound,'shape_path':E/'taenia-motion-margin-shape-106/report.json','interface_path':E/'taenia-motion-interface-classification-108/report.json',
      'output':E/'taenia-motion-native-input-111'}
(out/'invocation.json').write_text(json.dumps({'function':'numilab_human.resting_taenia_reference.build_candidate','arguments':{k:str(v) for k,v in args.items()},
                                             'driver_sha256':sha(__file__)},indent=2)+'\n')
result=build_candidate(**args);print(json.dumps(result['payload']),flush=True)
