"""Bind the exact bladder audit to the current integrated source records."""
from pathlib import Path
import hashlib,json,sys,numpy as np

E=Path('/Users/n/numi-human-resting-evidence-20261005');out=E/'bladder-reference-binding-112';out.mkdir(exist_ok=False)
sys.path.insert(0,'/Users/n/numi-human-resting-launcher-source-001/src')
from numilab_human.resting_pleura_proxy import _parse_payload,_record_content_bytes
from numilab_human.resting_passive_interfaces import build_candidate
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
base=E/'taenia-motion-native-input-111';payload=base/'resting-thorax.nhanatomy';receipt_path=base/'resting-anatomy-receipt.json'
receipt=json.loads(receipt_path.read_text());_,_,rows=_parse_payload(payload.read_bytes())
passive=receipt['functional_bindings']['passive_viscera_geometry_binding']['stable_ids']
source_npz=E/'passive-neighbor-depth-062/source-surfaces.npz';original=np.load(source_npz);checks=[]
for sid in passive:
    if sid in (3,13):
        d=np.load(E/f'passive-organ-interface-conditioned-071/surface-{sid}.npz');v,f=d['vertices'],d['faces']
    elif sid==457:
        d=np.load(E/'taenia-motion-margin-audit-103/surface-457.npz');v,f=d['vertices'],d['faces']
    else:v,f=original[f'v{sid}'],original[f'f{sid}']
    assert np.array_equal(rows[sid]['vertices6'][:,:3],v) and np.array_equal(rows[sid]['faces'],f),('audit source differs',sid)
    checks.append(sid)
audit_path=E/'bladder-interface-audit-109/report.json';audit=json.loads(audit_path.read_text())
audit.update(source_payload_sha256=sha(payload),source_record_sha256={str(s):hashlib.sha256(_record_content_bytes(rows[s])).hexdigest() for s in passive},
             exact_audit_sha256=sha(audit_path),candidate_sha256={'462':audit['candidate_sha256']},self={'462':audit['self']},
             source_binding_proof={'driver_sha256':sha(__file__),'audited_positions_and_indices_exactly_match_current_payload':checks,
                                   'audited_reference_source_npz_sha256':sha(source_npz)})
bound=out/'source-bound-exact-audit.json';bound.write_text(json.dumps(audit,indent=2)+'\n')
args={'base_payload':payload,'base_receipt':receipt_path,
      'reference_payload':E/'costal-current003-full-chain-003/resting-thorax.nhanatomy',
      'candidate_dir':E/'bladder-interface-audit-109','audit_path':bound,
      'derivation_path':E/'bladder-interface-candidate-105-50um/report.json',
      'interface_path':E/'bladder-neck-interface-110/report.json','organ_set':'bladder',
      'output':E/'bladder-native-input-112'}
(out/'invocation.json').write_text(json.dumps({'function':'numilab_human.resting_passive_interfaces.build_candidate','arguments':{k:str(v) for k,v in args.items()},
                                             'driver_sha256':sha(__file__)},indent=2)+'\n')
result=build_candidate(**args);print(json.dumps(result['payload']),flush=True)
