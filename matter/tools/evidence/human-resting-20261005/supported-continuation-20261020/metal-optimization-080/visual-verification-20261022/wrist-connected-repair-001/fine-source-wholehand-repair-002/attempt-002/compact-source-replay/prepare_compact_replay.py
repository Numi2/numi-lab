from pathlib import Path
import sys,json,hashlib
import numpy as np
E=Path('/Users/n/numi-human-aortic-contact-opt-20261019/matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001')
sys.path.insert(0,str(E/'registered-pose-frame-helper-194/tools'))
from rebind_human_skin_reference import decode
O=E/'fine-source-wholehand-repair-002/attempt-002/compact-source-replay';O.mkdir(exist_ok=True)
P0=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin'
P1=O.parent/'candidate.nhskin'
a=P0.read_bytes();b=P1.read_bytes();da=decode(a);db=decode(b)
sha=lambda x:hashlib.sha256(x).hexdigest()
assert sha(a)=='4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5'
assert sha(b)=='24ee1aa733d82cce0fa2bbc6a603eeb8fa102e67d712c8457a1057213a049050'
off=da['vertoff'];end=da['indexoff']
assert a[:off]==b[:off] and a[end:]==b[end:]
A=np.frombuffer(a[off:end],np.uint8).reshape(-1,56);B=np.frombuffer(b[off:end],np.uint8).reshape(-1,56)
ids=np.flatnonzero(np.any(A!=B,axis=1))
np.savez_compressed(O/'candidate-vertex-records.npz',
 changed_source_vertex_ids=ids.astype(np.uint32),changed_vertex_record_bytes=B[ids],
 baseline_payload_sha256=np.array(sha(a)),candidate_payload_sha256=np.array(sha(b)),
 payload_vertex_count=np.array(da['nv'],np.uint32),vertex_record_byte_offset=np.array(off,np.uint64))
z=np.load(O/'candidate-vertex-records.npz',allow_pickle=False)
restored=bytearray(a)
for i,row in zip(z['changed_source_vertex_ids'],z['changed_vertex_record_bytes']):
 s=off+int(i)*56;restored[s:s+56]=row.tobytes()
assert bytes(restored)==b
receipt={'status':'exact-byte-replay-pass','candidate_geometry_status':'failed exact surface/bone gate, not accepted skin',
 'baseline_payload_sha256':sha(a),'candidate_payload_sha256':sha(b),'source_bytes':len(b),
 'changed_vertex_records':len(ids),'compressed_bytes':(O/'candidate-vertex-records.npz').stat().st_size,
 'compressed_sha256':sha((O/'candidate-vertex-records.npz').read_bytes()),
 'bindings_weights_topology_and_outside_records_unchanged':True}
(O/'replay-receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
