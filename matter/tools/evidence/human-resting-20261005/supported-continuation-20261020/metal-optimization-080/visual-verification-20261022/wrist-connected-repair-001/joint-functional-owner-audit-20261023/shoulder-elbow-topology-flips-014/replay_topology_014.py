"""Replay the cumulative retained 014 face patch against its exact NHSKIN source.

No vertex, bind, weight, or physical state changes. The retained report still
has87 elbow crossings; successful byte replay is not anatomical admission.
"""
from pathlib import Path
import hashlib,importlib.util,json,struct
import numpy as np
R=Path('/Users/n/numi-human-aortic-contact-opt-20261019');E=R/'matter/tools/evidence/human-resting-20261005/supported-continuation-20261020/metal-optimization-080/visual-verification-20261022/wrist-connected-repair-001';O=E/'joint-functional-owner-audit-20261023/shoulder-elbow-topology-flips-014'
S=E/'left-ankle-rest-preserving-weight-repair-128/candidate.nhskin';A=O/'topology-flip-ancestry.npz'
sha=lambda b:hashlib.sha256(b).hexdigest()
raw=S.read_bytes();assert sha(raw)=='4fd80f449b2c4e6a94b0808c2e54a5735684711de1f7fc1c34e893c2494250a5'
assert sha(A.read_bytes())=='815a3cd1b8fa599399ff897673c23c3c1a7f3ab9859c26ec29738b77071ce325'
spec=importlib.util.spec_from_file_location('rebind_owner',R/'tools/rebind_human_skin_reference.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);p=m.decode(raw)
z=np.load(A);rows=z['modified_face_rows'];faces=z['final_faces_for_modified_rows'];assert rows.shape==(14,) and faces.shape==(14,3);assert len(set(map(int,rows)))==14;assert np.all((rows>=0)&(rows<len(p['inds']))) and np.all((faces>=0)&(faces<p['nv']))
blob=bytearray(raw)
for row,face in zip(rows,faces):struct.pack_into('<3I',blob,p['indexoff']+12*int(row),*map(int,face))
assert sha(blob)=='860a7ed05b142c03c1c4854f9803ff40a10255e3df9cf841681d976d4efb6782'
q=m.decode(bytes(blob));assert p['binds'].tobytes()==q['binds'].tobytes() and p['verts'].tobytes()==q['verts'].tobytes() and p['weights'].tobytes()==q['weights'].tobytes()
report={'source128_sha256':sha(raw),'replayed014_sha256':sha(blob),'cumulative_face_rows':14,'vertices_bindings_weights_byte_identical':True,'physical_admission':False,'remaining_exact_crossings':{'native_rest':2,'elbow':87,'shoulder_elevation':27,'shoulder_plane':49}}
(O/'compact-replay-verification-014.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))
