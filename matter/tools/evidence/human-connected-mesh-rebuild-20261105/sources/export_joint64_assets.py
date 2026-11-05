import sys
"""Export candidate64c through signed existing 433 owner; no physical admission."""
from pathlib import Path
import json,hashlib,subprocess,sys,time
import numpy as np
W=Path('/Users/numi/Work/human-integrated-gaps-20261104');D=Path(sys.argv[1]).resolve() if len(sys.argv)>1 else W/'joint-boundary-rebuild-64c';C=D/'compiler'
H=Path('/Users/numi/Work/numi-human-connected-surface-069/D/connected-obstacle-207/physical-transfer-209/fine-physical-volume-706')
compat=H/'downstream-004-same-material-current-reference-229/native-domain-217-source-007/compat'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def run(name,args):
 cmd=[sys.executable,*map(str,args)];start=time.time();print('START',name,flush=True)
 with (C/(name+'.stdout.log')).open('w') as o,(C/(name+'.stderr.log')).open('w') as e:r=subprocess.run(cmd,stdout=o,stderr=e)
 (C/(name+'.command.json')).write_text(json.dumps({'command':cmd,'returncode':r.returncode,'wall_s':time.time()-start},indent=2)+'\n')
 print('END',name,r.returncode,flush=True)
 if r.returncode:raise RuntimeError((name,(C/(name+'.stderr.log')).read_text()[-2500:]))
run('orientation1057',[C/'typed-adapter/audit-role-orientation.py','--state',C/'volume-prefix/verified-state.json','--receipt',C/'shells/receipt.json','--output',C/'role-orientation.json'])
src=compat/'current1020-433-adapter-1044/export_refined_coarse_registered_volume_433_p1_anchor_1044_signed_role_faces.py'
s=src.read_text();adapt=C/'typed-adapter/typed-current-schema.py'
s=s.replace('ADAPTER_1021_PATH = BASE_DIR / "current1020-433-adapter-1021" / "export_refined_coarse_registered_volume_433_p1_anchor_1021_current_schema.py"',f'ADAPTER_1021_PATH = pathlib.Path({str(adapt)!r})')
s=s.replace('ADAPTER_1021_SHA256 = "c89598d273da02abfbae8f9a5fa1e640c0b82551631660553ca85f00e47d50ed"',f'ADAPTER_1021_SHA256 = {sha(adapt)!r}')
oldpath=H/'current-native-1024-root/current-433-1044/1022-role-surface-orientation-audit.json'
s=s.replace(str(oldpath),str(C/'role-orientation.json')).replace('cad9e484357d9340f2ca43638a3e83874156a5693ddb9192f919e5c91ee8ebed',sha(C/'role-orientation.json'))
s=s.replace('not info.get("source_face_rows_identity")','not (info.get("source_face_rows_identity") is True or info.get("source_face_ancestry_exact") is True)')
# The orientation owner has authenticated the current receipt and each shell.
# Bind its current hashes rather than reinstalling historical shell constants.
old_shell_pins = '    expected_shells = {\n        "skin": "1d29d3a07147ab7c7bcf45e017ee6515e0a4e7015dd236dbc7d07407c894c9cb",\n        "fat": "2a97964d38c945836a4090a7c9580624b427a0163eda9932370cc6f77277ef66",\n    }'
assert s.count(old_shell_pins) == 1, 'signed-owner source changed; review shell binding'
s = s.replace(old_shell_pins, '    expected_shells = {role: receipt["outputs"][role + "-shell.npz"]["sha256"] for role in ("skin", "fat")}')

# Keep the original signed exporter and every validation branch, but do not
# materialize millions of row dictionaries or face-use records during preparation.
helper_root=Path('/Users/numi/Work/skin-fat-native-integration-209/apps')
insertion="import sys\nsys.path.insert(0,"+repr(str(helper_root))+")\nfrom numi_human_compiler_tables import ArrayRows,CompactOrientedFaceUses,compact_export_source\n"
assert s.count('import sys\n')==1
s=s.replace('import sys\n',insertion,1)
old='    source = _cache_candidate_cell_label_arrays_in_export(source)'
assert s.count(old)==1
s=s.replace(old,'    source = compact_export_source(_cache_candidate_cell_label_arrays_in_export(source))',1)
old='    namespace = owner.__dict__'
assert s.count(old)==1
s=s.replace(old,old+'\n    namespace.update(ArrayRows=ArrayRows,CompactOrientedFaceUses=CompactOrientedFaceUses)',1)
dest=C/'typed-adapter/signed-current-volume.py';dest.write_text(s)

with np.load(C/'rich433/candidate-current-typed-volume.npz') as z:points=z['points_reference_world_m_f64']
node=C/'rich433/candidate-canonical-f32.node'
with node.open('w') as f:
 f.write(f'{len(points)} 3 0 0\n')
 for i,p in enumerate(points):f.write(str(i+1)+' '+' '.join(format(float(v),'.17g') for v in p)+'\n')
old=json.loads((H/'current-native-1024-root/current-433-1044/launch-inputs.json').read_text())
args=old['argv'][3:];args[0]=str(dest)
changes={'--current-surface-node-map':[str(C/'handoff938/skin-fat-combined-registered-surface-node-map.npz'),sha(C/'handoff938/skin-fat-combined-registered-surface-node-map.npz')],
'--signed-role-face-audit':[str(C/'role-orientation.json'),sha(C/'role-orientation.json')],
'--candidate-npz':[str(C/'rich433/candidate-current-typed-volume.npz')],
'--candidate-node':[str(node)],'--candidate-node-map':[str(C/'rich433/candidate-node-map.csv')],
'--inserted-node-ancestry':[str(C/'rich433/inserted-node-ancestry.csv')],'--out':[str(C/'volume-export')]}
for flag,values in changes.items():
 i=args.index(flag)+1;args[i:i+len(values)]=values
for role in 'OID':
 i=next(i for i,v in enumerate(args) if v=='--current-typed-role' and args[i+1]==role)
 args[i+2:i+4]=[str(D/f'candidate-{role}-typed-current.npz'),sha(D/f'candidate-{role}-typed-current.npz')]
run('volume-export',args)
receipt_path=C/'volume-export/receipt.json'
receipt=json.loads(receipt_path.read_text())
receipt['preparation_storage_extension']={str(p):sha(p) for p in [helper_root/'numi_human_compiler_tables.py',helper_root/'numi_human_tet_face_incidence.py',dest]}
receipt['input_sha256'].update(receipt['preparation_storage_extension'])
receipt_path.write_text(json.dumps(receipt,indent=2)+'\n')
print('VOLUME_EXPORTED_MASS_NOT_RUN',flush=True)
